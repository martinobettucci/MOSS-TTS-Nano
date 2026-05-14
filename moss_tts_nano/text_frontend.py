"""Harmonized language-specific text frontend for TTS training and inference.

Routing policy (single source of truth, applied at every inference entry point):

  - en, zh                       → WeText + robust normalization (no phonemizer)
  - 17 other supported languages → espeak-ng IPA or ASCII phoneme tokens
                                   (preceded by MOSS robust normalization)
  - unknown / empty              → raw passthrough

The harmonized entry point is :func:`apply_harmonized_frontend`. Every TTS
entry point (CLI, direct PyTorch infer, Gradio app, ONNX app) calls it exactly
once before forwarding text to the model. Runtime methods MUST NOT re-process
text; they receive the final string.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from .phoneme_ascii import (
    INVENTORY_VERSION,
    MEDIUM_PAUSE_TOKEN,
    SENT_DECL_TOKEN,
    SENT_EXCL_TOKEN,
    SENT_Q_TOKEN,
    SHORT_PAUSE_TOKEN,
    WORD_TOKEN,
    ipa_to_ascii_text,
)

logger = logging.getLogger(__name__)

# Phonemizer emits per-utterance WARNING noise (language-switch flags removed,
# extra phones, words-count mismatches) on every French sample that contains a
# foreign-looking token — brand names, anglicisms — which is extremely common
# in MLS/CV17 transcripts. At 700K+ samples this floods Step 3 logs. The
# warnings narrate behavior we *want*: force the configured voice, ignore
# language switches, keep one phoneset per language for train/inference parity.
# We give phonemizer its own logger pinned to ERROR so real failures still
# surface but per-utterance chatter is dropped.
_PHONEMIZER_LOGGER = logging.getLogger("phonemizer.silent")
_PHONEMIZER_LOGGER.setLevel(logging.ERROR)
_PHONEMIZER_LOGGER.addHandler(logging.NullHandler())
_PHONEMIZER_LOGGER.propagate = False


# ---------------------------------------------------------------------------
# Language tables
# ---------------------------------------------------------------------------

# espeak-ng voice names per project language code (19 target languages).
ESPEAK_VOICE_BY_LANG: Dict[str, str] = {
    "en": "en-us",
    "fr": "fr-fr",
    "es": "es",
    "it": "it",
    "ar": "ar",
    "de": "de",
    "nl": "nl",
    "pt-br": "pt-br",
    "pt-pt": "pt",
    "pt": "pt",
    "pl": "pl",
    "tr": "tr",
    "ro": "ro",
    "cs": "cs",
    "hu": "hu",
    "sv": "sv",
    "no": "nb",
    "nb": "nb",
    "da": "da",
    "fi": "fi",
    "el": "el",
}

# Languages whose normalization is delegated to MOSS WeText (no IPA).
WETEXT_LANGS = frozenset({"en", "zh"})

# Languages handled by the IPA branch (intersection of supported voices and non-WeText).
IPA_LANGS = frozenset(set(ESPEAK_VOICE_BY_LANG) - WETEXT_LANGS)


# ---------------------------------------------------------------------------
# Compatibility shape kept for the legacy `prepare_text_for_tts` callers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrontendText:
    text: str
    original_text: str
    mode: str
    backend: str
    language: str


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_lang_code(lang: Optional[str]) -> str:
    """Lowercase + strip + underscore→hyphen so codes like ``pt_BR`` match ``pt-br``."""
    return str(lang or "").strip().lower().replace("_", "-")


def frontend_enabled(config: Mapping[str, Any] | None) -> bool:
    if not config:
        return False
    return bool(config.get("enabled", False))


# ---------------------------------------------------------------------------
# Phonemizer backend (cached, library-based — no subprocess per call)
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, EspeakBackend] = {}
_BACKENDS_LOCK = threading.Lock()

# `phonemizer.Separator` refuses identical separators. We use a sentinel for
# word boundaries and then render it as a structural token so word breaks remain
# distinct from phone-level spaces in the final text.
_WORD_MARK = "__W__"
_WORD_SEPARATOR = WORD_TOKEN
_PHONEMIZER_SEPARATOR = Separator(phone=" ", word=f" {_WORD_MARK} ", syllable="")
_PUNCTUATION_RE = re.compile(r"([,;:.!?…]+)")


def _get_backend(voice: str) -> EspeakBackend:
    backend = _BACKENDS.get(voice)
    if backend is not None:
        return backend
    with _BACKENDS_LOCK:
        backend = _BACKENDS.get(voice)
        if backend is None:
            backend = EspeakBackend(
                language=voice,
                with_stress=True,
                language_switch="remove-flags",
                words_mismatch="ignore",
                logger=_PHONEMIZER_LOGGER,
            )
            _BACKENDS[voice] = backend
    return backend


def _phonemize(text: str, lang: str) -> str:
    """Convert text to spaced IPA using the cached phonemizer backend."""
    original = str(text or "").strip()
    if not original:
        return ""
    language = normalize_lang_code(lang)
    voice = ESPEAK_VOICE_BY_LANG.get(language)
    if voice is None:
        raise ValueError(f"No espeak-ng voice configured for language '{lang}'")
    backend = _get_backend(voice)
    raw = backend.phonemize([original], separator=_PHONEMIZER_SEPARATOR, strip=True)[0]
    cleaned = raw.replace(_WORD_MARK, f" {_WORD_SEPARATOR} ")
    cleaned = re.sub(r"\s*[-‐‑‒–—]\s*", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _punctuation_pause_token(punctuation: str) -> str:
    if "?" in punctuation:
        return SENT_Q_TOKEN
    if "!" in punctuation:
        return SENT_EXCL_TOKEN
    if any(char in punctuation for char in ".…"):
        return SENT_DECL_TOKEN
    if any(char in punctuation for char in ";:"):
        return MEDIUM_PAUSE_TOKEN
    return SHORT_PAUSE_TOKEN


_SENT_FINAL_TOKENS = (SENT_DECL_TOKEN, SENT_Q_TOKEN, SENT_EXCL_TOKEN)


def _phonemize_ascii(text: str, lang: str) -> tuple[str, str]:
    """Return `(ascii_tokens, ipa_debug)` while preserving punctuation pauses.

    Guarantees a sentence-final token (`<sent_decl>` by default) at the end of
    non-empty output so train and inference share a consistent EOS-context signal
    regardless of whether the source text carried terminal punctuation.
    """
    parts = _PUNCTUATION_RE.split(str(text or ""))
    ascii_chunks: list[str] = []
    ipa_chunks: list[str] = []
    for part in parts:
        if not part:
            continue
        if _PUNCTUATION_RE.fullmatch(part):
            pause = _punctuation_pause_token(part)
            ascii_chunks.append(pause)
            ipa_chunks.append(pause)
            continue
        ipa = _phonemize(part, lang)
        if not ipa:
            continue
        ascii_text = ipa_to_ascii_text(ipa)
        if ascii_text:
            ascii_chunks.append(ascii_text)
            ipa_chunks.append(ipa)
    if ascii_chunks and ascii_chunks[-1] not in _SENT_FINAL_TOKENS:
        ascii_chunks.append(SENT_DECL_TOKEN)
        ipa_chunks.append(SENT_DECL_TOKEN)
    return " ".join(ascii_chunks).strip(), " ".join(ipa_chunks).strip()


# Legacy alias kept so existing callers and tests that import ``espeak_ipa``
# keep working. The implementation is now library-backed, not subprocess-backed.
def espeak_ipa(text: str, lang: str) -> str:
    return _phonemize(text, lang)


# ---------------------------------------------------------------------------
# MOSS pipeline bridge (lazy import — keeps imports cheap for unit tests)
# ---------------------------------------------------------------------------

_MOSS_PREPARE: Optional[Any] = None
_MOSS_IMPORT_TRIED = False


def _moss_prepare_tts_request_texts():
    """Return the MOSS WeText pipeline entrypoint, or ``None`` if unavailable."""
    global _MOSS_PREPARE, _MOSS_IMPORT_TRIED
    if _MOSS_PREPARE is not None or _MOSS_IMPORT_TRIED:
        return _MOSS_PREPARE
    _MOSS_IMPORT_TRIED = True
    moss_dir = Path(__file__).resolve().parent.parent
    if moss_dir.is_dir() and str(moss_dir) not in sys.path:
        sys.path.insert(0, str(moss_dir))
    try:
        from text_normalization_pipeline import prepare_tts_request_texts  # type: ignore
    except Exception as exc:
        logger.debug("MOSS text_normalization_pipeline unavailable: %s", exc)
        return None
    _MOSS_PREPARE = prepare_tts_request_texts
    return _MOSS_PREPARE


def _raw_dict(text: str, prompt_text: str, language: str, method: str) -> Dict[str, Any]:
    return {
        "text": text,
        "prompt_text": prompt_text,
        "normalized_text": text,
        "normalized_prompt_text": prompt_text,
        "normalization_method": method,
        "text_normalization_language": language,
        "text_normalization_enabled": False,
        "wetext_processing_enabled": False,
        "normalize_tts_text_enabled": False,
        "ipa_backend": None,
        "language": language,
    }


# ---------------------------------------------------------------------------
# Single inference boundary
# ---------------------------------------------------------------------------

def apply_harmonized_frontend(
    *,
    text: str,
    prompt_text: str = "",
    language: Optional[str] = None,
    voice: str = "",
    text_normalizer_manager: Optional[Any] = None,
    enable_wetext: bool = True,
    enable_normalize_tts_text: bool = True,
    frontend_mode: str = "ipa",
) -> Dict[str, Any]:
    """Single text-frontend boundary for all inference paths.

    Routing:
      * ``language`` in :data:`WETEXT_LANGS` → MOSS WeText + robust normalization
      * ``language`` in :data:`IPA_LANGS`    → MOSS robust normalization, then espeak-ng IPA
      * ``frontend_mode="phoneme_ascii"``    → espeak-ng IPA, then ASCII phoneme tokens
      * otherwise → raw passthrough

    Returns a dict shaped like :func:`text_normalization_pipeline.prepare_tts_request_texts`
    so call-sites can be migrated without changing downstream code. Extra keys:

    * ``ipa_backend`` — ``"phonemizer/espeak-ng"`` if IPA was applied, else ``None``
    * ``language``    — the resolved language code (may be empty for unknown)
    """
    lang_code = normalize_lang_code(language)
    raw_text = str(text or "")
    raw_prompt_text = str(prompt_text or "")
    requested_mode = str(frontend_mode or "ipa").strip().lower()

    moss_prepare = _moss_prepare_tts_request_texts()

    if not lang_code:
        # No language hint: defer entirely to MOSS pipeline (it infers en/zh).
        if moss_prepare is not None:
            return moss_prepare(
                text=raw_text,
                prompt_text=raw_prompt_text,
                voice=voice,
                language=None,
                enable_wetext=enable_wetext,
                enable_normalize_tts_text=enable_normalize_tts_text,
                text_normalizer_manager=text_normalizer_manager,
            )
        return _raw_dict(raw_text, raw_prompt_text, "", "raw")

    if lang_code in WETEXT_LANGS:
        if moss_prepare is None:
            return _raw_dict(raw_text, raw_prompt_text, lang_code, "raw")
        return moss_prepare(
            text=raw_text,
            prompt_text=raw_prompt_text,
            voice=voice,
            language=lang_code,
            enable_wetext=enable_wetext,
            enable_normalize_tts_text=enable_normalize_tts_text,
            text_normalizer_manager=text_normalizer_manager,
        )

    if lang_code in IPA_LANGS:
        # Pre-normalize via the MOSS robust normalizer (WeText is skipped for these
        # languages internally), then convert to IPA via phonemizer.
        # Robust normalization is FORCED ON here regardless of the caller's flag
        # so that training preprocessing and every inference entry point feed
        # identical strings to phonemizer — otherwise the IPA output would
        # diverge on smart quotes / full-width punctuation / etc., and a model
        # trained on one normalization would mispredict on the other.
        del enable_normalize_tts_text  # intentionally ignored on this branch
        if moss_prepare is not None:
            prepared = dict(moss_prepare(
                text=raw_text,
                prompt_text=raw_prompt_text,
                voice=voice,
                language=lang_code,
                enable_wetext=False,
                enable_normalize_tts_text=True,
                text_normalizer_manager=None,
            ))
        else:
            prepared = _raw_dict(raw_text, raw_prompt_text, lang_code, "raw")

        if requested_mode == "phoneme_ascii":
            final_text, ipa_debug = _phonemize_ascii(str(prepared.get("text") or ""), lang_code)
            final_prompt, ipa_prompt_debug = (
                _phonemize_ascii(str(prepared.get("prompt_text") or ""), lang_code)
                if prepared.get("prompt_text")
                else ("", "")
            )
            prepared["ipa_text"] = ipa_debug
            prepared["ipa_prompt_text"] = ipa_prompt_debug
            prepared["phoneme_inventory_version"] = INVENTORY_VERSION
            prepared["text"] = final_text
            prepared["prompt_text"] = final_prompt
            prepared["normalized_text"] = final_text
            prepared["normalized_prompt_text"] = final_prompt
        else:
            ipa_text = _phonemize(str(prepared.get("text") or ""), lang_code)
            ipa_prompt = (
                _phonemize(str(prepared.get("prompt_text") or ""), lang_code)
                if prepared.get("prompt_text")
                else ""
            )
            prepared["text"] = ipa_text
            prepared["prompt_text"] = ipa_prompt
            prepared["normalized_text"] = ipa_text
            prepared["normalized_prompt_text"] = ipa_prompt
        method = prepared.get("normalization_method") or "raw"
        frontend_label = f"phoneme_ascii:{lang_code}" if requested_mode == "phoneme_ascii" else f"ipa:{lang_code}"
        prepared["normalization_method"] = f"{method}+{frontend_label}" if method != "raw" else frontend_label
        prepared["text_normalization_language"] = lang_code
        prepared["ipa_backend"] = "phonemizer/espeak-ng"
        prepared["text_frontend"] = "phoneme_ascii" if requested_mode == "phoneme_ascii" else "ipa"
        prepared["language"] = lang_code
        return prepared

    # Unknown but non-empty language code: do not phonemize, do not WeText.
    return _raw_dict(raw_text, raw_prompt_text, lang_code, "raw")


# ---------------------------------------------------------------------------
# Backward-compatible legacy entry point
# ---------------------------------------------------------------------------

def prepare_text_for_tts(
    text: str,
    lang: str,
    config: Mapping[str, Any] | None = None,
) -> FrontendText:
    """Legacy single-string API used by training preprocessing and evaluation.

    Routes through :func:`apply_harmonized_frontend` so the en/zh-vs-IPA policy
    is identical to inference. The ``config`` argument is honored for
    backwards-compatibility: when ``enabled=False`` the call short-circuits to
    raw passthrough regardless of language.
    """
    original = str(text or "").strip()
    language = normalize_lang_code(lang)
    cfg = dict(config or {})
    mode = str(cfg.get("mode", "raw")).strip().lower()
    backend_name = str(cfg.get("backend", "none")).strip().lower()

    if not original or not frontend_enabled(cfg) or mode in {"raw", "none"}:
        return FrontendText(original, original, "raw", "none", language)

    if mode not in {"ipa", "phoneme_ascii"} or backend_name not in {"espeak-ng", "phonemizer", "phonemizer/espeak-ng"}:
        raise ValueError(f"Unsupported text frontend mode/backend: {mode}/{backend_name}")

    prepared = apply_harmonized_frontend(
        text=original,
        language=language,
        # Preprocessing never has a WeText manager available; that's fine —
        # en/zh fall through to robust normalization without WeText.
        text_normalizer_manager=None,
        enable_wetext=True,
        enable_normalize_tts_text=True,
        frontend_mode=mode,
    )

    out_text = str(prepared.get("text") or original)
    if language in IPA_LANGS:
        return FrontendText(out_text, original, mode, "phonemizer/espeak-ng", language)
    # en / zh / unknown — raw text after WeText/robust cleanup.
    return FrontendText(out_text, original, "raw", "none", language)
