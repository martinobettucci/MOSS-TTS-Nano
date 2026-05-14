"""Bundled reference voice lookup for local voice cloning."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

DEFAULT_REFERENCE_VOICE_LANGUAGE = "en"
REFERENCE_VOICE_GENDERS = frozenset({"male", "female"})


def normalize_reference_voice_gender(gender: str | None) -> str:
    normalized = str(gender or "male").strip().lower()
    if normalized not in REFERENCE_VOICE_GENDERS:
        raise ValueError(f"Unsupported reference voice gender: {gender!r}")
    return normalized


def normalize_reference_voice_language(language: str | None) -> str:
    normalized = str(language or DEFAULT_REFERENCE_VOICE_LANGUAGE).strip().lower().replace("_", "-")
    if not normalized:
        return DEFAULT_REFERENCE_VOICE_LANGUAGE
    return normalized.split("-", 1)[0]


def _package_reference_voice_path(language: str, gender: str) -> Path | None:
    try:
        resource = resources.files("moss_tts_nano").joinpath(
            "assets",
            "reference_voices",
            language,
            f"{gender}.wav",
        )
        if resource.is_file():
            with resources.as_file(resource) as resolved:
                return Path(resolved)
    except Exception:
        return None
    return None


def _source_tree_reference_voice_path(language: str, gender: str) -> Path | None:
    candidate = Path(__file__).resolve().parents[1] / "assets" / "reference_voices" / language / f"{gender}.wav"
    if candidate.is_file():
        return candidate
    return None


def resolve_reference_voice_path(
    *,
    language: str | None,
    gender: str | None = None,
) -> Path:
    resolved_gender = normalize_reference_voice_gender(gender)
    requested_language = normalize_reference_voice_language(language)
    for candidate_language in (requested_language, DEFAULT_REFERENCE_VOICE_LANGUAGE):
        for resolver in (_package_reference_voice_path, _source_tree_reference_voice_path):
            path = resolver(candidate_language, resolved_gender)
            if path is not None:
                return path
    raise FileNotFoundError(
        f"No bundled {resolved_gender} reference voice for language={requested_language!r} "
        f"or fallback language={DEFAULT_REFERENCE_VOICE_LANGUAGE!r}."
    )
