"""High-level TTS API: ``(language, checkpoint, text) -> audio``.

This is the unified inference entry point for the fork. It hides:

- **Checkpoint resolution** — accepts an integer epoch, ``"last"``, ``"0"``, an HF
  repo-id, or a filesystem path.
- **Text frontend** — phoneme-ASCII checkpoints phonemize raw text internally
  via ``model.inference_stream`` (the patched modeling code reads
  ``config.text_frontend_mode``). English/Chinese route to the upstream
  base model and use MOSS WeText normalization for safe number/punctuation
  reading.

Public surface:

- :func:`synthesize` — blocking, returns the final event dict (waveform, codes, RTF).
- :func:`synthesize_stream` — generator yielding stream events, same shape as
  :meth:`~modeling_moss_tts_nano.MossTTSNanoForCausalLM.inference_stream`.

Example::

    from moss_tts_nano import synthesize
    result = synthesize(
        language="fr",
        checkpoint="last",          # or 2, or "0", or a path
        text="Bonjour le monde",
        output_audio_path="out.wav",
        reference_audio_path="ref.wav",
    )
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import torch
from transformers import AutoModelForCausalLM

from .reference_voices import resolve_reference_voice_path

_DEFAULT_BASE_HF_ID = "OpenMOSS-Team/MOSS-TTS-Nano"


def _normalize_checkpoint_spec(checkpoint: Union[int, str, Path, None]) -> str:
    if checkpoint is None:
        return "last"
    if isinstance(checkpoint, int):
        return "0" if checkpoint == 0 else str(int(checkpoint))
    return str(checkpoint)


def _resolve_checkpoint(
    language: Optional[str],
    checkpoint: Union[int, str, Path, None],
    *,
    base_model: str,
) -> str:
    """Map (language, checkpoint) → model path or HF repo-id.

    A direct path or HF id is passed through unchanged. Numeric / "last" / "0"
    go through :func:`checkpoint_resolver.resolve_pytorch_checkpoint`.
    """
    spec = _normalize_checkpoint_spec(checkpoint)

    if Path(spec).is_dir():
        return spec
    if "/" in spec and spec not in {"0"} and not spec.lstrip("-").isdigit() and spec != "last":
        # Looks like an HF repo-id (org/name).
        return spec

    # Import inside the function so the package stays importable even when the
    # parent project's directory layout (which the resolver references) is
    # absent (e.g. when shipping the fork as a standalone wheel).
    from checkpoint_resolver import resolve_pytorch_checkpoint  # type: ignore

    return resolve_pytorch_checkpoint(language, spec, default_checkpoint=base_model)


def _resolve_onnx_checkpoint(
    language: Optional[str],
    checkpoint: Union[int, str, Path, None],
) -> str | None:
    spec = _normalize_checkpoint_spec(checkpoint)
    if Path(spec).is_dir():
        return spec
    from checkpoint_resolver import resolve_onnx_model_dir  # type: ignore

    return resolve_onnx_model_dir(language, spec)


def _load_model(
    checkpoint_path: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    attention_implementation: str = "sdpa",
) -> AutoModelForCausalLM:
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )
    if device is not None:
        model.to(device=device, dtype=dtype)
    elif dtype is not None:
        model.to(dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        try:
            model._set_attention_implementation(attention_implementation)
        except Exception:
            pass
    model.eval()
    return model


def _supports_kwarg(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _build_inference_kwargs(
    model,
    *,
    text: str,
    language: Optional[str],
    output_audio_path: Union[str, Path],
    extra: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "text": text,
        "output_audio_path": str(output_audio_path),
    }
    if language and _supports_kwarg(model.inference_stream, "language"):
        kwargs["language"] = language
    kwargs.update(extra)
    return kwargs


def _resolve_prompt_audio_kwargs(
    *,
    language: Optional[str],
    mode: str | None,
    prompt_audio_path: Any,
    reference_audio_path: Any,
    reference_voice_gender: str,
) -> tuple[Any, Any]:
    if prompt_audio_path or reference_audio_path:
        return prompt_audio_path, reference_audio_path
    if str(mode or "voice_clone").strip().lower() != "voice_clone":
        return prompt_audio_path, reference_audio_path
    return str(resolve_reference_voice_path(language=language, gender=reference_voice_gender)), None


def synthesize_stream(
    *,
    language: Optional[str],
    checkpoint: Union[int, str, Path, None],
    text: str,
    output_audio_path: Union[str, Path],
    base_model: str = _DEFAULT_BASE_HF_ID,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    model: Any = None,
    reference_voice_gender: str = "male",
    **inference_kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Stream-yield inference events.

    Use this when you want chunks as they're decoded. For a single final result,
    call :func:`synthesize`.

    Pass an already-loaded ``model`` to avoid reloading on every call.
    """
    if model is None:
        path = _resolve_checkpoint(language, checkpoint, base_model=base_model)
        model = _load_model(path, device=device, dtype=dtype)

    # The patched inference_stream runs _normalize_text_for_inference internally
    # (phoneme-ASCII or IPA + WeText) based on `config.text_frontend_mode`.
    prompt_audio_path, reference_audio_path = _resolve_prompt_audio_kwargs(
        language=language,
        mode=inference_kwargs.get("mode"),
        prompt_audio_path=inference_kwargs.get("prompt_audio_path"),
        reference_audio_path=inference_kwargs.get("reference_audio_path"),
        reference_voice_gender=reference_voice_gender,
    )
    inference_kwargs["prompt_audio_path"] = prompt_audio_path
    inference_kwargs["reference_audio_path"] = reference_audio_path
    kwargs = _build_inference_kwargs(
        model,
        text=text,
        language=language,
        output_audio_path=output_audio_path,
        extra=inference_kwargs,
    )
    yield from model.inference_stream(**kwargs)


def _looks_like_onnx_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.glob("**/*.onnx")) or (path / "browser_poc_manifest.json").is_file()


def _resolve_engine(checkpoint_path: str, engine: str) -> str:
    if engine != "auto":
        return engine
    candidate = Path(checkpoint_path)
    if candidate.exists() and _looks_like_onnx_dir(candidate):
        return "onnx"
    return "pytorch"


def _synthesize_onnx(
    *,
    language: Optional[str],
    checkpoint_path: str | None,
    text: str,
    output_audio_path: Union[str, Path],
    inference_kwargs: dict[str, Any],
    reference_voice_gender: str,
) -> dict[str, Any]:
    # Import lazily so PyTorch users don't pay onnxruntime import cost.
    from onnx_tts_runtime import OnnxTtsRuntime  # type: ignore

    runtime = OnnxTtsRuntime(model_dir=checkpoint_path)
    if not inference_kwargs.get("prompt_audio_path"):
        inference_kwargs["prompt_audio_path"] = str(
            resolve_reference_voice_path(language=language, gender=reference_voice_gender)
        )
    return runtime.synthesize(
        text=text,
        language=language,
        output_audio_path=str(output_audio_path),
        **inference_kwargs,
    )


def synthesize(
    *,
    language: Optional[str],
    checkpoint: Union[int, str, Path, None],
    text: str,
    output_audio_path: Union[str, Path],
    engine: str = "auto",
    base_model: str = _DEFAULT_BASE_HF_ID,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    model: Any = None,
    reference_voice_gender: str = "male",
    **inference_kwargs: Any,
) -> dict[str, Any]:
    """Run inference and return the final event dict (waveform, RTF, etc.).

    Args:
        language: Language code (``"fr"``, ``"de"``, ``"en"``…). ``None`` skips
            language conditioning and routes to the base model.
        checkpoint: ``int`` epoch number, ``"last"``, ``"0"`` (base), a path to a
            checkpoint dir, or an HF repo-id. For ONNX, pass a directory that
            contains ``*.onnx`` files or ``browser_poc_manifest.json``.
        text: Raw text. The engine normalizes it internally — phonemizes for
            ASCII-phoneme checkpoints, applies WeText for en/zh.
        output_audio_path: Where to write the generated WAV.
        engine: ``"auto"`` (default — detect from checkpoint path), ``"pytorch"``,
            or ``"onnx"``.
        base_model: HF repo-id of the base model (used when no finetune applies).
        device, dtype: PyTorch model placement options (ignored for ONNX).
        model: Pre-loaded PyTorch model to avoid reloading (ignored for ONNX).
        **inference_kwargs: Forwarded to the underlying inference call
            (``model.inference_stream`` for PyTorch, ``runtime.synthesize`` for ONNX).

    Returns:
        For PyTorch: the last event dict emitted by ``inference_stream``.
        For ONNX: the result dict returned by ``OnnxTtsRuntime.synthesize``.
    """
    if engine == "onnx":
        onnx_checkpoint_path = _resolve_onnx_checkpoint(language, checkpoint)
        return _synthesize_onnx(
            language=language,
            checkpoint_path=onnx_checkpoint_path,
            text=text,
            output_audio_path=output_audio_path,
            inference_kwargs=inference_kwargs,
            reference_voice_gender=reference_voice_gender,
        )

    checkpoint_path = _resolve_checkpoint(language, checkpoint, base_model=base_model)
    resolved_engine = _resolve_engine(checkpoint_path, engine)
    if resolved_engine == "onnx":
        return _synthesize_onnx(
            language=language,
            checkpoint_path=checkpoint_path,
            text=text,
            output_audio_path=output_audio_path,
            inference_kwargs=inference_kwargs,
            reference_voice_gender=reference_voice_gender,
        )

    last: Optional[dict[str, Any]] = None
    for event in synthesize_stream(
        language=language,
        checkpoint=checkpoint_path,
        text=text,
        output_audio_path=output_audio_path,
        base_model=base_model,
        device=device,
        dtype=dtype,
        model=model,
        reference_voice_gender=reference_voice_gender,
        **inference_kwargs,
    ):
        last = event
    if last is None:
        raise RuntimeError("inference_stream yielded no events")
    return last
