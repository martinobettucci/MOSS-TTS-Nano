"""Resolve finetuned checkpoint paths from the parent TTS project.

Checkpoint selection convention:
  0    -> base model (HuggingFace default, no fine-tuning)
  N    -> models/checkpoints/<lang>/final/checkpoint-epoch-N   (PyTorch)
           or  models/checkpoints/<lang>/final-onnx/checkpoint-epoch-N  (ONNX)
  last -> checkpoint-last (latest saved epoch)
"""
from __future__ import annotations

from pathlib import Path

# Two levels up from .local_deps/MOSS-TTS-Nano/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS_ROOT = PROJECT_ROOT / "models" / "checkpoints"


def add_checkpoint_args(parser) -> None:
    """Attach --lang and --checkpoint to an ArgumentParser."""
    parser.add_argument(
        "--lang",
        default=None,
        help=(
            "Language code of a finetuned checkpoint (e.g. fr, en, de). "
            "When set, --checkpoint selects which epoch; omit for the base model."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="last",
        help=(
            "Checkpoint to load when --lang is set: "
            "0 = base model, N = checkpoint-epoch-N, last = checkpoint-last (default)."
        ),
    )


def resolve_pytorch_checkpoint(
    lang: str | None,
    checkpoint_spec: str,
    *,
    default_checkpoint: str,
) -> str:
    """Return the HF repo-id or local path for a PyTorch checkpoint.

    Without --lang the checkpoint_spec is used as-is (path or HF id).
    With --lang: 0 = base model, N = checkpoint-epoch-N, last = checkpoint-last.
    """
    if not lang:
        return checkpoint_spec  # original path / HF id passthrough
    if checkpoint_spec == "0":
        return default_checkpoint

    final_dir = CHECKPOINTS_ROOT / lang / "final"
    if not final_dir.is_dir():
        raise FileNotFoundError(
            f"No finetuned checkpoint directory for lang={lang!r}: {final_dir}\n"
            "Run step 5 (training) first."
        )

    if checkpoint_spec == "last":
        path = final_dir / "checkpoint-last"
        if not path.is_dir():
            raise FileNotFoundError(f"checkpoint-last not found under {final_dir}.")
        return str(path)

    try:
        epoch = int(checkpoint_spec)
    except ValueError as exc:
        raise ValueError(
            f"--checkpoint must be 0, 'last', or an epoch number; got: {checkpoint_spec!r}"
        ) from exc

    path = final_dir / f"checkpoint-epoch-{epoch}"
    if not path.is_dir():
        raise FileNotFoundError(
            f"checkpoint-epoch-{epoch} not found under {final_dir}.\n"
            f"Available: {[d.name for d in final_dir.iterdir() if d.is_dir()]}"
        )
    return str(path)


def resolve_onnx_model_dir(
    lang: str | None,
    checkpoint_spec: str,
) -> str | None:
    """Return the ONNX model-dir path, or None to use the built-in default."""
    if not lang or checkpoint_spec == "0":
        return None

    onnx_base = CHECKPOINTS_ROOT / lang / "final-onnx"

    if checkpoint_spec == "last":
        path = onnx_base / "checkpoint-last"
    else:
        try:
            epoch = int(checkpoint_spec)
        except ValueError as exc:
            raise ValueError(
                f"--checkpoint must be 0, 'last', or an epoch number; got: {checkpoint_spec!r}"
            ) from exc
        path = onnx_base / f"checkpoint-epoch-{epoch}"

    if not path.is_dir():
        raise FileNotFoundError(
            f"ONNX model dir not found: {path}\n"
            "Run scripts/07_export_onnx.py first."
        )
    return str(path)
