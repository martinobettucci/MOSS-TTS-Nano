# model_patches

Canonical patched copies of files that ship inside an `OpenMOSS-Team/MOSS-TTS-Nano` checkpoint. They are version-controlled here in the private fork because the upstream HF Hub artifact does not honor `language=` as a prompt conditioner.

Two files are overridden:

- `prompting.py` — `build_user_prompt_after_reference`, `build_prompt_prefix`, `build_prompt_token_ids` accept an optional `language: str | None = None`. When `None`, the encoded suffix is **byte-identical** to upstream (all six metadata slots = `None`). When set, only the `Language:` line is replaced; the other five stay hardcoded to `None`.
- `modeling_moss_tts_nano.py` — `inference`, `inference_stream`, `build_inference_input_ids` accept the same optional `language` kwarg and forward it to the prompt builders. `_split_text_into_best_sentences` reads `config.training_chunk_text_tokens_recommended` when `max_tokens <= 0`.

## Backwards compatibility

These files are designed to be drop-in replacements for upstream:

- Loading the original `OpenMOSS-Team/MOSS-TTS-Nano` weights and calling `inference_stream(text=...)` without `language=` produces the same prompt token sequence as upstream.
- Loading a fine-tuned checkpoint and calling `inference_stream(text=..., language="fr")` produces the prompt the SFT trainer wrote.

## How they reach checkpoints

1. `scripts/prepare_phoneme_tokenizer.py` calls `model.save_pretrained(output_dir)` which writes the **unpatched** upstream copies into `models/MOSS-TTS-Nano-phoneme-ascii/`.
2. The same script overlays the files from this directory on top, restoring the patches.
3. `finetuning/sft.py:save_checkpoint` then copies all `MODEL_SUPPORT_FILES` from `models/MOSS-TTS-Nano-phoneme-ascii/` into each saved checkpoint, propagating the patches.

If you ever need to refresh against a newer upstream release, regenerate the local model dir from upstream, manually re-apply the patches by editing these two files, and commit. The pipeline will pick them up on the next run.
