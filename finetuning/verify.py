from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "MOSS-TTS-Nano"
DEFAULT_CODEC_PATH = REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"
DEFAULT_OUTPUT_AUDIO_PATH = REPO_ROOT / "generated_audio" / "finetune_verify.wav"
MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"
QUANT_MODES = {"fp8", "fp4", "int8"}
CUDA_ONLY_QUANT = {"fp8", "fp4"}
FAST_DECODE_CHUNK_FRAMES = 8
FAST_AUDIO_TEMPERATURE = 0.8
FAST_AUDIO_TOP_P = 0.95
FAST_AUDIO_TOP_K = 25
FAST_AUDIO_REPETITION_PENALTY = 1.2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick non-streaming validation for MOSS-TTS-Nano finetune checkpoints.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output-audio-path", default=str(DEFAULT_OUTPUT_AUDIO_PATH))

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument("--text-file", help="UTF-8 text file to synthesize.")

    prompt_text_group = parser.add_mutually_exclusive_group(required=False)
    prompt_text_group.add_argument("--prompt-text", help="Prompt transcript for continuation mode.")
    prompt_text_group.add_argument("--prompt-text-file", help="UTF-8 prompt transcript file.")

    parser.add_argument("--mode", default="voice_clone", choices=("continuation", "voice_clone"))
    parser.add_argument("--prompt-audio-path", default=None)
    parser.add_argument("--reference-audio-path", default=None, help="Compatibility alias for --prompt-audio-path.")
    parser.add_argument(
        "--audio-tokenizer-pretrained-name-or-path",
        default=str(DEFAULT_CODEC_PATH),
        help="Local codec path or HF repo id. Defaults to repo-local models/MOSS-Audio-Tokenizer-Nano.",
    )
    parser.add_argument("--text-tokenizer-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16", "fp8", "fp4", "int8"))
    parser.add_argument("--nq", type=int, default=None, help="Number of VQ channels to decode (1-16). Lower = faster, lower quality.")
    parser.add_argument("--max-new-frames", type=int, default=375)
    parser.add_argument("--do-sample", type=int, nargs="?", const=1, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help=(
            "Force fully deterministic CUDA ops. Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "in the environment (set automatically by tts_infer.py). May be slower."
        ),
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Run a warmup inference pass before timing (warms CUDA kernels, counted in model_load_s).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply torch.compile to the transformer for faster inference (requires warmup for best results).",
    )
    parser.add_argument(
        "--tries",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run N timed inferences (after warmup/compile) and report per-run and aggregate stats. "
            "Must be >= 1. Default: 1."
        ),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Use the optimised direct loop (no inference_stream generator overhead, "
            "KV cache enabled for local transformer). Continuation mode only."
        ),
    )
    parser.add_argument(
        "--local-2pass",
        dest="local_two_pass",
        action="store_true",
        default=False,
        help=(
            "With --fast: use two full-sequence passes for the local decoder "
            "(~1.5 ms) instead of 17 × KV-cache 1-token calls (~9.4 ms). "
            "Pass-1 samples approximate tokens; pass-2 refines with actual embeddings. "
            "Slight quality tradeoff for channels > 2; enables near 0.25 RTF targets."
        ),
    )
    return parser.parse_args(argv)


def resolve_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.text_file).read_text(encoding="utf-8")


def resolve_prompt_text(args: argparse.Namespace) -> Optional[str]:
    if args.prompt_text is not None:
        return args.prompt_text
    if args.prompt_text_file is not None:
        return Path(args.prompt_text_file).read_text(encoding="utf-8")
    return None


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bf16_or_fp16(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_quantized_model(checkpoint: str, device: torch.device, dtype_arg: str) -> AutoModelForCausalLM:
    if dtype_arg in CUDA_ONLY_QUANT and device.type != "cuda":
        raise SystemExit(
            f"[error] --dtype {dtype_arg} requires CUDA — pass --gpu in tts_infer.py"
        )

    if dtype_arg == "fp4":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=_bf16_or_fp16(device),
        )
        return AutoModelForCausalLM.from_pretrained(
            checkpoint, quantization_config=bnb_config, trust_remote_code=True
        )

    if dtype_arg == "fp8":
        from torchao.quantization import Float8WeightOnlyConfig, quantize_
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        model.to(device=device, dtype=_bf16_or_fp16(device))
        quantize_(model, Float8WeightOnlyConfig())
        return model

    if dtype_arg == "int8":
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        base_dtype = _bf16_or_fp16(device) if device.type == "cuda" else torch.float32
        if device.type != "cuda":
            print(
                "[int8] Note: on CPU, INT8 weight-only reduces model RAM by ~50% "
                "but does not speed up inference on this hardware (torchao 0.17 / ARM). "
                "Use float32 for best latency.",
                flush=True,
            )
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        model.to(device=device, dtype=base_dtype)
        quantize_(model, Int8WeightOnlyConfig(version=2))
        return model

    raise ValueError(f"Unknown quant mode: {dtype_arg}")


def _apply_compile(model: AutoModelForCausalLM, device: torch.device) -> None:
    """Apply torch.compile to the hot inference paths.

    Uses 'default' mode with dynamic=True to handle variable sequence lengths
    and KV-cache growth safely (CUDA-graph mode conflicts with rotary pos emb + KV cache).
    """
    if device.type != "cuda":
        print("[compile] torch.compile skipped: not a CUDA device", flush=True)
        return
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        model.transformer = torch.compile(
            model.transformer,
            mode="default",
            fullgraph=False,
            dynamic=True,
        )
        model.local_transformer = torch.compile(
            model.local_transformer,
            mode="default",
            fullgraph=False,
            dynamic=True,
        )
        print("[compile] torch.compile applied (mode=default, dynamic=True)", flush=True)
    except Exception as exc:
        print(f"[compile] torch.compile failed ({exc}), continuing without compile", flush=True)


def _sample_audio(
    logits: "torch.Tensor",
    do_sample: bool,
    temperature: float = FAST_AUDIO_TEMPERATURE,
    top_k: int = FAST_AUDIO_TOP_K,
    top_p: float = FAST_AUDIO_TOP_P,
    previous_seen_mask: "torch.Tensor | None" = None,
    repetition_penalty: float = FAST_AUDIO_REPETITION_PENALTY,
) -> "torch.Tensor":
    """Apply temperature / top-k / top-p sampling to audio channel logits.

    Matches the defaults used by inference_stream so the fast path produces the
    same token distribution as the original code.
    logits: (batch, codebook_size) — raw logits from audio_lm_heads[ch]
    Returns: (batch,) int tensor with sampled token indices.
    """
    scores = logits.float()
    if previous_seen_mask is not None and repetition_penalty != 1.0 and bool(previous_seen_mask.any()):
        seen = previous_seen_mask.to(device=scores.device, dtype=torch.bool).unsqueeze(0)
        penalized = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
        scores = torch.where(seen, penalized, scores)
    if not do_sample:
        return scores.argmax(dim=-1)
    scaled = scores / max(temperature, 1e-6)
    if top_k > 0 and top_k < scaled.shape[-1]:
        kth_vals = scaled.topk(top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth_vals, float("-inf"))
    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = torch.cat(
            [torch.zeros_like(cum_probs[..., :1], dtype=torch.bool),
             cum_probs[..., :-1] >= top_p],
            dim=-1,
        )
        sorted_logits.masked_fill_(remove, float("-inf"))
        scaled.scatter_(-1, sorted_idx, sorted_logits)
    probs = scaled.softmax(dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def _run_warmup(model: AutoModelForCausalLM, device: torch.device, codec, nq: int | None = None) -> None:
    """Warm up CUDA kernels with a short inference using the same nq as the real run."""
    print("[warmup] running warmup inference...", flush=True)
    t_wu = time.perf_counter()
    list(model.inference_stream(
        text="Ok.",
        output_audio_path="/tmp/_warmup_verify.wav",
        mode="continuation",
        prompt_text=None,
        prompt_audio_path=None,
        text_tokenizer_path=None,
        audio_tokenizer=codec,
        device=device,
        nq=nq,
        max_new_frames=10,
        do_sample=False,
        use_kv_cache=True,
    ))
    _cuda_sync(device)
    print(f"[warmup] done in {time.perf_counter() - t_wu:.2f}s", flush=True)


def _local_transformer_step(
    model: AutoModelForCausalLM,
    inputs_embeds: "torch.Tensor",
    past_key_values=None,
    attention_len: int = 1,
):
    return model.local_transformer(
        input_ids=None,
        past_key_values=past_key_values,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=inputs_embeds,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cu_seqlens=None,
        num_sequences=None,
    )


@torch.inference_mode()
def _local_decode_frame_twopass(
    model: AutoModelForCausalLM,
    g_h: "torch.Tensor",
    eff_nq: int,
    do_sample: bool,
    slot_id: int,
    end_id: int,
    audio_pad: int,
    device: "torch.device",
    # pre-stacked weight tensors to avoid re-stacking per frame
    lm_weights: "torch.Tensor | None" = None,   # (n_vq, codebook, emb)
    emb_weights: "torch.Tensor | None" = None,  # (n_vq, codebook, emb)
    slot_emb: "torch.Tensor | None" = None,     # (emb,) pre-embedded slot token
) -> "tuple[bool, list[int]]":
    """
    Decode one full local frame using a two-pass full-sequence approach.

    Pass-1 (0.70 ms):  run all (n_vq+1) positions in ONE forward call with
                       global_h at position-0 and zeros at positions 1-n_vq.
                       - Position-0 logit is EXACT (causal: attends only to itself).
                       - Positions 1+ logits are approximate (zero-embedding inputs).
                       Use pass-1 to determine the text decision and to build the
                       approximate token sequence for pass-2.

    Pass-2 (0.70 ms):  re-run with actual embeddings at every position (from pass-1
                       approximate tokens).
                       - Each position k now has the correct embeddings for positions 0..k-1
                         (from pass-1 tokens), so logits are GREATLY IMPROVED.
                       - For ch0 the logit is now EXACT (text_emb at position-1 is correct).
                       - For later channels the logit uses pass-1 approximate tokens which
                         are typically close to the final answer.

    Total: ~1.5 ms for 2 full-sequence passes vs ~9.4 ms for 17 × KV-cache 1-token steps.
    """
    ldtype = g_h.dtype
    n_vq = model.config.n_vq
    emb_dim = g_h.shape[-1]

    # ---------- pass 1: all zeros for positions 1-n_vq ----------
    seq1 = torch.zeros(1, n_vq + 1, emb_dim, dtype=ldtype, device=device)
    seq1[:, 0] = g_h.squeeze(1)

    out1 = model.local_transformer(inputs_embeds=seq1, use_cache=False, return_dict=True)
    hs1 = out1.last_hidden_state  # (1, n_vq+1, emb_dim)

    # text decision from position 0 — this is EXACT
    text_logits = model.text_lm_head(hs1[:, 0])
    if text_logits[0, end_id] >= text_logits[0, slot_id]:
        return False, []

    # approximate tokens from pass-1 (wrong embeddings → wrong logits, but close)
    approx_tokens: list[int] = []
    for ch in range(eff_nq):
        ch_logits = model.audio_lm_heads[ch](hs1[:, ch + 1])
        approx_tokens.append(int(_sample_audio(ch_logits, do_sample).item()))

    # ---------- pass 2: fill with approximate token embeddings ----------
    seq2 = torch.zeros(1, n_vq + 1, emb_dim, dtype=ldtype, device=device)
    seq2[:, 0] = g_h.squeeze(1)
    seq2[:, 1] = model.transformer.wte(
        torch.tensor([[slot_id]], device=device)
    ).to(ldtype).squeeze(1)
    for ch, tok in enumerate(approx_tokens[: eff_nq - 1]):
        seq2[:, ch + 2] = model.audio_embeddings[ch](
            torch.tensor([[tok]], device=device)
        ).to(ldtype).squeeze(1)

    out2 = model.local_transformer(inputs_embeds=seq2, use_cache=False, return_dict=True)
    hs2 = out2.last_hidden_state

    # final tokens from pass-2 (corrected logits)
    frame: list[int] = []
    for ch in range(eff_nq):
        ch_logits = model.audio_lm_heads[ch](hs2[:, ch + 1])
        frame.append(int(_sample_audio(ch_logits, do_sample).item()))

    return True, frame


@torch.inference_mode()
def _fast_continuation_inference(
    model: AutoModelForCausalLM,
    codec,
    device: torch.device,
    text: str,
    checkpoint_path: str,
    nq: Optional[int],
    max_new_frames: int,
    output_audio_path: str,
    do_sample: bool,
    local_two_pass: bool = False,
) -> dict:
    """
    Tight generation loop for continuation-mode TTS.

    Replaces inference_stream which has ~28 ms/frame of Python generator overhead.
    Key optimisations vs the default path:
      - KV cache enabled for the local (per-frame) transformer
      - Pre-allocated per-frame local KV; reset each frame (no growing sequence copies)
      - Chunked codec.batch_decode after the first frame, preserving TTFT
      - No torch.stack(history) O(n) copy per step
    """
    import torchaudio

    ldtype = model.local_transformer.ln_f.weight.dtype
    n_vq: int = model.config.n_vq
    eff_nq: int = min(int(nq or n_vq), n_vq)
    audio_pad = model.config.audio_pad_token_id
    slot_id = model.config.audio_assistant_slot_token_id
    end_id = model.config.audio_end_token_id

    # ---- tokenise and prefill -----------------------------------------------
    text_tok = model._load_text_tokenizer(text_tokenizer_path=checkpoint_path)
    input_ids, attention_mask = model.build_inference_input_ids(
        text=text, text_tokenizer=text_tok, mode="continuation", device=device
    )

    prefill_embeds = model._build_inputs_embeds(input_ids)
    g_out = model.transformer(
        inputs_embeds=prefill_embeds,
        attention_mask=attention_mask,
        use_cache=True, return_dict=True,
    )
    g_kv = g_out.past_key_values
    g_h = g_out.last_hidden_state[:, -1:, :].to(ldtype)  # (1,1,d)
    g_attn_len = input_ids.shape[1]
    g_attn_full = torch.ones(1, g_attn_len + max_new_frames + 1, dtype=torch.bool, device=device)
    slot_token_tensor = torch.tensor([[slot_id]], dtype=torch.long, device=device)
    audio_token_tensor = torch.empty((1, 1), dtype=torch.long, device=device)
    next_row = torch.empty((1, 1, n_vq + 1), dtype=torch.long, device=device)

    # ---- frame loop ---------------------------------------------------------
    all_frames: list[list[int]] = []
    all_audio: list[torch.Tensor] = []
    pending_decode_frames: list[list[int]] = []
    sample_rate: Optional[int] = None
    t_first_audio: Optional[float] = None
    codec_reset = True
    audio_codebook_size = int(model.config.audio_codebook_sizes[0])
    previous_seen_by_channel = torch.zeros(
        n_vq,
        audio_codebook_size,
        dtype=torch.bool,
        device=device,
    )
    profile_enabled = os.environ.get("MOSS_PROFILE") == "1"
    profile_local_s = 0.0
    profile_global_s = 0.0
    profile_codec_s = 0.0
    final_decode = os.environ.get("MOSS_FINAL_DECODE") == "1"

    def flush_decode() -> None:
        nonlocal codec_reset, sample_rate, t_first_audio, pending_decode_frames, profile_codec_s
        if not pending_decode_frames:
            return
        if profile_enabled:
            _cuda_sync(device)
            t_profile = time.perf_counter()
        codes = torch.tensor(
            pending_decode_frames, dtype=torch.long, device=device
        ).transpose(0, 1).contiguous()  # (eff_nq, chunk_frames)
        pending_decode_frames = []
        codec_out = codec.batch_decode(
            [codes],
            num_quantizers=eff_nq,
            streaming=True,
            max_batch_size=1,
            reset_stream=codec_reset,
        )
        codec_reset = False
        if t_first_audio is None:
            _cuda_sync(device)
            t_first_audio = time.perf_counter()

        raw_audio = codec_out.audio  # (1, channels, samples)
        length = int(codec_out.audio_lengths[0].item())
        if length > 0:
            all_audio.append(raw_audio[0, :, :length].cpu())
        if sample_rate is None:
            sample_rate = int(getattr(model.config, "audio_tokenizer_sample_rate", 48000))
        if profile_enabled:
            _cuda_sync(device)
            profile_codec_s += time.perf_counter() - t_profile

    for _step in range(max_new_frames):
        if profile_enabled:
            _cuda_sync(device)
            t_profile = time.perf_counter()
        if local_two_pass:
            # -- two-pass full-sequence local decoder (1.5 ms vs 9.4 ms) ------
            ok, frame = _local_decode_frame_twopass(
                model, g_h, eff_nq, do_sample, slot_id, end_id, audio_pad, device
            )
            if not ok:
                break
        else:
            # -- exact local decoder with KV cache --------------------------------
            # Same autoregressive inputs as inference_stream, but without repeatedly
            # re-running the full local prefix for every VQ channel.
            local_out = _local_transformer_step(
                model,
                g_h,
                past_key_values=None,
                attention_len=1,
            )
            local_kv = local_out.past_key_values
            local_h = local_out.last_hidden_state[:, -1, :]
            text_logits = model.text_lm_head(local_h)
            if text_logits[0, slot_id] < text_logits[0, end_id]:
                break

            # text slot embedding becomes position 1
            cur_emb = model.transformer.wte(slot_token_tensor).to(ldtype)
            frame = []
            for ch in range(eff_nq):
                local_out = _local_transformer_step(
                    model,
                    cur_emb,
                    past_key_values=local_kv,
                    attention_len=ch + 2,
                )
                local_kv = local_out.past_key_values
                local_h = local_out.last_hidden_state[:, -1, :]
                ch_logits = model.audio_lm_heads[ch](local_h)
                tok = int(
                    _sample_audio(
                        ch_logits,
                        do_sample,
                        previous_seen_mask=previous_seen_by_channel[ch],
                    ).item()
                )
                frame.append(tok)
                if 0 <= tok < audio_codebook_size:
                    previous_seen_by_channel[ch, tok] = True
                audio_token_tensor.fill_(tok)
                cur_emb = model.audio_embeddings[ch](audio_token_tensor).to(ldtype)
        if profile_enabled:
            _cuda_sync(device)
            profile_local_s += time.perf_counter() - t_profile

        # pad unused channels
        padded_frame = frame + [audio_pad] * (n_vq - eff_nq)
        all_frames.append(padded_frame)
        pending_decode_frames.append(frame)
        if t_first_audio is None or (
            not final_decode and len(pending_decode_frames) >= FAST_DECODE_CHUNK_FRAMES
        ):
            flush_decode()

        # -- global transformer decode step -----------------------------------
        if profile_enabled:
            _cuda_sync(device)
            t_profile = time.perf_counter()
        next_row.fill_(audio_pad)
        next_row[0, 0, 0] = slot_id
        for ci, t in enumerate(frame):
            next_row[0, 0, ci + 1] = t

        g_attn_len += 1
        g_attn = g_attn_full[:, :g_attn_len]
        g_out = model.transformer(
            inputs_embeds=model._build_inputs_embeds(next_row),
            past_key_values=g_kv,
            attention_mask=g_attn,
            use_cache=True, return_dict=True,
        )
        g_kv = g_out.past_key_values
        g_h = g_out.last_hidden_state[:, -1:, :].to(ldtype)
        if profile_enabled:
            _cuda_sync(device)
            profile_global_s += time.perf_counter() - t_profile

    if final_decode:
        if profile_enabled:
            _cuda_sync(device)
            t_profile = time.perf_counter()
        if all_frames:
            final_codes = torch.tensor(
                [frame[:eff_nq] for frame in all_frames],
                dtype=torch.long,
                device=device,
            ).transpose(0, 1).contiguous()
            codec_out = codec.batch_decode(
                [final_codes],
                num_quantizers=eff_nq,
                streaming=False,
                reset_stream=True,
            )
            raw_audio = codec_out.audio
            length = int(codec_out.audio_lengths[0].item())
            all_audio = [raw_audio[0, :, :length].cpu()] if length > 0 else []
            sample_rate = int(getattr(model.config, "audio_tokenizer_sample_rate", 48000))
        if profile_enabled:
            _cuda_sync(device)
            profile_codec_s += time.perf_counter() - t_profile
    else:
        flush_decode()
    if profile_enabled:
        print(
            f"[PROFILE] local_s={profile_local_s:.3f} "
            f"global_s={profile_global_s:.3f} codec_s={profile_codec_s:.3f}",
            flush=True,
        )

    # ---- save audio ---------------------------------------------------------
    # each element of all_audio is (channels, samples); cat along last dim
    if all_audio:
        waveform = torch.cat(all_audio, dim=-1)  # (channels, total_samples)
        torchaudio.save(output_audio_path, waveform.float(), sample_rate or 48000)
    else:
        waveform = torch.zeros(1, 0)

    frames_tensor = torch.tensor(all_frames, dtype=torch.long)  # (T, n_vq)
    return {
        "audio_path": output_audio_path,
        "audio_token_ids": frames_tensor,
        "sample_rate": sample_rate or 48000,
        "waveform": waveform,
        "t_first_audio": t_first_audio,
    }


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    args = parse_args(argv)

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = resolve_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    t_load_start = time.perf_counter()
    if args.dtype in QUANT_MODES:
        model = _load_quantized_model(args.checkpoint, device, args.dtype)
    else:
        dtype = resolve_dtype(args.dtype, device)
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True)
        model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa" if device.type == "cuda" else "eager")
    model.eval()
    _cuda_sync(device)
    print(f"[PERF] dtype={args.dtype}", flush=True)

    # Pre-load codec (audio tokenizer) BEFORE the TTFT timer so it doesn't inflate TTFT.
    codec = model._load_audio_tokenizer(
        audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=args.audio_tokenizer_pretrained_name_or_path,
        device=device,
    )
    _cuda_sync(device)

    if args.compile:
        _apply_compile(model, device)

    if args.warmup:
        _run_warmup(model, device, codec, nq=args.nq)
        _cuda_sync(device)

    t_load_end = time.perf_counter()
    print(f"[PERF] model_load_s={t_load_end - t_load_start:.3f}", flush=True)

    tries = max(1, int(args.tries))
    text = resolve_text(args)
    prompt_text = resolve_prompt_text(args)

    run_ttft: list[float] = []
    run_generate: list[float] = []
    run_frames: list[int] = []
    last_result: dict | None = None

    use_fast = bool(args.fast) and args.mode in ("continuation", "auto") and args.prompt_audio_path is None and args.reference_audio_path is None

    for try_index in range(tries):
        if args.seed is not None:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        t_gen_start = time.perf_counter()
        t_first_audio: float | None = None
        result: dict | None = None

        if use_fast:
            result = _fast_continuation_inference(
                model=model,
                codec=codec,
                device=device,
                text=text,
                checkpoint_path=args.checkpoint,
                nq=args.nq,
                max_new_frames=args.max_new_frames,
                output_audio_path=args.output_audio_path,
                do_sample=bool(args.do_sample),
                local_two_pass=getattr(args, "local_two_pass", False),
            )
            _cuda_sync(device)
            t_first_audio = result.pop("t_first_audio", None)
        else:
            for event in model.inference_stream(
                text=text,
                output_audio_path=args.output_audio_path,
                mode=args.mode,
                prompt_text=prompt_text,
                prompt_audio_path=args.prompt_audio_path,
                reference_audio_path=args.reference_audio_path,
                text_tokenizer_path=args.text_tokenizer_path or args.checkpoint,
                audio_tokenizer=codec,
                device=device,
                nq=args.nq,
                max_new_frames=args.max_new_frames,
                do_sample=bool(args.do_sample),
                use_kv_cache=True,
            ):
                if event["type"] == "audio" and t_first_audio is None:
                    _cuda_sync(device)
                    t_first_audio = time.perf_counter()
                elif event["type"] == "result":
                    result = event

        _cuda_sync(device)
        t_gen_end = time.perf_counter()

        frames = int(result['audio_token_ids'].shape[0])
        ttft = t_first_audio - t_gen_start if t_first_audio is not None else None
        generate_s = t_gen_end - t_gen_start

        run_frames.append(frames)
        run_generate.append(generate_s)
        if ttft is not None:
            run_ttft.append(ttft)

        last_result = result

        if tries > 1:
            ttft_str = f"{ttft:.3f}" if ttft is not None else "n/a"
            print(
                f"[PERF_RUN_{try_index + 1}] ttft_s={ttft_str} "
                f"generate_s={generate_s:.3f} frames={frames}",
                flush=True,
            )

    # Emit aggregate PERF lines (mean across all tries) — these are what tts_infer.py parses.
    mean_generate = sum(run_generate) / len(run_generate)
    mean_frames = sum(run_frames) / len(run_frames)
    if run_ttft:
        mean_ttft = sum(run_ttft) / len(run_ttft)
        print(f"[PERF] ttft_s={mean_ttft:.3f}", flush=True)
    print(f"[PERF] generate_s={mean_generate:.3f}", flush=True)
    print(f"[PERF] audio_token_frames={mean_frames:.1f}", flush=True)
    print(f"[PERF] sample_rate={last_result['sample_rate']}", flush=True)

    if tries > 1:
        print(
            f"[PERF_SUMMARY] tries={tries} "
            + (f"ttft_min={min(run_ttft):.3f} ttft_max={max(run_ttft):.3f} " if run_ttft else "")
            + f"gen_min={min(run_generate):.3f} gen_max={max(run_generate):.3f}",
            flush=True,
        )

    print(
        f"saved {last_result['audio_path']} sample_rate={last_result['sample_rate']} "
        f"frames={run_frames[-1]}"
    )
    return last_result


if __name__ == "__main__":
    main()
