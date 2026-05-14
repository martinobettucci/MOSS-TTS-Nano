from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from checkpoint_resolver import add_checkpoint_args, resolve_onnx_model_dir
from onnx_tts_runtime import (
    DEFAULT_BROWSER_ONNX_MODEL_DIR,
    DEFAULT_BROWSER_ONNX_OUTPUT_PATH,
    OnnxTtsRuntime,
)
from moss_tts_nano.reference_voices import resolve_reference_voice_path


def set_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native onnxruntime inference on browser_onnx exported assets.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "browser_onnx model directory. If omitted, the script uses "
            f"{DEFAULT_BROWSER_ONNX_MODEL_DIR} and auto-downloads the ONNX assets on first run."
        ),
    )
    parser.add_argument(
        "--output-audio-path",
        default=str(DEFAULT_BROWSER_ONNX_OUTPUT_PATH),
        help="Where to save the generated waveform.",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument("--text-file", help="Path to a UTF-8 text file to synthesize.")
    parser.add_argument(
        "--voice",
        default="Junhao",
        help="Built-in voice preset name used only when no reference audio path is provided.",
    )
    parser.add_argument(
        "--prompt-audio-path",
        "--reference-audio-path",
        dest="prompt_audio_path",
        default=None,
        help="Local reference audio path used directly for voice cloning. When provided, it overrides --voice.",
    )
    gender_group = parser.add_mutually_exclusive_group()
    gender_group.add_argument(
        "--male",
        dest="reference_voice_gender",
        action="store_const",
        const="male",
        default="male",
        help="Use the bundled male reference voice when no prompt audio path is provided.",
    )
    gender_group.add_argument(
        "--female",
        dest="reference_voice_gender",
        action="store_const",
        const="female",
        help="Use the bundled female reference voice when no prompt audio path is provided.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("greedy", "fixed", "full"),
        default="fixed",
        help="greedy=do_sample false, fixed=fixed hyperparameter sampled frame, full=host sampled full frame.",
    )
    parser.add_argument(
        "--do-sample",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Whether to sample. If 0, sample_mode is forced to greedy.",
    )
    parser.add_argument(
        "--realtime-streaming-decode",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Use codec streaming decode path internally instead of full decode.",
    )
    parser.add_argument("--cpu-threads", type=int, default=4, help="onnxruntime intra-op thread count.")
    parser.add_argument(
        "--execution-provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="onnxruntime execution provider. cuda requires an onnxruntime-gpu build.",
    )
    parser.add_argument("--max-new-frames", type=int, default=375, help="Maximum generated audio frames.")
    parser.add_argument(
        "--nq",
        type=int,
        default=None,
        help=(
            "Requested VQ channel count. Current ONNX codec exports require the full "
            "16-channel input; lower values are rejected instead of producing invalid audio."
        ),
    )
    parser.add_argument(
        "--voice-clone-max-text-tokens",
        type=int,
        default=0,
        help=(
            "Chunk long text by token budget. 0 (default) = use the checkpoint's "
            "`config.training_chunk_text_tokens_recommended` (set by trainer). "
            "Positive int = explicit override. Negative = disable chunking."
        ),
    )
    parser.add_argument("--text-temperature", type=float, default=1.0, help="Text-layer sampling temperature.")
    parser.add_argument("--text-top-p", type=float, default=1.0, help="Text-layer top-p sampling.")
    parser.add_argument("--text-top-k", type=int, default=50, help="Text-layer top-k sampling.")
    parser.add_argument("--audio-temperature", type=float, default=0.8, help="Audio-layer sampling temperature.")
    parser.add_argument("--audio-top-p", type=float, default=0.95, help="Audio-layer top-p sampling.")
    parser.add_argument("--audio-top-k", type=int, default=25, help="Audio-layer top-k sampling.")
    parser.add_argument(
        "--audio-repetition-penalty",
        type=float,
        default=1.2,
        help="Audio-layer repetition penalty.",
    )
    parser.add_argument(
        "--enable-wetext-processing",
        type=int,
        nargs="?",
        const=1,
        default=1,
        choices=[0, 1],
        help="Enable WeTextProcessing text normalization before inference.",
    )
    parser.add_argument(
        "--disable-wetext-processing",
        action="store_true",
        help="Disable WeTextProcessing even if enabled above.",
    )
    parser.add_argument(
        "--enable-normalize-tts-text",
        dest="enable_normalize_tts_text",
        action="store_true",
        default=True,
        help="Enable normalize_tts_text robust cleanup before inference.",
    )
    parser.add_argument(
        "--disable-normalize-tts-text",
        dest="disable_normalize_tts_text",
        action="store_true",
        help="Disable normalize_tts_text robust cleanup before inference.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--wer-language",
        default=None,
        help=(
            "Language hint for Whisper WER, e.g. en or fr. "
            "Defaults to --lang when evaluating a finetuned checkpoint; omit for Whisper auto-detect."
        ),
    )
    parser.add_argument(
        "--print-voice-clone-text-chunks",
        action="store_true",
        help="Print the effective chunked text before synthesis.",
    )
    parser.add_argument("--eval-wer", action="store_true", help="Compute WER via Whisper ASR after generation (slow, requires jiwer).")
    parser.add_argument("--eval-mos", action="store_true", help="Compute UTMOS (real MOS estimate) after generation (slow, requires speechmos).")
    add_checkpoint_args(parser)
    return parser.parse_args(argv)


def resolve_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return str(args.text)
    return Path(args.text_file).read_text(encoding="utf-8")


def resolve_prompt_audio_path(args: argparse.Namespace) -> str:
    if args.prompt_audio_path:
        return str(args.prompt_audio_path)
    return str(
        resolve_reference_voice_path(
            language=args.lang,
            gender=args.reference_voice_gender,
        )
    )


def maybe_print_voice_clone_text_chunks(runtime: OnnxTtsRuntime, text: str, max_tokens: int) -> None:
    chunks = runtime.split_voice_clone_text(text, max_tokens=max_tokens)
    effective_chunks = chunks if len(chunks) > 1 else [text]
    print("Voice clone text chunks")
    print("----------------------")
    print(f"max_tokens={max_tokens} chunks={len(effective_chunks)}")
    for chunk_index, chunk_text in enumerate(effective_chunks, start=1):
        print(f"[chunk {chunk_index}]")
        print(chunk_text)
        print()


def _mos_proxy(wav: np.ndarray) -> float:
    wav = np.asarray(wav, dtype=np.float32).flatten()
    frame_len = 480
    frames = [wav[i: i + frame_len] for i in range(0, len(wav) - frame_len, frame_len)]
    if not frames:
        return 1.0
    energies = np.array([np.mean(f ** 2) for f in frames])
    threshold = float(np.percentile(energies, 20))
    speech_e = energies[energies > threshold]
    noise_e = energies[energies <= threshold]
    if len(noise_e) == 0 or float(noise_e.mean()) <= 0:
        return 3.5
    snr = 10.0 * np.log10(float(speech_e.mean()) / (float(noise_e.mean()) + 1e-9))
    return float(np.clip(1.0 + (snr / 40.0) * 4.0, 1.0, 5.0))


def _normalize_whisper_language(language: str | None) -> str | None:
    normalized = str(language or "").strip().lower().replace("_", "-")
    if not normalized:
        return None
    primary = normalized.split("-", 1)[0]
    language_names = {
        "en": "english",
        "fr": "french",
        "de": "german",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
        "nl": "dutch",
        "pl": "polish",
        "ja": "japanese",
        "zh": "chinese",
        "ko": "korean",
    }
    return language_names.get(primary, primary)


def _eval_wer(
    audio_path: str,
    reference_text: str,
    execution_provider: str,
    language: str | None = None,
) -> float:
    try:
        import warnings
        from transformers import pipeline as hf_pipeline
        from jiwer import wer as compute_wer
        whisper_language = _normalize_whisper_language(language)
        logging.info(
            "[metrics] Running Whisper ASR for WER%s...",
            f" language={whisper_language}" if whisper_language else " with language auto-detect",
        )
        dev_id = 0 if execution_provider == "cuda" else -1
        generate_kwargs = {"task": "transcribe"}
        if whisper_language:
            generate_kwargs["language"] = whisper_language
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-base", device=dev_id)
            out = asr(audio_path, generate_kwargs=generate_kwargs)
        hyp = (out.get("text") or "").strip().lower()
        ref = reference_text.strip().lower()
        logging.info("[metrics] Whisper hypothesis: %s", hyp)
        logging.info("[metrics] WER reference: %s", ref)
        return float(min(1.0, compute_wer([ref], [hyp])))
    except ImportError:
        logging.warning("[metrics] jiwer or transformers not installed; WER skipped.")
        return -1.0
    except Exception as exc:
        logging.warning("[metrics] WER computation failed: %s", exc)
        return -1.0


def _eval_utmos(wav: np.ndarray, sample_rate: int) -> float:
    try:
        import speechmos
        return float(speechmos.utmos(wav.flatten().astype(np.float32), sr=sample_rate))
    except ImportError:
        logging.warning("[metrics] speechmos not installed; using MOS proxy instead.")
        return _mos_proxy(wav)
    except Exception as exc:
        logging.warning("[metrics] UTMOS computation failed: %s", exc)
        return _mos_proxy(wav)


def _print_metrics(
    *,
    generation_time_sec: float,
    audio_duration_sec: float,
    rtf: float,
    ttft_sec: float | None,
    mos_proxy: float,
    mos_utmos: float | None,
    wer: float | None,
) -> None:
    sep = "─" * 46
    lines = [
        f"[METRICS] {sep}",
        f"  Generation time : {generation_time_sec:.3f}s",
        f"  Audio duration  : {audio_duration_sec:.3f}s",
        f"  RTF             : {rtf:.3f}" + (" ✓" if 0 < rtf < 0.5 else (" ✗" if rtf >= 0 else " (n/a)")),
        f"  First audio     : {ttft_sec:.3f}s" if ttft_sec is not None else "  First audio     : n/a (non-streaming mode)",
        f"  MOS (proxy/SNR) : {mos_proxy:.2f}/5.0" + (" ✓" if mos_proxy > 3.0 else (" ✗" if mos_proxy > 0 else " (n/a)")),
    ]
    if mos_utmos is not None and mos_utmos >= 0:
        lines.append(f"  MOS (UTMOS)     : {mos_utmos:.2f}/5.0" + (" ✓" if mos_utmos > 3.0 else " ✗"))
    if wer is not None and wer >= 0:
        lines.append(f"  WER             : {wer:.1%}" + (" ✓" if wer < 0.1 else " ✗"))
    lines.append(f"[METRICS] {sep}")
    print("\n".join(lines), flush=True)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    set_logging()
    args = parse_args(argv)
    resolved_model_dir = resolve_onnx_model_dir(args.lang, args.checkpoint) or args.model_dir
    runtime = OnnxTtsRuntime(
        model_dir=resolved_model_dir,
        thread_count=args.cpu_threads,
        max_new_frames=args.max_new_frames,
        do_sample=bool(args.do_sample),
        sample_mode=args.sample_mode,
        execution_provider=args.execution_provider,
    )
    generation_defaults = runtime.manifest["generation_defaults"]
    generation_defaults["text_temperature"] = float(args.text_temperature)
    generation_defaults["text_top_p"] = float(args.text_top_p)
    generation_defaults["text_top_k"] = int(args.text_top_k)
    generation_defaults["audio_temperature"] = float(args.audio_temperature)
    generation_defaults["audio_top_p"] = float(args.audio_top_p)
    generation_defaults["audio_top_k"] = int(args.audio_top_k)
    generation_defaults["audio_repetition_penalty"] = float(args.audio_repetition_penalty)
    raw_text = resolve_text(args)
    prompt_audio_path = resolve_prompt_audio_path(args)
    enable_wetext = bool(args.enable_wetext_processing) and not bool(args.disable_wetext_processing)
    enable_normalize_tts_text = bool(args.enable_normalize_tts_text) and not bool(args.disable_normalize_tts_text)
    prepared = runtime.prepare_synthesis_text(
        text=raw_text,
        voice=str(args.voice or ""),
        language=args.lang,
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize_tts_text,
    )
    prepared_text = str(prepared["text"])
    logging.info(
        "text normalization method=%s language=%s text_chars=%d",
        prepared["normalization_method"],
        prepared["text_normalization_language"] or "n/a",
        len(prepared_text),
    )
    if args.print_voice_clone_text_chunks:
        maybe_print_voice_clone_text_chunks(runtime, prepared_text, args.voice_clone_max_text_tokens)
    if args.prompt_audio_path:
        logging.info("using direct reference audio path for voice cloning: %s", prompt_audio_path)
    else:
        logging.info(
            "using bundled %s reference voice for language=%s: %s",
            args.reference_voice_gender,
            args.lang or "en",
            prompt_audio_path,
        )
    t_gen_start = time.perf_counter()
    result = runtime.synthesize(
        text=raw_text,
        voice=args.voice,
        prompt_audio_path=prompt_audio_path,
        language=args.lang,
        output_audio_path=args.output_audio_path,
        sample_mode=args.sample_mode,
        do_sample=bool(args.do_sample),
        streaming=bool(args.realtime_streaming_decode),
        max_new_frames=args.max_new_frames,
        nq=args.nq,
        voice_clone_max_text_tokens=args.voice_clone_max_text_tokens,
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize_tts_text,
        seed=args.seed,
    )
    t_gen_end = time.perf_counter()
    generation_time_sec = t_gen_end - t_gen_start

    sample_rate = int(result["sample_rate"])
    waveform = np.asarray(result["waveform"], dtype=np.float32)
    wav_1d = waveform.flatten() if waveform.ndim > 1 else waveform
    audio_duration_sec = float(wav_1d.shape[0]) / sample_rate if sample_rate > 0 and wav_1d.size > 0 else 0.0
    rtf = generation_time_sec / audio_duration_sec if audio_duration_sec > 0 else -1.0
    mos_proxy = _mos_proxy(wav_1d) if wav_1d.size > 0 else -1.0

    first_audio_at = result.get("first_audio_at_perf")
    ttft_sec = (first_audio_at - t_gen_start) if first_audio_at is not None else None

    wer_language = args.wer_language or args.lang
    wer = (
        _eval_wer(
            str(result["audio_path"]),
            prepared_text,
            args.execution_provider,
            language=wer_language,
        )
        if args.eval_wer
        else None
    )
    mos_utmos = _eval_utmos(wav_1d, sample_rate) if args.eval_mos else None

    _print_metrics(
        generation_time_sec=generation_time_sec,
        audio_duration_sec=audio_duration_sec,
        rtf=rtf,
        ttft_sec=ttft_sec,
        mos_proxy=mos_proxy,
        mos_utmos=mos_utmos,
        wer=wer,
    )

    logging.info(
        "saved generated audio to %s sample_rate=%s frames=%s nq=%s sample_mode=%s streaming=%s execution_provider=%s",
        result["audio_path"],
        sample_rate,
        int(result["audio_token_ids"].shape[0]),
        result.get("nq"),
        result["sample_mode"],
        result["streaming"],
        runtime.execution_provider,
    )
    return result


if __name__ == "__main__":
    main()
