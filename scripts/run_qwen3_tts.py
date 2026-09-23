"""Run one pinned Qwen3-TTS CustomVoice synthesis without resampling."""

import argparse
import contextlib
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def _load_model(model_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("The pinned Qwen3-TTS pilot requires CUDA")
    with contextlib.redirect_stdout(sys.stderr):
        model = Qwen3TTSModel.from_pretrained(
            str(model_path.resolve()),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
    return model


def _synthesize(model, request: dict) -> dict:
    seed = int(request["seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    speaker = str(request["speaker"])
    speakers = model.get_supported_speakers() or []
    if speaker.lower() not in speakers:
        raise RuntimeError(f"Pinned speaker {speaker!r} not supported: {speakers}")
    with contextlib.redirect_stdout(sys.stderr):
        wavs, sample_rate = model.generate_custom_voice(
            text=str(request["text"]).strip(),
            language=str(request["language"]),
            speaker=speaker,
            instruct=str(request["instruct"]),
            do_sample=True,
            top_k=int(request["top_k"]),
            top_p=float(request["top_p"]),
            temperature=float(request["temperature"]),
            repetition_penalty=float(request["repetition_penalty"]),
            subtalker_dosample=True,
            subtalker_top_k=int(request["top_k"]),
            subtalker_top_p=float(request["top_p"]),
            subtalker_temperature=float(request["temperature"]),
            max_new_tokens=int(request["max_new_tokens"]),
        )
    output_file = Path(request["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, wavs[0], sample_rate, subtype="PCM_16")
    return {
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "sample_rate": sample_rate,
        "speaker": speaker,
        "torch_version": torch.__version__,
    }


def _worker(model_path: Path) -> None:
    model = _load_model(model_path)
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            metadata = _synthesize(model, request)
            response = {"status": "completed", "metadata": metadata}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--speaker")
    parser.add_argument("--language")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    if args.worker:
        _worker(args.model)
        return
    required = (
        args.input_file,
        args.output_file,
        args.speaker,
        args.language,
        args.seed,
        args.top_k,
        args.top_p,
        args.temperature,
        args.repetition_penalty,
        args.max_new_tokens,
    )
    if any(value is None for value in required):
        parser.error("single-file mode requires input/output, speaker, language, and seed")
    model = _load_model(args.model)
    metadata = _synthesize(
        model,
        {
            "text": args.input_file.read_text(encoding="utf-8"),
            "output_file": str(args.output_file),
            "speaker": args.speaker,
            "language": args.language,
            "instruct": args.instruct,
            "seed": args.seed,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
        },
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
