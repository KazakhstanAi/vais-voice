"""Run one pinned Qwen3-TTS CustomVoice synthesis without resampling."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("The pinned Qwen3-TTS pilot requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = Qwen3TTSModel.from_pretrained(
        str(args.model.resolve()),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    speakers = model.get_supported_speakers() or []
    if args.speaker.lower() not in speakers:
        raise RuntimeError(f"Pinned speaker {args.speaker!r} not supported: {speakers}")
    wavs, sample_rate = model.generate_custom_voice(
        text=args.input_file.read_text(encoding="utf-8").strip(),
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=True,
        subtalker_top_k=args.top_k,
        subtalker_top_p=args.top_p,
        subtalker_temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    sf.write(args.output_file, wavs[0], sample_rate, subtype="PCM_16")
    print(
        json.dumps(
            {
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "sample_rate": sample_rate,
                "speaker": args.speaker,
                "torch_version": torch.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
