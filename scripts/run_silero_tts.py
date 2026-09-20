"""Run one pinned Silero Torch package without network access."""

import argparse
from pathlib import Path

import soundfile as sf
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    args = parser.parse_args()
    text = args.input_file.read_text(encoding="utf-8").strip()
    model = torch.package.PackageImporter(str(args.model)).load_pickle("tts_models", "model")
    model.to(torch.device("cpu"))
    audio = model.apply_tts(text=text, speaker=args.speaker, sample_rate=args.sample_rate)
    sf.write(args.output_file, audio.detach().cpu().numpy(), args.sample_rate, subtype="PCM_16")


if __name__ == "__main__":
    main()
