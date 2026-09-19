"""20 mathematical tones, NOT speech. Never use these metrics as model evidence."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml


def run(root: Path) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "raw").mkdir()
    rows = []
    for group in range(10):
        for index, label in enumerate(("real", "synthetic")):
            sid = f"fixture_{group}_{index}"
            sf.write(
                root / "raw" / f"{sid}.wav",
                0.1 * np.sin(2 * np.pi * (200 + group * 23 + index * 7) * np.arange(800) / 8000),
                8000,
            )
            rows.append(
                dict(
                    sample_id=sid,
                    path=f"{sid}.wav",
                    label=label,
                    language=("kk", "ru", "kk_ru")[group % 3],
                    speaker_id=f"toy_{group}",
                    source_dataset="MATHEMATICAL_FIXTURE_NOT_SPEECH",
                    source_id=f"tone_{group}",
                    original_path=f"{sid}.wav",
                    generator_id="toy_waveform" if index else None,
                    parent_sample_id=f"fixture_{group}_0" if index else None,
                    license="CC0-1.0",
                    rights_reference="Generated mathematical tones, no human voice",
                    usage_permission="approved",
                    split="train" if group < 5 else ("val" if group < 7 else "test"),
                )
            )
    (root / "input.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    config = dict(
        dataset_version="TOY_NOT_SPEECH_v1",
        split_version="fixture_v1",
        seed=42,
        project_root=".",
        manifest="input.jsonl",
        audio_root="raw",
        output_dir="processed",
        sample_rate=16000,
        max_duration_seconds=1,
        max_file_bytes=100000,
        train_fraction=0.7,
        val_fraction=0.15,
        unseen_generators=[],
    )
    (root / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    def cli(*args: str) -> None:
        subprocess.run([sys.executable, "-m", "vais_voice", *args], cwd=root, check=True)

    cli("prepare", "--config", "config.yaml")
    cli(
        "split",
        "--manifest",
        "processed/manifest.parquet",
        "--preparation",
        "processed/preparation.json",
    )
    common = [
        "--manifest",
        "processed/manifest.split.parquet",
        "--preparation",
        "processed/split_preparation.json",
    ]
    cli("predict", *common, "--detector", "dummy", "--output", "runs/dummy/predictions.csv")
    cli(
        "evaluate", *common, "--predictions", "runs/dummy/predictions.csv", "--output", "runs/dummy"
    )
    print(f"TOY ONLY. Artifacts: {root / 'runs/dummy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    run(parser.parse_args().output)
