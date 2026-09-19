"""Dummy inference only: no learned model or authenticity prediction."""

import argparse
import csv
from pathlib import Path

from vais_voice.data.validate import validate_files, verify_manifest
from vais_voice.datasets.manifest import read_manifest
from vais_voice.detection.base import Detector, DummyDetector
from vais_voice.utils.io import contained_path, sha256, write_json


def predict(manifest: Path, preparation: Path, output: Path) -> None:
    verify_manifest(manifest, preparation)
    validate_files(manifest)
    rows = [r for r in read_manifest(manifest) if r.split == "test"]
    if not rows:
        raise ValueError("No test samples")
    metadata = output.with_suffix(".meta.json")
    if output.exists() or metadata.exists():
        raise ValueError("Prediction outputs already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    detector: Detector = DummyDetector()
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "synthetic_score"])
        for row in rows:
            writer.writerow(
                [row.sample_id, detector.predict_score(contained_path(manifest.parent, row.path))]
            )
    write_json(
        metadata,
        {
            "detector": "dummy_constant_0.5",
            "checkpoint": None,
            "warning": "DUMMY_INFRASTRUCTURE_ONLY: not a scientific benchmark",
            "manifest_sha256": sha256(manifest),
            "predictions_sha256": sha256(output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", choices=["dummy"], required=True)
    args = parser.parse_args()
    try:
        predict(args.manifest, args.preparation, args.output)
        print("DUMMY_INFRASTRUCTURE_ONLY: constant scores, no trained model")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
