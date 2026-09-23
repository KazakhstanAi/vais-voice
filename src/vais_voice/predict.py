"""Run exact-ID detector inference over the protected test split."""

import argparse
import csv
from pathlib import Path

from vais_voice.data.validate import validate_files, verify_manifest
from vais_voice.datasets.manifest import read_manifest
from vais_voice.detection.base import Detector, DummyDetector
from vais_voice.utils.io import contained_path, sha256, write_json


def predict(
    manifest: Path,
    preparation: Path,
    output: Path,
    *,
    detector_name: str = "dummy",
    checkpoint: Path | None = None,
    device: str = "auto",
) -> None:
    verify_manifest(manifest, preparation)
    validate_files(manifest)
    rows = [r for r in read_manifest(manifest) if r.split == "test"]
    if not rows:
        raise ValueError("No test samples")
    metadata = output.with_suffix(".meta.json")
    if output.exists() or metadata.exists():
        raise ValueError("Prediction outputs already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, object]
    if detector_name == "dummy":
        detector: Detector = DummyDetector()
        provenance = {
            "detector": "dummy_constant_0.5",
            "checkpoint": None,
            "warning": "DUMMY_INFRASTRUCTURE_ONLY: not a scientific benchmark",
        }
    else:
        if checkpoint is None:
            raise ValueError("--checkpoint is required for checkpoint detector")
        try:
            from vais_voice.detection.runtime import CheckpointDetector
        except ImportError as exc:
            raise ValueError("Checkpoint inference requires the ML environment") from exc
        learned = CheckpointDetector(checkpoint, device=device)
        detector = learned
        provenance = {
            "detector": learned.metadata(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": learned.metadata()["checkpoint_sha256"],
        }
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
            **provenance,
            "manifest_sha256": sha256(manifest),
            "predictions_sha256": sha256(output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", choices=["dummy", "checkpoint"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        predict(
            args.manifest,
            args.preparation,
            args.output,
            detector_name=args.detector,
            checkpoint=args.checkpoint,
            device=args.device,
        )
        print(
            "DUMMY_INFRASTRUCTURE_ONLY: constant scores, no trained model"
            if args.detector == "dummy"
            else "Checkpoint predictions saved"
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")


if __name__ == "__main__":
    main()
