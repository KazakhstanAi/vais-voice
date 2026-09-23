"""Stage eligible real and synthetic audio into one immutable detector corpus."""

import argparse
import os
import shutil
from pathlib import Path

from vais_voice.datasets.manifest import read_manifest, write_manifest
from vais_voice.datasets.schema import Sample
from vais_voice.utils.io import contained_path, sha256, write_json


def _eligible_synthetic(row: Sample) -> bool:
    return (
        row.label == "synthetic"
        and row.quality_gate == "pass"
        and row.training_eligible is True
        and row.diagnostic_only is False
        and row.intended_role == "train"
    )


def compose_detector_corpus(
    real_manifest: Path, synthetic_manifest: Path, output: Path
) -> Path:
    real_manifest = real_manifest.resolve()
    synthetic_manifest = synthetic_manifest.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new corpus version, never overwrite")

    real_rows = read_manifest(real_manifest)
    if any(row.label != "real" for row in real_rows):
        raise ValueError("The real manifest must contain only real samples")
    synthetic_rows = read_manifest(synthetic_manifest)
    eligible = [row for row in synthetic_rows if _eligible_synthetic(row)]
    if not eligible:
        raise ValueError("No pass-gated, training-eligible synthetic samples were found")

    rows = [*real_rows, *eligible]
    sample_ids = [row.sample_id for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Real and synthetic manifests contain colliding sample_id values")

    output.mkdir(parents=True, exist_ok=False)
    audio_dir = output / "audio"
    audio_dir.mkdir()
    staged: list[Sample] = []
    methods = {"hardlink": 0, "copy": 0}
    roots = {"real": real_manifest.parent, "synthetic": synthetic_manifest.parent}
    for row in rows:
        source = contained_path(roots[row.label], row.path)
        if not source.is_file():
            raise ValueError(f"Missing source audio for {row.sample_id}: {source}")
        digest = sha256(source)
        expected = row.processed_sha256 or row.source_sha256
        if expected and digest != expected:
            raise ValueError(f"Audio checksum mismatch for {row.sample_id}")
        target = audio_dir / f"{row.sample_id}{source.suffix.lower()}"
        try:
            os.link(source, target)
            methods["hardlink"] += 1
        except OSError:
            shutil.copy2(source, target)
            methods["copy"] += 1
        staged.append(
            row.model_copy(update={"path": target.relative_to(output).as_posix(), "split": None})
        )

    jsonl = output / "manifest.jsonl"
    parquet = output / "manifest.parquet"
    write_manifest(jsonl, staged)
    write_manifest(parquet, staged)
    write_json(
        output / "composition.json",
        {
            "status": "complete",
            "real_manifest": str(real_manifest),
            "real_manifest_sha256": sha256(real_manifest),
            "synthetic_manifest": str(synthetic_manifest),
            "synthetic_manifest_sha256": sha256(synthetic_manifest),
            "manifest_hashes": {"jsonl": sha256(jsonl), "parquet": sha256(parquet)},
            "counts": {
                "total": len(staged),
                "real": len(real_rows),
                "synthetic": len(eligible),
                "synthetic_excluded": len(synthetic_rows) - len(eligible),
                "kk": sum(row.language == "kk" for row in staged),
                "ru": sum(row.language == "ru" for row in staged),
            },
            "staging_methods": methods,
            "eligibility_rule": {
                "quality_gate": "pass",
                "training_eligible": True,
                "diagnostic_only": False,
                "intended_role": "train",
            },
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(compose_detector_corpus(args.real_manifest, args.synthetic_manifest, args.output))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Composition failed: {exc}\n")


if __name__ == "__main__":
    main()
