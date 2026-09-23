"""Validate or train a VAIS detector baseline from a reviewed split manifest."""

import argparse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vais_voice.data.validate import verify_manifest
from vais_voice.datasets.manifest import read_manifest
from vais_voice.datasets.split import assign_splits
from vais_voice.utils.io import contained_path, load_yaml, sha256, write_json
from vais_voice.utils.tracking import snapshot


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    experiment: str = Field(min_length=1)
    seed: int = Field(ge=0)
    project_root: str
    prepared_manifest: str
    preparation_report: str
    run_dir: str
    model: dict[str, Any]


def preflight(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = TrainConfig.model_validate(load_yaml(config_path))
    root = (config_path.parent / config.project_root).resolve()
    manifest = root / config.prepared_manifest
    prep_path = root / config.preparation_report
    prep = verify_manifest(manifest, prep_path)
    rows = read_manifest(manifest)
    if any(row.split is None for row in rows):
        raise ValueError("Run split before training preflight")
    assign_splits(
        rows,
        seed=config.seed,
        train_fraction=0.7,
        val_fraction=0.15,
        unseen_generators=prep["config"]["unseen_generators"],
    )
    for row in rows:
        path = contained_path(manifest.parent, row.path)
        if not row.processed_sha256 or sha256(path) != row.processed_sha256:
            raise ValueError(f"Prepared audio missing or changed: {row.sample_id}")
        if row.sample_rate != config.model.get("sample_rate"):
            raise ValueError("Model and prepared audio sample rates differ")
    output = root / config.run_dir
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "preflight.json",
        {
            **snapshot(root),
            "status": "preflight_only_no_training",
            "dataset_version": prep["dataset_version"],
            "split_version": prep["split_version"],
            "seed": config.seed,
            "config": config.model_dump(),
            "manifest_sha256": sha256(manifest),
            "preparation_sha256": sha256(prep_path),
            "checkpoint": None,
            "metrics": None,
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.dry_run:
            print(f"Preflight saved: {preflight(args.config)}. No model trained.")
        else:
            try:
                from vais_voice.detection.training import train_detector
            except ImportError as exc:
                raise ValueError(
                    "Detector training requires the pinned ML environment; install .[ml]"
                ) from exc
            print(f"Training artifacts saved: {train_detector(args.config)}")
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
