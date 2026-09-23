"""python -m vais_voice.data.prepare --config configs/data/kzru_v1.yaml"""

import argparse
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vais_voice.datasets.build import build_dataset
from vais_voice.datasets.manifest import read_manifest, write_manifest
from vais_voice.datasets.split import assign_splits
from vais_voice.preprocessing.audio import preprocess
from vais_voice.utils.io import contained_path, load_yaml, sha256, write_json
from vais_voice.utils.tracking import snapshot


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    project_root: str
    manifest: str
    audio_root: str
    output_dir: str
    sample_rate: int = Field(ge=8000, le=96000)
    max_duration_seconds: float = Field(gt=0, le=3600)
    max_file_bytes: int = Field(gt=0, le=1073741824)
    train_fraction: float = Field(gt=0, lt=1)
    val_fraction: float = Field(gt=0, lt=1)
    unseen_generators: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_fractions(self) -> "DataConfig":
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("A positive test fraction is required")
        return self


def prepare(config_path: Path, *, split_now: bool = False) -> Path:
    config_path = config_path.resolve()
    raw_config = load_yaml(config_path)
    if "sources" in raw_config:
        if split_now:
            raise ValueError("Adapter dataset builds cannot be split during ingestion")
        return build_dataset(config_path)
    config = DataConfig.model_validate(raw_config)
    root = (config_path.parent / config.project_root).resolve()
    manifest = (root / config.manifest).resolve()
    audio_root = (root / config.audio_root).resolve()
    output = (root / config.output_dir).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new dataset version, never overwrite")
    rows = read_manifest(manifest)
    sources = {}
    hashed = []
    for row in rows:
        source = contained_path(audio_root, row.path)
        if not source.is_file():
            raise ValueError(f"Missing audio for sample {row.sample_id}")
        if source.stat().st_size > config.max_file_bytes:
            raise ValueError(f"Oversized audio for sample {row.sample_id}")
        digest = sha256(source)
        expected = row.processed_sha256 or row.source_sha256
        if expected and expected != digest:
            raise ValueError(f"Source checksum mismatch for {row.sample_id}")
        hashed.append(row.model_copy(update={"source_sha256": digest}))
        sources[row.sample_id] = source
    rows = hashed
    if split_now:
        rows = assign_splits(
            hashed,
            seed=config.seed,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            unseen_generators=config.unseen_generators,
        )
    output.mkdir(parents=True, exist_ok=False)
    (output / "audio").mkdir()
    report = {
        **snapshot(root),
        "status": "incomplete",
        "config": config.model_dump(),
        "input_manifest_sha256": sha256(manifest),
    }
    # Completion marker is written only after every sample succeeds.
    write_json(output / "started.json", report)
    prepared = []
    for row in rows:
        target = output / "audio" / f"{row.sample_id}.wav"
        metadata = preprocess(
            sources[row.sample_id],
            target,
            sample_rate=config.sample_rate,
            max_duration_seconds=config.max_duration_seconds,
            max_file_bytes=config.max_file_bytes,
        )
        prepared.append(
            row.model_copy(
                update={
                    "path": target.relative_to(output).as_posix(),
                    "processed_sha256": sha256(target),
                    **metadata,
                }
            )
        )
    write_manifest(output / "manifest.jsonl", prepared)
    write_manifest(output / "manifest.parquet", prepared)
    write_json(
        output / "preparation.json",
        {
            **report,
            "status": "complete",
            "dataset_version": config.dataset_version,
            "split_version": config.split_version,
            "seed": config.seed,
            "prepared_manifest_sha256": sha256(output / "manifest.jsonl"),
            "manifest_hashes": {
                name: sha256(output / name) for name in ("manifest.jsonl", "manifest.parquet")
            },
            "counts": {s: sum(r.split == s for r in prepared) for s in ("train", "val", "test")},
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(f"Prepared dataset: {prepare(args.config)}")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
