"""Build an immutable canonical dataset from local source adapters."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vais_voice.datasets.adapters import adapter_from_config
from vais_voice.datasets.adapters.base import AdaptedSample
from vais_voice.datasets.manifest import validate_rows, write_manifest
from vais_voice.datasets.roles import ResearchPartition, validate_generator_roles
from vais_voice.preprocessing.audio import preprocess
from vais_voice.utils.io import load_yaml, sha256, write_json
from vais_voice.utils.tracking import snapshot


class SourceBuild(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    enabled: bool = True
    max_samples: int | None = Field(default=None, gt=0)
    max_duration_hours: float | None = Field(default=None, gt=0)
    languages: list[Literal["kk", "ru", "kk_ru", "other"]] | None = None
    filters: dict[str, list[str]] = Field(default_factory=dict)


class DatasetBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    project_root: str
    output_dir: str
    sources: list[SourceBuild] = Field(min_length=1)
    include_roles: list[Literal["primary", "external_sanity"]] = Field(
        default_factory=lambda: ["primary"]
    )
    languages: list[Literal["kk", "ru", "kk_ru", "other"]] | None = None
    max_samples: int | None = Field(default=None, gt=0)
    max_duration_hours: float | None = Field(default=None, gt=0)
    duplicate_policy: Literal["report", "error"] = "report"
    sample_rate: int = Field(default=16000, ge=8000, le=96000)
    max_duration_seconds: float = Field(default=120, gt=0, le=3600)
    max_file_bytes: int = Field(default=52428800, gt=0, le=1073741824)
    research_partition: ResearchPartition = "train"


def _limited(samples: list[AdaptedSample], source: SourceBuild) -> list[AdaptedSample]:
    selected = sorted(samples, key=lambda item: item.row.sample_id)
    for field, accepted in source.filters.items():
        if field not in {"language", "label", "speaker_id", "source_id", "generator_id"}:
            raise ValueError(f"Unsupported canonical source filter: {field}")
        selected = [item for item in selected if str(getattr(item.row, field) or "") in accepted]
    if source.languages:
        selected = [item for item in selected if item.row.language in source.languages]
    if source.max_samples is not None:
        selected = selected[: source.max_samples]
    if source.max_duration_hours is not None:
        limit = source.max_duration_hours * 3600
        total = 0.0
        bounded = []
        for item in selected:
            duration = item.row.duration_sec or 0.0
            if total + duration > limit:
                break
            bounded.append(item)
            total += duration
        selected = bounded
    return selected


def _distribution(rows, field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(row, field) or "unknown") for row in rows).items()))


def build_dataset(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = DatasetBuildConfig.model_validate(load_yaml(config_path))
    root = (config_path.parent / config.project_root).resolve()
    output = (root / config.output_dir).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new dataset version, never overwrite")

    selected: list[AdaptedSample] = []
    source_metadata = []
    source_config_hashes = {}
    for entry in config.sources:
        if not entry.enabled:
            continue
        source_config_path = (config_path.parent / entry.config).resolve()
        adapter = adapter_from_config(source_config_path)
        metadata = adapter.config.metadata
        if metadata.intended_role not in config.include_roles:
            continue
        source_metadata.append(metadata.model_dump(mode="json"))
        source_config_hashes[metadata.source_id] = sha256(source_config_path)
        selected.extend(_limited(adapter.iter_samples(), entry))
    if config.languages:
        selected = [item for item in selected if item.row.language in config.languages]
    selected.sort(key=lambda item: item.row.sample_id)
    if config.max_samples is not None:
        selected = selected[: config.max_samples]
    if config.max_duration_hours is not None:
        maximum = config.max_duration_hours * 3600
        total = 0.0
        duration_selected = []
        for item in selected:
            duration = item.row.duration_sec or 0.0
            if total + duration > maximum:
                break
            duration_selected.append(item)
            total += duration
        selected = duration_selected
    if not selected:
        raise ValueError("Dataset build selected no samples")
    validate_rows([item.row.model_dump(mode="json") for item in selected])
    validate_generator_roles([item.row for item in selected], config.research_partition)

    hashes: dict[str, list[str]] = defaultdict(list)
    for item in selected:
        hashes[item.row.source_sha256 or ""].append(item.row.sample_id)
    duplicate_groups = [ids for digest, ids in hashes.items() if digest and len(ids) > 1]
    if duplicate_groups and config.duplicate_policy == "error":
        raise ValueError(f"Exact duplicate audio content found: {duplicate_groups}")

    output.mkdir(parents=True, exist_ok=False)
    (output / "audio").mkdir()
    started = {
        **snapshot(root),
        "status": "incomplete",
        "config": config.model_dump(mode="json"),
        "config_sha256": sha256(config_path),
        "source_config_hashes": source_config_hashes,
        "sources": source_metadata,
    }
    write_json(output / "started.json", started)
    prepared = []
    for item in selected:
        target = output / "audio" / f"{item.row.sample_id}.wav"
        metadata = preprocess(
            item.source_path,
            target,
            sample_rate=config.sample_rate,
            max_duration_seconds=config.max_duration_seconds,
            max_file_bytes=config.max_file_bytes,
        )
        prepared.append(
            item.row.model_copy(
                update={
                    "path": target.relative_to(output).as_posix(),
                    "processed_sha256": sha256(target),
                    **metadata,
                }
            )
        )
    write_manifest(output / "manifest.jsonl", prepared)
    write_manifest(output / "manifest.parquet", prepared)
    duration_by_language: dict[str, float] = defaultdict(float)
    duration_by_source: dict[str, float] = defaultdict(float)
    for row in prepared:
        duration_by_language[row.language] += row.duration_sec or 0.0
        duration_by_source[row.source_dataset] += row.duration_sec or 0.0
    missing = Counter(name for row in prepared for name in row.missing_metadata)
    for field in ("speaker_id", "source_id", "sample_rate", "duration_sec", "codec"):
        absent = sum(getattr(row, field) in {None, "", "unknown"} for row in prepared)
        missing[field] = max(missing[field], absent)
    speakers = {row.speaker_id for row in prepared if "speaker_id" not in row.missing_metadata}
    report = {
        "dataset_id": config.dataset_id,
        "dataset_version": config.dataset_version,
        "manifest_sha256": sha256(output / "manifest.parquet"),
        "number_of_samples": len(prepared),
        "total_duration_sec": sum(row.duration_sec or 0.0 for row in prepared),
        "duration_by_language_sec": dict(sorted(duration_by_language.items())),
        "duration_by_source_sec": dict(sorted(duration_by_source.items())),
        "sample_count_by_language": _distribution(prepared, "language"),
        "sample_count_by_source": _distribution(prepared, "source_dataset"),
        "verified_code_switch_samples": sum(row.language == "kk_ru" for row in prepared),
        "number_of_known_speakers": len(speakers),
        "sample_rate_distribution": _distribution(prepared, "sample_rate"),
        "codec_format_distribution": _distribution(prepared, "codec"),
        "missing_metadata_counts": dict(sorted(missing.items())),
        "duplicate_content": {
            "groups": duplicate_groups,
            "group_count": len(duplicate_groups),
            "duplicate_sample_count": sum(len(ids) - 1 for ids in duplicate_groups),
        },
        "invalid_file_count": 0,
        "sources": source_metadata,
    }
    write_json(output / "dataset_report.json", report)
    write_json(
        output / "preparation.json",
        {
            **started,
            "status": "complete",
            "dataset_version": config.dataset_version,
            "manifest_hashes": {
                name: sha256(output / name) for name in ("manifest.jsonl", "manifest.parquet")
            },
            "dataset_report_sha256": sha256(output / "dataset_report.json"),
        },
    )
    return output
