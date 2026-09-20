"""Deterministically plan synthesis without importing or running model code."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vais_voice.generation.inventory import read_inventory
from vais_voice.generation.models import GenerationJob, GeneratorRole
from vais_voice.generation.registry import load_generator_config
from vais_voice.utils.io import load_yaml, sha256, write_json
from vais_voice.utils.tracking import snapshot


class PlannedGenerator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: str
    enabled: bool = True
    voices: list[str] | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)


class GenerationPlanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    dataset_version: str
    project_root: str
    text_inventory: str
    output_dir: str
    languages: list[Literal["kk", "ru"]] = Field(min_length=1)
    max_texts_per_language: int = Field(gt=0)
    allowed_source_datasets: list[str] = Field(default_factory=list)
    allowed_generator_roles: list[GeneratorRole] = Field(min_length=1)
    generators: list[PlannedGenerator] = Field(min_length=1)
    seed_strategy: Literal["none", "per_job_hash"] = "per_job_hash"
    base_seed: int = Field(default=0, ge=0)
    output_format: Literal["wav"] = "wav"
    output_sample_rate: int | None = Field(default=None, ge=8000, le=96000)
    failure_policy: Literal["continue", "stop"] = "continue"
    existing_output_policy: Literal["skip", "fail", "regenerate"] = "skip"

    @model_validator(mode="after")
    def unique_languages(self) -> "GenerationPlanConfig":
        if len(self.languages) != len(set(self.languages)):
            raise ValueError("languages must be unique")
        return self


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _job_id(payload: dict[str, Any]) -> str:
    return f"job_{_canonical_hash(payload)[:24]}"


def _role_accepts_split(role: GeneratorRole, split: str | None) -> bool:
    if role == "train":
        return True
    if role == "validation":
        return split in {"val", "test"}
    if role in {"unseen_test", "external_challenge"}:
        return split == "test"
    return False


def create_plan(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = GenerationPlanConfig.model_validate(load_yaml(config_path))
    root = (config_path.parent / config.project_root).resolve()
    inventory_path = (root / config.text_inventory).resolve()
    output = (root / config.output_dir).resolve()
    if output.exists():
        raise ValueError("Generation plan output already exists; plans are immutable")
    items = read_inventory(inventory_path)
    selected = []
    for language in config.languages:
        candidates = [
            item
            for item in items
            if item.language == language
            and (
                not config.allowed_source_datasets
                or item.source_dataset in config.allowed_source_datasets
            )
        ]
        selected.extend(
            sorted(candidates, key=lambda item: item.text_id)[: config.max_texts_per_language]
        )
    if not selected:
        raise ValueError("Generation plan selected no text items")

    generator_configs: dict[str, dict[str, Any]] = {}
    generator_hashes: dict[str, str] = {}
    jobs: list[GenerationJob] = []
    for entry in config.generators:
        if not entry.enabled:
            continue
        generator_path = (config_path.parent / entry.config).resolve()
        generator = load_generator_config(generator_path)
        provenance = generator.provenance
        if provenance.generator_id in generator_configs:
            raise ValueError(f"Duplicate generator_id: {provenance.generator_id}")
        if provenance.intended_role not in config.allowed_generator_roles:
            raise ValueError(
                f"Generator {provenance.generator_id} role "
                f"{provenance.intended_role} is not allowed"
            )
        generator_configs[provenance.generator_id] = generator.model_dump(mode="json")
        generator_hashes[provenance.generator_id] = sha256(generator_path)
        voices = entry.voices if entry.voices is not None else generator.voices
        voice_options: list[str | None] = (
            [generator.voice_id] if generator.voice_id else voices or [None]
        )
        unknown = set(voices) - set(generator.voices)
        if unknown:
            raise ValueError(f"Unknown voices for {provenance.generator_id}: {sorted(unknown)}")
        params = {**generator.generation_params, **entry.generation_params}
        if config.output_sample_rate is not None:
            params.setdefault("sample_rate", config.output_sample_rate)
        for item in selected:
            if item.language not in provenance.languages or not _role_accepts_split(
                provenance.intended_role, item.split
            ):
                continue
            for voice in voice_options:
                immutable = {
                    "text_id": item.text_id,
                    "generator_id": provenance.generator_id,
                    "model_revision": provenance.model_revision,
                    "model_version": provenance.model_version,
                    "language": item.language,
                    "voice_id": voice,
                    "speaker_id": generator.selected_speaker_id,
                    "speaker_name": generator.selected_speaker_name,
                    "generation_params": params,
                    "intended_role": provenance.intended_role,
                }
                seed = (
                    int(_canonical_hash({**immutable, "base_seed": config.base_seed})[:16], 16)
                    if config.seed_strategy == "per_job_hash" and generator.supports_seed
                    else None
                )
                identity = {**immutable, "seed": seed}
                job_id = _job_id(identity)
                jobs.append(
                    GenerationJob(
                        job_id=job_id,
                        text_id=item.text_id,
                        generator_id=provenance.generator_id,
                        language=item.language,
                        voice_id=voice,
                        speaker_id=generator.selected_speaker_id,
                        speaker_name=generator.selected_speaker_name,
                        seed=seed,
                        generation_params=params,
                        intended_role=provenance.intended_role,
                        output_relative_path=(
                            f"audio/{provenance.generator_id}/{job_id}.{config.output_format}"
                        ),
                    )
                )
    jobs.sort(key=lambda job: job.job_id)
    if not jobs:
        raise ValueError("Generation plan produced no compatible jobs")
    role_partitions = {
        "train" if job.intended_role in {"train", "validation"} else job.intended_role
        for job in jobs
    }
    if len(role_partitions) != 1:
        raise ValueError(
            "A generation plan cannot mix train, protected unseen, and external challenge data"
        )
    if len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Duplicate generation job identity")

    output.mkdir(parents=True, exist_ok=False)
    jobs_path = output / "generation_jobs.jsonl"
    with jobs_path.open("x", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(json.dumps(job.model_dump(mode="json"), ensure_ascii=False) + "\n")
    plan_hash = sha256(jobs_path)
    write_json(
        output / "generation_plan.json",
        {
            **snapshot(root),
            "dataset_id": config.dataset_id,
            "dataset_version": config.dataset_version,
            "config": config.model_dump(mode="json"),
            "config_sha256": sha256(config_path),
            "text_inventory_path": str(inventory_path),
            "text_inventory_sha256": sha256(inventory_path),
            "generation_jobs_sha256": plan_hash,
            "number_of_planned_jobs": len(jobs),
            "text_items": [item.model_dump(mode="json") for item in selected],
            "generator_configs": generator_configs,
            "generator_config_hashes": generator_hashes,
        },
    )
    return jobs_path


def read_jobs(path: Path) -> list[GenerationJob]:
    jobs = [
        GenerationJob.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Generation plan is empty or contains duplicate jobs")
    return jobs
