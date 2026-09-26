"""Strict evidence schema for the VAIS KZ/RU benchmark pilot."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DimensionValue = Literal["pass", "warning", "fail", "not_evaluated", "planned", "derived"]
EvidenceType = Literal[
    "human_review", "automatic_metric", "derived_finding", "not_evaluated"
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class DimensionEvidence(StrictModel):
    value: DimensionValue
    evidenceType: EvidenceType
    source: str | None = None
    reviewers: int | None = Field(default=None, ge=1)
    sampleCount: int | None = Field(default=None, ge=1)
    note: str | None = None

    @model_validator(mode="after")
    def enforce_semantics(self) -> DimensionEvidence:
        if self.evidenceType == "not_evaluated" and self.value not in {
            "not_evaluated",
            "planned",
        }:
            raise ValueError("not_evaluated evidence cannot assert an observed result")
        if self.evidenceType == "human_review" and not self.source:
            raise ValueError("human_review evidence requires a source artifact")
        return self


class ModelRecord(StrictModel):
    model_id: str
    family: str
    provider: str
    revision: str | None = None
    runtime_id: str
    runtime_version: str
    language: Literal["kk", "ru"]
    voice: str | None = None
    license: str | None = None
    license_url: str | None = None
    intended_role: str
    verification_status: Literal["verified", "partially_verified", "unverified"]


class HumanReview(StrictModel):
    status: Literal["pass", "warning", "fail", "pending"]
    finding: str | None = None
    source: str
    reviewerCount: int | None = Field(default=None, ge=1)


class SampleRecord(StrictModel):
    sample_id: str
    prompt_id: str
    model_id: str
    runtime_id: str
    language: Literal["kk", "ru"]
    audio_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    channels: int = Field(gt=0)
    quality_gate: Literal["pass", "warning", "fail", "pending"]
    human_review: HumanReview | None = None
    evaluation_dimensions: dict[str, DimensionEvidence]
    source_artifact: str

    @model_validator(mode="after")
    def public_paths_are_portable(self) -> SampleRecord:
        if ":\\" in self.audio_path or self.audio_path.startswith("/"):
            raise ValueError("audio_path must be a portable repository-relative path")
        return self


class DatasetRecord(StrictModel):
    dataset_id: str
    version: str
    language: Literal["kk", "ru"]
    role: str
    counts: dict[str, int]
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: str | None = None
    limitations: list[str] = []


class RunRecord(StrictModel):
    run_id: str
    task: str
    status: Literal["complete", "planned", "no_go", "undefined"]
    model_id: str | None = None
    dataset_version: str | None = None
    metrics: dict[str, Any] = {}
    limitations: list[str] = []


class BenchmarkSnapshot(StrictModel):
    benchmarkId: str
    version: Literal["pilot-v0.2"]
    publishedAt: date
    protocolVersion: Literal["0.2"]
    status: Literal["research"]
    models: list[ModelRecord]
    runs: list[RunRecord]
    datasets: list[DatasetRecord]
    samples: list[SampleRecord]
    metrics: dict[str, Any]
    limitations: list[str]
    provenance: dict[str, Any]
    artifactHashes: dict[str, str]

    @model_validator(mode="after")
    def validate_references(self) -> BenchmarkSnapshot:
        model_ids = {model.model_id for model in self.models}
        if len(model_ids) != len(self.models):
            raise ValueError("model_id values must be unique")
        sample_ids = {sample.sample_id for sample in self.samples}
        if len(sample_ids) != len(self.samples):
            raise ValueError("sample_id values must be unique")
        missing = {sample.model_id for sample in self.samples} - model_ids
        if missing:
            raise ValueError(f"samples reference unknown models: {sorted(missing)}")
        return self
