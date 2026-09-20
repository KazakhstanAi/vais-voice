"""Canonical metadata. kk is ISO 639-1 for Kazakh."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    sample_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    path: str = Field(min_length=1)
    label: Literal["real", "synthetic"]
    language: Literal["kk", "ru", "kk_ru", "other"]
    speaker_id: str = Field(min_length=1)
    related_speaker_ids: list[str] = Field(default_factory=list)
    source_dataset: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    generator_id: str | None = None
    license: str = Field(min_length=1)
    original_path: str = Field(min_length=1)
    parent_sample_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    rights_reference: str = Field(min_length=1)
    usage_permission: Literal["approved"]
    codec: str = "unknown"
    condition: str = "clean"
    split: Literal["train", "val", "test"] | None = None
    source_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    processed_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    sample_rate: int | None = Field(default=None, gt=0)
    duration_sec: float | None = Field(default=None, gt=0)
    attack_id: str | None = None
    source_record_id: str | None = None
    transcript: str | None = None
    normalized_text: str | None = None
    text_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    source_text_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]+$")
    generator_family: str | None = None
    generator_version: str | None = None
    generator_model: str | None = None
    generator_runtime: str | None = None
    voice_id: str | None = None
    generator_speaker_id: int | None = Field(default=None, ge=0)
    generator_speaker_name: str | None = None
    generation_seed: int | None = None
    generation_params: dict[str, Any] | None = None
    intended_role: Literal["train", "validation", "unseen_test", "external_challenge"] | None = None
    quality_gate: Literal["pending", "pass", "warning", "fail"] | None = None
    training_eligible: bool | None = None
    diagnostic_only: bool | None = None
    missing_metadata: list[str] = Field(default_factory=list)

    @field_validator("related_speaker_ids")
    @classmethod
    def clean_speakers(cls, value: list[str]) -> list[str]:
        if any(not speaker.strip() for speaker in value):
            raise ValueError("Related speaker identifiers must be nonempty")
        return [speaker.strip() for speaker in value]

    @field_validator("missing_metadata")
    @classmethod
    def clean_missing_metadata(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Missing-metadata field names must be nonempty")
        return sorted(set(item.strip() for item in value))

    @model_validator(mode="after")
    def check_provenance(self) -> "Sample":
        if self.label == "synthetic" and not self.generator_id:
            raise ValueError("Synthetic samples require generator_id family/version")
        if self.label == "real" and self.generator_id is not None:
            raise ValueError("Real samples must have generator_id=null")
        if self.label == "real" and any(
            value is not None
            for value in (
                self.generator_family,
                self.generator_version,
                self.generator_model,
                self.generator_runtime,
                self.voice_id,
                self.generator_speaker_id,
                self.generator_speaker_name,
                self.generation_seed,
                self.generation_params,
                self.intended_role,
                self.quality_gate,
                self.training_eligible,
                self.diagnostic_only,
            )
        ):
            raise ValueError("Real samples must not carry generator metadata")
        if self.source_text_id and self.text_id and self.source_text_id != self.text_id:
            raise ValueError("text_id and source_text_id must identify the same text")
        if self.parent_sample_id == self.sample_id:
            raise ValueError("A sample cannot be its own parent")
        return self
