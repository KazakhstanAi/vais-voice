"""Canonical metadata. kk is ISO 639-1 for Kazakh."""

from typing import Literal

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

    @field_validator("related_speaker_ids")
    @classmethod
    def clean_speakers(cls, value: list[str]) -> list[str]:
        if any(not speaker.strip() for speaker in value):
            raise ValueError("Related speaker identifiers must be nonempty")
        return [speaker.strip() for speaker in value]

    @model_validator(mode="after")
    def check_provenance(self) -> "Sample":
        if self.label == "synthetic" and not self.generator_id:
            raise ValueError("Synthetic samples require generator_id family/version")
        if self.label == "real" and self.generator_id is not None:
            raise ValueError("Real samples must have generator_id=null")
        if self.parent_sample_id == self.sample_id:
            raise ValueError("A sample cannot be its own parent")
        return self
