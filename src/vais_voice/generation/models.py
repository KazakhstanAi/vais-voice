"""Validated provenance and immutable records for synthetic generation."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

GeneratorRole = Literal["train", "validation", "unseen_test", "external_challenge"]
CommercialStatus = Literal["permissive", "research_only", "needs_review"]


class GeneratorProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    generator_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    display_name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_id: str | None = None
    model_revision: str | None = None
    model_version: str | None = None
    languages: list[Literal["kk", "ru"]] = Field(min_length=1)
    license: str | None = None
    license_url: str | None = None
    homepage_url: str | None = None
    model_url: str | None = None
    citation: str | None = None
    intended_role: GeneratorRole
    commercial_use_status: CommercialStatus
    generator_model: str | None = None
    generator_runtime: str | None = None
    voice_repository: str | None = None
    voice_repository_revision: str | None = None
    voice_repository_license: str | None = None
    source_dataset: str | None = None
    source_dataset_license: str | None = None
    source_dataset_license_url: str | None = None
    retrieved_at: str | None = None
    notes: str | None = None

    @field_validator("languages")
    @classmethod
    def unique_languages(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Generator languages must be unique")
        return value


class GeneratorSpeaker(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    speaker_id: int = Field(ge=0)
    name: str = Field(min_length=1)


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str = Field(min_length=1)
    provenance: GeneratorProvenance
    runtime: dict[str, Any] = Field(default_factory=dict)
    voices: list[str] = Field(default_factory=list)
    voice_id: str | None = None
    speakers: list[GeneratorSpeaker] = Field(default_factory=list)
    selected_speaker_id: int | None = Field(default=None, ge=0)
    selected_speaker_name: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    supports_seed: bool = False
    license_review_acknowledged: bool = False
    verification_status: Literal["verified", "runtime_unverified", "infrastructure_test_only"]
    quality_gate: Literal["pending", "pass", "warning", "fail"] = "pending"
    training_eligible: bool = False
    diagnostic_only: bool = False
    candidate: bool = False
    quality_reason: str | None = None

    @model_validator(mode="after")
    def check_test_adapter(self) -> "GeneratorConfig":
        if (self.adapter == "fake_sine") != (
            self.verification_status == "infrastructure_test_only"
        ):
            raise ValueError("fake_sine is the only infrastructure_test_only adapter")
        selected = [
            speaker for speaker in self.speakers if speaker.speaker_id == self.selected_speaker_id
        ]
        if self.selected_speaker_id is not None:
            if len(selected) != 1 or selected[0].name != self.selected_speaker_name:
                raise ValueError("Selected speaker ID/name must match the configured speaker map")
        elif self.selected_speaker_name is not None:
            raise ValueError("selected_speaker_name requires selected_speaker_id")
        if self.training_eligible and self.quality_gate != "pass":
            raise ValueError("Only quality-gate pass generators may be training eligible")
        if self.training_eligible and self.diagnostic_only:
            raise ValueError("A generator cannot be training eligible and diagnostic only")
        if self.candidate and self.diagnostic_only:
            raise ValueError("A generator cannot be both a candidate and diagnostic only")
        return self

    def require_generation_approval(self) -> None:
        provenance = self.provenance
        unresolved_license = not provenance.license or provenance.license.upper().startswith("TODO")
        if self.verification_status != "infrastructure_test_only" and (
            unresolved_license or provenance.commercial_use_status == "needs_review"
        ):
            if not self.license_review_acknowledged:
                raise ValueError(
                    f"Generator {provenance.generator_id!r} has unresolved licensing metadata; "
                    "set license_review_acknowledged only after an explicit review"
                )


class TextItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    text: str = Field(min_length=1)
    language: Literal["kk", "ru"]
    source_dataset: str = Field(min_length=1)
    source_record_id: str = Field(min_length=1)
    source_sample_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    split: Literal["train", "val", "test"] | None = None
    normalized_text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(pattern=r"^job_[a-f0-9]{24}$")
    text_id: str
    generator_id: str
    language: Literal["kk", "ru"]
    voice_id: str | None = None
    speaker_id: int | None = Field(default=None, ge=0)
    speaker_name: str | None = None
    seed: int | None = Field(default=None, ge=0)
    generation_params: dict[str, Any] = Field(default_factory=dict)
    intended_role: GeneratorRole
    output_relative_path: str


class GenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["completed", "skipped", "failed"]
    output_relative_path: str
    output_sha256: str | None = None
    sample_rate: int | None = None
    duration_sec: float | None = None
    codec: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PronunciationReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    language: Literal["kk", "ru"]
    source_text: str
    voice: str
    speaker: str
    speaker_id: int | None = Field(default=None, ge=0)
    audio_path: str
    review_status: Literal["pending", "pass", "warning", "fail"] = "pending"
    review_notes: str = ""
    training_eligible: bool = False
    diagnostic_only: bool = False
    candidate: bool = False
    quality_tier: Literal["easy_spoof", "medium_spoof", "high_quality_spoof"] | None = None

    @model_validator(mode="after")
    def check_quality_decision(self) -> "PronunciationReviewItem":
        if self.training_eligible and self.review_status != "pass":
            raise ValueError("Only reviewed pass samples may be training eligible")
        if self.training_eligible and self.diagnostic_only:
            raise ValueError("A review cannot be training eligible and diagnostic only")
        if self.candidate and self.diagnostic_only:
            raise ValueError("A review cannot be both a candidate and diagnostic only")
        return self
