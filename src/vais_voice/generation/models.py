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
    notes: str | None = None

    @field_validator("languages")
    @classmethod
    def unique_languages(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Generator languages must be unique")
        return value


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str = Field(min_length=1)
    provenance: GeneratorProvenance
    runtime: dict[str, Any] = Field(default_factory=dict)
    voices: list[str] = Field(default_factory=list)
    generation_params: dict[str, Any] = Field(default_factory=dict)
    supports_seed: bool = False
    license_review_acknowledged: bool = False
    verification_status: Literal["verified", "runtime_unverified", "infrastructure_test_only"]

    @model_validator(mode="after")
    def check_test_adapter(self) -> "GeneratorConfig":
        if (self.adapter == "fake_sine") != (
            self.verification_status == "infrastructure_test_only"
        ):
            raise ValueError("fake_sine is the only infrastructure_test_only adapter")
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
