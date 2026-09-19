"""Explicit, reviewable provenance for an external dataset source."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    homepage_url: str | None = None
    download_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    citation: str | None = None
    doi: str | None = None
    source_config: str | None = None
    track: str | None = None
    release_revision: str | None = None
    layout_version: str | None = None
    languages: list[Literal["kk", "ru", "kk_ru", "other"]] = Field(min_length=1)
    intended_role: Literal["primary", "external_sanity"]
    notes: str | None = None

    @field_validator("license")
    @classmethod
    def reject_placeholder_license(cls, value: str | None) -> str | None:
        if value is not None and value.upper().startswith("TODO"):
            raise ValueError("license must be verified or null, not a placeholder")
        return value

    def require_license(self) -> None:
        if not self.license:
            raise ValueError(
                f"Source {self.source_id!r} has no verified license; set license explicitly "
                "after reviewing the dataset terms"
            )
