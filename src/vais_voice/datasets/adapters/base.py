"""Shared source-adapter contract and parsing helpers."""

import csv
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field

from vais_voice.datasets.schema import Sample
from vais_voice.datasets.source import SourceMetadata
from vais_voice.preprocessing.audio import SUPPORTED
from vais_voice.utils.io import contained_path, load_yaml, sha256


class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter: str
    local_root: str | None = None
    metadata: SourceMetadata
    metadata_file: str | None = None
    protocol_file: str | None = None
    audio_dir: str = "."
    delimiter: str | None = None
    columns: dict[str, str] = Field(default_factory=dict)
    language: str | None = None
    language_map: dict[str, str] = Field(default_factory=dict)
    filters: dict[str, list[str]] = Field(default_factory=dict)


@dataclass(frozen=True)
class AdaptedSample:
    row: Sample
    source_path: Path


def deterministic_id(source_id: str, record_id: str) -> str:
    digest = hashlib.sha256(f"{source_id}\0{record_id}".encode()).hexdigest()[:20]
    return f"{source_id}_{digest}"


def load_source_config(path: Path) -> AdapterConfig:
    return AdapterConfig.model_validate(load_yaml(path))


def read_table(path: Path, delimiter: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Required source metadata file not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        return frame.where(pd.notna(frame), None).to_dict("records")
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def inspect_audio(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported audio extension: {path.suffix}")
    try:
        info = sf.info(path)
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Invalid audio file {path}: {exc}") from exc
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Empty or invalid audio file: {path}")
    return {
        "codec": path.suffix.lower().lstrip("."),
        "sample_rate": info.samplerate,
        "duration_sec": info.duration,
        "source_sha256": sha256(path),
    }


class DatasetAdapter(ABC):
    def __init__(self, config: AdapterConfig, config_path: Path):
        self.config = config
        self.config_path = config_path.resolve()
        if not config.local_root:
            raise ValueError(f"Source {config.metadata.source_id!r} requires local_root")
        root = Path(config.local_root)
        self.root = root.resolve() if root.is_absolute() else (config_path.parent / root).resolve()

    def validate(self) -> None:
        self.config.metadata.require_license()
        if not self.root.is_dir():
            raise ValueError(
                f"Local dataset root not found for {self.config.metadata.source_id}: {self.root}. "
                "No dataset is downloaded automatically."
            )

    def source_file(self, relative: str) -> Path:
        return contained_path(self.root, relative)

    def audio_file(self, relative: str) -> Path:
        return contained_path(self.source_file(self.config.audio_dir), relative)

    @abstractmethod
    def iter_samples(self) -> list[AdaptedSample]:
        """Return deterministic canonical rows without changing source files."""
