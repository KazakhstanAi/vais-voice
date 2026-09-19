"""Adapter for reviewed local KSC2 exports/subsets."""

from pathlib import Path

from vais_voice.datasets.adapters.base import (
    AdaptedSample,
    DatasetAdapter,
    deterministic_id,
    inspect_audio,
    read_table,
)
from vais_voice.datasets.schema import Sample
from vais_voice.preprocessing.audio import SUPPORTED


class KSC2Adapter(DatasetAdapter):
    def read_records(self, audio_root: Path) -> list[dict]:
        if self.config.metadata_file:
            return read_table(self.source_file(self.config.metadata_file), self.config.delimiter)
        return [
            {"path": path.relative_to(audio_root).as_posix()}
            for path in sorted(audio_root.rglob("*"))
            if path.suffix.lower() in SUPPORTED
        ]

    def iter_samples(self) -> list[AdaptedSample]:
        self.validate()
        audio_root = self.source_file(self.config.audio_dir)
        if not audio_root.is_dir():
            raise ValueError(f"KSC2 audio directory not found: {audio_root}")
        records = self.read_records(audio_root)
        columns = {
            "path": "path",
            "speaker_id": "speaker_id",
            "source_id": "source_id",
            "language": "language",
            **self.config.columns,
        }
        result = []
        for number, record in enumerate(records, 1):
            if any(
                str(record.get(field, "")) not in accepted
                for field, accepted in self.config.filters.items()
            ):
                continue
            relative = str(record.get(columns["path"], "")).strip()
            if not relative:
                raise ValueError(f"KSC2 metadata row {number} has no audio path")
            source = self.audio_file(relative)
            if not source.is_file():
                raise ValueError(f"KSC2 audio not found for metadata row {number}: {source}")
            record_id = str(
                record.get(columns["source_id"]) or Path(relative).with_suffix("").as_posix()
            )
            sample_id = deterministic_id(self.config.metadata.source_id, record_id)
            raw_language = str(record.get(columns["language"]) or self.config.language or "kk")
            language = self.config.language_map.get(raw_language, raw_language)
            if language not in {"kk", "ru", "kk_ru", "other"}:
                raise ValueError(f"Unsupported KSC2 language {raw_language!r} in row {number}")
            speaker = str(record.get(columns["speaker_id"]) or f"unknown_{sample_id}")
            missing = [] if record.get(columns["speaker_id"]) else ["speaker_id"]
            result.append(
                AdaptedSample(
                    Sample(
                        sample_id=sample_id,
                        path=relative,
                        label="real",
                        language=language,
                        speaker_id=speaker,
                        source_dataset=self.config.metadata.source_id,
                        source_id=record_id,
                        generator_id=None,
                        license=self.config.metadata.license or "",
                        original_path=relative,
                        rights_reference=self.config.metadata.license_url
                        or self.config.metadata.homepage_url
                        or "configured source metadata",
                        usage_permission="approved",
                        source_record_id=record_id,
                        missing_metadata=missing,
                        **inspect_audio(source),
                    ),
                    source,
                )
            )
        return result
