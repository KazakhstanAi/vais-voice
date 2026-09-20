"""Adapter for local FLEURS TSV/CSV/Parquet metadata exports."""

from pathlib import Path

from vais_voice.datasets.adapters.ksc2 import KSC2Adapter


class FleursAdapter(KSC2Adapter):
    """FLEURS tabular exports share the configurable real-speech parser."""

    def read_records(self, audio_root: Path) -> list[dict]:
        if self.config.metadata.layout_version != "google_fleurs_tsv_v1":
            return super().read_records(audio_root)
        metadata_file = self.source_file(self.config.metadata_file or "")
        if not metadata_file.is_file():
            raise ValueError(f"Required FLEURS metadata file not found: {metadata_file}")
        records = []
        for number, line in enumerate(metadata_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 7:
                raise ValueError(
                    f"Invalid official FLEURS TSV row {number}: expected 7 fields, "
                    f"received {len(fields)}"
                )
            record_id, filename, raw_text, text, _, num_samples, gender = fields
            records.append(
                {
                    "id": record_id,
                    "file": filename,
                    "raw_transcription": raw_text,
                    "transcription": text,
                    "transcript": text,
                    "num_samples": num_samples,
                    "gender": gender,
                }
            )
        return records

    def iter_samples(self):
        if not self.config.metadata_file:
            raise ValueError("FLEURS requires metadata_file (TSV, CSV, or Parquet)")
        if self.config.language not in {"kk", "ru"}:
            raise ValueError("FLEURS source config requires an explicit kk or ru language")
        return super().iter_samples()
