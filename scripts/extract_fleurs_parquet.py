"""Extract a pinned FLEURS parquet split into the reviewed local adapter layout."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf


def extension(payload: bytes) -> str:
    if payload.startswith(b"RIFF"):
        return ".wav"
    if payload.startswith(b"fLaC"):
        return ".flac"
    if payload.startswith(b"OggS"):
        return ".ogg"
    raise ValueError("Unsupported embedded FLEURS audio container")


def extract(source: Path, destination: Path, split: str, maximum: int | None) -> int:
    if not source.is_file():
        raise ValueError(f"Parquet file not found: {source}")
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata = destination / "metadata.tsv"
    existing = metadata.read_text(encoding="utf-8").splitlines() if metadata.exists() else []
    existing_ids = {line.split("\t", 1)[0] for line in existing if line.strip()}
    lines: list[str] = []
    written = 0
    columns = [
        "id",
        "audio",
        "raw_transcription",
        "transcription",
        "num_samples",
        "gender",
    ]
    parquet = pq.ParquetFile(source)
    schema_names = parquet.schema_arrow.names
    if not set(columns).issubset(schema_names):
        raise ValueError(f"Unexpected FLEURS parquet schema: {schema_names}")
    for batch in parquet.iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            record_id = str(row["id"])
            audio = row["audio"]
            payload = audio.get("bytes") if isinstance(audio, dict) else None
            if not payload:
                raise ValueError(f"FLEURS row {record_id} has no embedded audio bytes")
            digest = hashlib.sha256(payload).hexdigest()
            qualified_id = f"{split}_{record_id}_{digest[:12]}"
            if qualified_id in existing_ids:
                continue
            suffix = extension(payload)
            filename = f"{qualified_id}{suffix}"
            target = audio_dir / filename
            with target.open("xb") as stream:
                stream.write(payload)
            info = sf.info(target)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"Invalid extracted audio: {target}")
            raw = str(row["raw_transcription"]).replace("\t", " ").replace("\n", " ")
            normalized = str(row["transcription"]).replace("\t", " ").replace("\n", " ")
            gender = str(row["gender"])
            lines.append(
                "\t".join(
                    [
                        qualified_id,
                        filename,
                        raw,
                        normalized,
                        "",
                        str(info.frames),
                        gender,
                    ]
                )
            )
            written += 1
            if maximum is not None and written >= maximum:
                break
        if maximum is not None and written >= maximum:
            break
    if lines:
        with metadata.open("a", encoding="utf-8", newline="") as stream:
            for line in lines:
                stream.write(line + "\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--maximum", type=int)
    args = parser.parse_args()
    try:
        print(f"Extracted {extract(args.source, args.destination, args.split, args.maximum)} rows")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Extraction failed: {exc}\n")


if __name__ == "__main__":
    main()
