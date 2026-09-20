"""Strict JSONL / Parquet serialization with a shared canonical schema."""

import json
from pathlib import Path

import pandas as pd

from vais_voice.datasets.schema import Sample


def validate_rows(records: list[dict]) -> list[Sample]:
    rows = []
    seen = set()
    for number, record in enumerate(records, 1):
        try:
            row = Sample.model_validate(record)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid manifest row {number}: {exc}") from exc
        if row.sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {row.sample_id}")
        seen.add(row.sample_id)
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    parents = {row.sample_id: row.parent_sample_id for row in rows}
    for row in rows:
        visited = {row.sample_id}
        parent = row.parent_sample_id
        while parent:
            if parent not in parents:
                raise ValueError("parent_sample_id must resolve inside the manifest")
            if parent in visited:
                raise ValueError("Cyclic parent_sample_id lineage")
            visited.add(parent)
            parent = parents[parent]
    return rows


def read_manifest(path: Path) -> list[Sample]:
    if not path.is_file():
        raise ValueError(f"Manifest not found: {path}. No dataset is downloaded automatically.")
    if path.suffix == ".parquet":
        records = json.loads(pd.read_parquet(path).to_json(orient="records"))
        for record in records:
            params = record.get("generation_params")
            if isinstance(params, str):
                record["generation_params"] = json.loads(params)
    elif path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError("Manifest extension must be .jsonl or .parquet")
    return validate_rows(records)


def write_manifest(path: Path, rows: list[Sample], *, overwrite: bool = False) -> None:
    records = [row.model_dump(mode="json") for row in rows]
    if path.suffix == ".parquet":
        parquet_records = []
        for record in records:
            parquet_record = dict(record)
            if isinstance(parquet_record.get("generation_params"), dict):
                parquet_record["generation_params"] = json.dumps(
                    parquet_record["generation_params"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            parquet_records.append(parquet_record)
        with path.open("wb" if overwrite else "xb") as stream:
            pd.DataFrame(parquet_records).to_parquet(stream, index=False)
    elif path.suffix == ".jsonl":
        with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        raise ValueError("Manifest extension must be .jsonl or .parquet")
