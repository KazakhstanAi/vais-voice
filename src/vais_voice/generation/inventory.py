"""Build a stable text inventory from canonical real-speech manifests."""

import json
from pathlib import Path

from vais_voice.datasets.manifest import read_manifest
from vais_voice.datasets.text import normalize_text, text_identity
from vais_voice.generation.models import TextItem


def extract_text_inventory(manifest: Path, *, languages: set[str] | None = None) -> list[TextItem]:
    accepted = languages or {"kk", "ru"}
    if not accepted <= {"kk", "ru"}:
        raise ValueError("Synthetic v0.1 supports only kk and ru; kk_ru is deferred")
    items: dict[str, TextItem] = {}
    for row in read_manifest(manifest):
        if row.label != "real" or row.language not in accepted or not row.transcript:
            continue
        normalized = normalize_text(row.transcript)
        if not normalized:
            continue
        text_id = row.text_id or text_identity(row.language, normalized)
        item = TextItem(
            text_id=text_id,
            text=row.transcript,
            language=row.language,
            source_dataset=row.source_dataset,
            source_record_id=row.source_record_id or row.source_id,
            source_sample_id=row.sample_id,
            split=row.split,
            normalized_text=normalized,
            metadata={"source_sample_ids": [row.sample_id]},
        )
        previous = items.get(text_id)
        if previous:
            if previous.normalized_text != normalized or previous.language != row.language:
                raise ValueError(f"Conflicting content for text_id {text_id}")
            if previous.split != row.split:
                raise ValueError(f"Text leakage: {text_id} crosses protected splits")
            previous.metadata["source_sample_ids"].append(row.sample_id)
        else:
            items[text_id] = item
    if not items:
        raise ValueError("No validated kk/ru transcripts found in canonical manifest")
    return [items[key] for key in sorted(items)]


def write_inventory(path: Path, items: list[TextItem]) -> None:
    if path.exists():
        raise ValueError(f"Text inventory already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for item in items:
            stream.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")


def read_inventory(path: Path) -> list[TextItem]:
    if not path.is_file():
        raise ValueError(f"Text inventory not found: {path}")
    items = [
        TextItem.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError("Text inventory is empty")
    seen: dict[str, TextItem] = {}
    for item in items:
        if item.text_id in seen:
            raise ValueError(f"Duplicate text_id: {item.text_id}")
        seen[item.text_id] = item
    return items
