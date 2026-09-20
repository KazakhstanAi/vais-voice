"""Build the bounded 5 kk + 5 ru Piper pilot inventory from pinned FLEURS TSV files."""

import argparse
import csv
from pathlib import Path

from vais_voice.datasets.adapters.base import deterministic_id
from vais_voice.datasets.text import normalize_text, text_identity
from vais_voice.generation.inventory import write_inventory
from vais_voice.generation.models import TextItem
from vais_voice.utils.io import sha256

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
EXPECTED_HASHES = {
    "kk": "618a8e83ab0b5fd1cf4c7dfb9f6777fc3dad145e9daf250e8698e56d69f944df",
    "ru": "a18a051553daba44130d323ff8c7387dcde64ce855ade317c53209c0922885dc",
}


def read_items(path: Path, language: str) -> list[TextItem]:
    digest = sha256(path)
    if digest != EXPECTED_HASHES[language]:
        raise ValueError(f"Pinned FLEURS {language} TSV checksum mismatch")
    source_dataset = f"fleurs_{language}"
    items = []
    with path.open(encoding="utf-8", newline="") as stream:
        for number, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(fields) != 7:
                raise ValueError(f"Invalid FLEURS row {number}: expected 7 fields")
            record_id, filename, raw_text, text, _, _, gender = fields
            normalized = normalize_text(text)
            if not normalized:
                continue
            items.append(
                TextItem(
                    text_id=text_identity(language, normalized),
                    text=text,
                    language=language,
                    source_dataset=source_dataset,
                    source_record_id=record_id,
                    source_sample_id=deterministic_id(source_dataset, record_id),
                    split="train",
                    normalized_text=normalized,
                    metadata={
                        "filename": filename,
                        "gender": gender,
                        "raw_transcription": raw_text,
                        "source_revision": FLEURS_REVISION,
                        "source_tsv_sha256": digest,
                    },
                )
            )
            if len(items) == 5:
                break
    if len(items) != 5:
        raise ValueError(f"Pinned FLEURS {language} TSV has fewer than five valid texts")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kk-tsv", type=Path, required=True)
    parser.add_argument("--ru-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = [*read_items(args.kk_tsv, "kk"), *read_items(args.ru_tsv, "ru")]
    write_inventory(args.output, items)
    print(args.output)


if __name__ == "__main__":
    main()
