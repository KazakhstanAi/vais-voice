from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmark" / "kzru-enterprise-v0.1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(DATASET.glob("*.jsonl")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_lines(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (DATASET / name).read_text(encoding="utf-8").splitlines() if line]


def load_benchmark() -> dict[str, Any]:
    metadata = yaml.safe_load((DATASET / "benchmark.yaml").read_text(encoding="utf-8"))
    cases = load_lines("cases.jsonl")
    documents = load_lines("documents.jsonl")
    metadata["case_count"] = len(cases)
    metadata["dataset_hash"] = dataset_hash()
    metadata["documents"] = [{key: value for key, value in item.items() if key != "text"} for item in documents]
    return metadata


def load_cases() -> list[dict[str, Any]]:
    return load_lines("cases.jsonl")


def load_documents() -> list[dict[str, Any]]:
    return load_lines("documents.jsonl")
