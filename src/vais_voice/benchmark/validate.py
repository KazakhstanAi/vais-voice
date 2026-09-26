"""Validate a benchmark snapshot and all local public audio references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vais_voice.benchmark.schema import BenchmarkSnapshot


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(snapshot_path: Path, repository: Path) -> BenchmarkSnapshot:
    snapshot = BenchmarkSnapshot.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
    for sample in snapshot.samples:
        public_file = repository / sample.audio_path
        if not public_file.is_file():
            raise ValueError(f"missing public audio: {sample.audio_path}")
        if public_file.stat().st_size != sample.bytes:
            raise ValueError(f"byte count mismatch: {sample.audio_path}")
        if file_sha256(public_file) != sample.sha256:
            raise ValueError(f"SHA-256 mismatch: {sample.audio_path}")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args()
    validated = validate_snapshot(args.snapshot, args.repository)
    print(json.dumps({"status": "ok", "samples": len(validated.samples)}))


if __name__ == "__main__":
    main()
