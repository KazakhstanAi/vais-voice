"""Validate completion markers and physical prepared files."""

import json
from pathlib import Path

from vais_voice.datasets.manifest import read_manifest
from vais_voice.utils.io import contained_path, sha256


def verify_manifest(manifest: Path, preparation: Path) -> dict:
    prep = json.loads(preparation.read_text(encoding="utf-8"))
    expected = prep.get("manifest_hashes", {}).get(
        manifest.name, prep.get("prepared_manifest_sha256")
    )
    if prep.get("status") != "complete" or expected != sha256(manifest):
        raise ValueError("Prepared manifest checksum mismatch or incomplete preparation")
    return prep


def validate_files(manifest: Path) -> None:
    for row in read_manifest(manifest):
        path = contained_path(manifest.parent, row.path)
        if not row.processed_sha256 or sha256(path) != row.processed_sha256:
            raise ValueError(f"Prepared audio missing or changed: {row.sample_id}")
