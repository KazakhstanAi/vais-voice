import hashlib
from pathlib import Path

from vais_voice.benchmark.schema import BenchmarkSnapshot, DimensionEvidence
from vais_voice.benchmark.validate import validate_snapshot

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "artifacts/benchmark/kzru-voice-benchmark-pilot-v0.2.json"


def test_dimension_missing_data_semantics() -> None:
    DimensionEvidence(value="not_evaluated", evidenceType="not_evaluated")


def test_snapshot_schema_and_audio_hashes() -> None:
    snapshot = validate_snapshot(SNAPSHOT, ROOT)
    assert isinstance(snapshot, BenchmarkSnapshot)
    assert snapshot.status == "research"
    assert snapshot.runs[-1].status == "planned"


def test_snapshot_sidecar_hash() -> None:
    expected = (SNAPSHOT.with_suffix(".sha256")).read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == expected
