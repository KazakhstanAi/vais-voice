import importlib.util
import json
from pathlib import Path

import pytest

from vais_voice.datasets.manifest import read_manifest
from vais_voice.datasets.split import assign_splits


def test_twenty_sample_cli_pipeline(tmp_path):
    script = Path(__file__).parents[1] / "scripts/smoke_pipeline.py"
    spec = importlib.util.spec_from_file_location("smoke", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "smoke"
    module.run(root)
    rows = read_manifest(root / "processed/manifest.split.parquet")
    assert len(rows) == 20
    assert sum(r.label == "real" for r in rows) == 10
    for split in ("train", "val", "test"):
        assert all(r.split == split for r in read_manifest(root / f"processed/{split}.parquet"))
    output = root / "runs/dummy"
    for name in (
        "metrics.json",
        "predictions.csv",
        "config.yaml",
        "dataset_snapshot.json",
        "experiment.json",
    ):
        assert (output / name).is_file()
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["overall"]["status"] == "degenerate_constant_scores"
    assert metrics["overall"]["roc_auc"] is None
    assert metrics["overall"]["eer"] is None
    assert metrics["overall"]["fpr_at_tpr95"] is None
    assert "DUMMY_INFRASTRUCTURE_ONLY" in metrics["inference"]["warning"]


def test_parent_lineage_protects_split(corpus):
    root, _ = corpus
    rows = read_manifest(root / "input.jsonl")
    rows[4] = rows[4].model_copy(update={"parent_sample_id": rows[0].sample_id})
    with pytest.raises(ValueError, match="leakage"):
        assign_splits(rows, seed=42, train_fraction=0.7, val_fraction=0.15, unseen_generators=[])


def test_orphan_parent_rejected(corpus):
    root, _ = corpus
    path = root / "input.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["parent_sample_id"] = "missing"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        read_manifest(path)
