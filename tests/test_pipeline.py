import json
import subprocess
import sys

import pytest
import yaml

from vais_voice.data.prepare import prepare
from vais_voice.datasets.manifest import read_manifest
from vais_voice.evaluation.report import evaluate_predictions
from vais_voice.train import preflight


def ready(corpus):
    root, config = corpus
    output = prepare(config)
    predictions = root / "predictions.csv"
    # Hand-authored fixture scores test report mechanics, NOT detection capability.
    predictions.write_text("sample_id,synthetic_score\ns2_0,0.4\ns2_1,0.6\n")
    return root, output, predictions


def test_prepare_and_report(corpus):
    root, output, predictions = ready(corpus)
    report = evaluate_predictions(
        output / "manifest.jsonl", predictions, output / "preparation.json", root / "report", root
    )
    assert report["dataset_version"] == "SOFTWARE_TEST_FIXTURE_NOT_SPEECH"
    assert report["overall"]["n"] == 2
    assert report["overall"]["roc_auc"] == 1
    assert report["slices"]["language"]["kk_ru"]["n"] == 2
    assert report["slices"]["generator_with_all_real_references"]["test_fixture_v1"]["n"] == 2
    assert (root / "report/report.md").is_file()
    with pytest.raises(FileExistsError):
        evaluate_predictions(
            output / "manifest.jsonl",
            predictions,
            output / "preparation.json",
            root / "report",
            root,
        )
    with pytest.raises(ValueError, match="already exists"):
        prepare(corpus[1])


def test_preflight_no_training(corpus):
    root, output, _ = ready(corpus)
    config = root / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment": "fixture",
                "seed": 42,
                "project_root": ".",
                "prepared_manifest": "processed/manifest.jsonl",
                "preparation_report": "processed/preparation.json",
                "run_dir": "run",
                "model": {"name": "not_implemented", "sample_rate": 16000},
            }
        )
    )
    run = preflight(config)
    report = json.loads((run / "preflight.json").read_text())
    assert report["checkpoint"] is None and report["metrics"] is None
    assert report["git_commit"] is None  # temporary fixture, not a repo
    row = read_manifest(output / "manifest.jsonl")[0]
    (output / row.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        preflight(config)


@pytest.mark.parametrize(
    "csv",
    [
        "sample_id,synthetic_score\ns2_0,0.1\n",
        "sample_id,synthetic_score\ns2_0,0.1\ns2_0,0.2\n",
        "sample_id,synthetic_score\ns0_0,0.1\ns2_1,0.2\n",
        "sample_id,synthetic_score\ns2_0,nan\ns2_1,0.2\n",
    ],
)
def test_prediction_coverage_validation(corpus, csv):
    root, output, predictions = ready(corpus)
    predictions.write_text(csv)
    with pytest.raises(ValueError):
        evaluate_predictions(
            output / "manifest.jsonl",
            predictions,
            output / "preparation.json",
            root / "report",
            root,
        )


def test_manifest_tampering_rejected(corpus):
    root, output, predictions = ready(corpus)
    manifest = output / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_predictions(
            manifest, predictions, output / "preparation.json", root / "report", root
        )


def test_missing_data_does_not_create_output(tmp_path):
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset_version": "v1",
                "split_version": "v1",
                "seed": 42,
                "project_root": ".",
                "manifest": "missing.jsonl",
                "audio_root": "raw",
                "output_dir": "processed",
                "sample_rate": 16000,
                "max_duration_seconds": 120,
                "max_file_bytes": 100000,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
            }
        )
    )
    with pytest.raises(ValueError, match="Manifest not found"):
        prepare(config)
    assert not (tmp_path / "processed").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["vais_voice.train", "--config", "configs/detection/baseline.yaml"],
        ["vais_voice.evaluate", "--checkpoint", "runs/baseline/best.pt"],
        ["vais_voice.generation.synthesize", "--text", "Сәлеметсіз бе"],
    ],
)
def test_future_commands_fail_explicitly_without_side_effects(args, tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        cwd=tmp_path,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 2
    assert not list(tmp_path.iterdir())
