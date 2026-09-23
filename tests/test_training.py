import json

import pytest
import yaml

from vais_voice.data.prepare import prepare


def test_logmel_fixture_training_checkpoint_and_inference(corpus):
    pytest.importorskip("torch")
    from vais_voice.detection.runtime import CheckpointDetector
    from vais_voice.detection.training import train_detector

    root, data_config = corpus
    output = prepare(data_config)
    config = root / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "experiment": "SOFTWARE_FIXTURE_NOT_A_DETECTOR",
                "seed": 7,
                "project_root": ".",
                "prepared_manifest": "processed/manifest.jsonl",
                "preparation_report": "processed/preparation.json",
                "run_dir": "training-run",
                "device": "cpu",
                "epochs": 2,
                "patience": 2,
                "amp": False,
                "model": {
                    "architecture": "logmel_cnn",
                    "sample_rate": 16000,
                    "max_duration_seconds": 0.1,
                    "n_mels": 24,
                    "hidden_dim": 32,
                    "ssl_bundle": "WAV2VEC2_BASE",
                },
                "optimizer": {"learning_rate": 0.001, "weight_decay": 0.0},
                "loader": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "require_training_eligible": False,
                },
                "unseen_generators": [],
            }
        ),
        encoding="utf-8",
    )
    run = train_detector(config)
    assert (run / "best.pt").is_file()
    assert (run / "benchmark.json").is_file()
    assert (run / "predictions.csv").is_file()
    report = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
    assert report["report_type"] == "trained_detector_benchmark"
    assert report["selection_rule"].startswith("maximum validation")
    assert report["counts"] == {"train": 2, "validation": 2, "test": 2}
    detector = CheckpointDetector(run / "best.pt", device="cpu")
    score = detector.predict_score(output / "audio" / "s2_0.wav")
    assert 0 <= score <= 1
    assert detector.metadata()["model_id"] == "SOFTWARE_FIXTURE_NOT_A_DETECTOR"
