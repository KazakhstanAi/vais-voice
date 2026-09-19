import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Mathematical waveforms ONLY: pipeline fixtures, not genuine/synthetic speech evidence."""
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for group, split in enumerate(("train", "val", "test")):
        for label_index, label in enumerate(("real", "synthetic")):
            sample_id = f"s{group}_{label_index}"
            signal = 0.1 * np.sin(
                2 * np.pi * (200 + group * 31 + label_index * 13) * np.arange(800) / 8000
            )
            sf.write(raw / f"{sample_id}.wav", signal, 8000)
            rows.append(
                {
                    "sample_id": sample_id,
                    "path": f"{sample_id}.wav",
                    "label": label,
                    "language": "kk_ru",
                    "speaker_id": f"fixture_speaker_{group}",
                    "source_dataset": "software_fixture",
                    "original_path": f"{sample_id}.wav",
                    "source_id": f"fixture_source_{group}",
                    "generator_id": "test_fixture_v1" if label_index else None,
                    "license": "test-only mathematical fixture",
                    "rights_reference": "pytest generated tone; not speech",
                    "usage_permission": "approved",
                    "split": split,
                }
            )
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = tmp_path / "prepare.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset_version": "SOFTWARE_TEST_FIXTURE_NOT_SPEECH",
                "split_version": "fixture_v1",
                "seed": 42,
                "project_root": ".",
                "manifest": "input.jsonl",
                "audio_root": "raw",
                "output_dir": "processed",
                "sample_rate": 16000,
                "max_duration_seconds": 1,
                "max_file_bytes": 100000,
                "train_fraction": 0.7,
                "val_fraction": 0.15,
                "unseen_generators": [],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, config
