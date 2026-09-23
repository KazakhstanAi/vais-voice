import json

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from vais_voice.data.compose import compose_detector_corpus
from vais_voice.datasets.manifest import Sample, read_manifest, write_manifest
from vais_voice.datasets.split import assign_splits, connected_groups
from vais_voice.preprocessing.audio import preprocess
from vais_voice.utils.io import contained_path


def test_unknown_fields_and_missing_synthetic_provenance(corpus):
    root, _ = corpus
    row = read_manifest(root / "input.jsonl")[1].model_dump()
    row["generator_id"] = None
    with pytest.raises(ValidationError):
        Sample.model_validate(row)
    row["generator_id"] = "fixture"
    row["secret"] = "must_not_be_silently_accepted"
    with pytest.raises(ValidationError):
        Sample.model_validate(row)


def test_duplicate_ids(corpus):
    root, _ = corpus
    path = root / "input.jsonl"
    line = path.read_text().splitlines()[0]
    path.write_text(line + "\n" + line)
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(path)


def test_speaker_source_and_hash_leakage(corpus):
    root, _ = corpus
    rows = read_manifest(root / "input.jsonl")
    for update in (
        {"speaker_id": rows[0].speaker_id},
        {"source_id": rows[0].source_id},
    ):
        changed = list(rows)
        changed[2] = rows[2].model_copy(update=update)
        with pytest.raises(ValueError, match="leakage"):
            assign_splits(
                changed, seed=42, train_fraction=0.7, val_fraction=0.15, unseen_generators=[]
            )
    rows[0] = rows[0].model_copy(update={"source_sha256": "a" * 64})
    rows[2] = rows[2].model_copy(update={"source_sha256": "a" * 64})
    with pytest.raises(ValueError, match="leakage"):
        assign_splits(rows, seed=42, train_fraction=0.7, val_fraction=0.15, unseen_generators=[])


def test_same_speaker_cannot_cross_train_and_test(corpus):
    root, _ = corpus
    rows = read_manifest(root / "input.jsonl")
    rows[4] = rows[4].model_copy(update={"speaker_id": rows[0].speaker_id})
    with pytest.raises(ValueError, match="leakage"):
        assign_splits(rows, seed=42, train_fraction=0.7, val_fraction=0.15, unseen_generators=[])


def test_transitive_grouping(corpus):
    root, _ = corpus
    rows = read_manifest(root / "input.jsonl")[:3]
    rows[2] = rows[2].model_copy(update={"speaker_id": rows[1].speaker_id})
    assert len(connected_groups(rows)) == 1


def test_same_unseen_generator_cannot_cross_train_and_test(corpus):
    root, _ = corpus
    with pytest.raises(ValueError, match="Unseen"):
        assign_splits(
            read_manifest(root / "input.jsonl"),
            seed=42,
            train_fraction=0.7,
            val_fraction=0.15,
            unseen_generators=["test_fixture_v1"],
        )


def test_partial_supplied_split_rejected(corpus):
    root, _ = corpus
    rows = read_manifest(root / "input.jsonl")
    rows[0] = rows[0].model_copy(update={"split": None})
    with pytest.raises(ValueError, match="every row"):
        assign_splits(rows, seed=42, train_fraction=0.7, val_fraction=0.15, unseen_generators=[])


def test_automatic_splits_deterministic_and_order_independent(corpus):
    root, _ = corpus
    base = read_manifest(root / "input.jsonl")[:2]
    rows = []
    for group in range(60):
        for label_index, row in enumerate(base):
            rows.append(
                row.model_copy(
                    update={
                        "sample_id": f"g{group}_{label_index}",
                        "speaker_id": f"p{group}",
                        "source_id": f"source{group}",
                        "split": None,
                    }
                )
            )
    kwargs = {"seed": 42, "train_fraction": 0.7, "val_fraction": 0.15, "unseen_generators": []}
    first = assign_splits(rows, **kwargs)
    second = assign_splits(list(reversed(rows)), **kwargs)
    assert {r.sample_id: r.split for r in first} == {r.sample_id: r.split for r in second}
    assert {r.split for r in first} == {"train", "val", "test"}


def test_known_synthetic_voice_can_cross_splits_while_text_pairs_stay_together(corpus):
    root, _ = corpus
    base_real, base_synthetic = read_manifest(root / "input.jsonl")[:2]
    rows = []
    for group in range(60):
        text_id = f"text_{group}"
        rows.extend(
            [
                base_real.model_copy(
                    update={
                        "sample_id": f"real_{group}",
                        "speaker_id": f"real_speaker_{group}",
                        "source_id": f"real_source_{group}",
                        "text_id": text_id,
                        "split": None,
                    }
                ),
                base_synthetic.model_copy(
                    update={
                        "sample_id": f"synthetic_{group}",
                        "speaker_id": "one_known_tts_voice",
                        "source_id": f"synthetic_source_{group}",
                        "source_text_id": text_id,
                        "split": None,
                    }
                ),
            ]
        )
    assigned = assign_splits(
        rows, seed=42, train_fraction=0.7, val_fraction=0.15, unseen_generators=[]
    )
    by_text = {}
    for row in assigned:
        by_text.setdefault(row.source_text_id or row.text_id, set()).add(row.split)
    assert all(len(splits) == 1 for splits in by_text.values())
    assert {row.split for row in assigned} == {"train", "val", "test"}


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        contained_path(tmp_path, "../elsewhere.wav")


def test_stereo_downmix_and_resample(tmp_path):
    source = tmp_path / "stereo.wav"
    sf.write(source, np.ones((800, 2), dtype="float32") * 0.1, 8000)
    dest = tmp_path / "mono.wav"
    preprocess(source, dest, sample_rate=16000, max_duration_seconds=1, max_file_bytes=100000)
    info = sf.info(dest)
    assert info.channels == 1
    assert info.samplerate == 16000
    assert info.frames == 1600
    with pytest.raises(FileExistsError):
        preprocess(source, dest, sample_rate=16000, max_duration_seconds=1, max_file_bytes=100000)


@pytest.mark.parametrize(("duration", "size"), [(0.01, 100000), (1, 1)])
def test_audio_limits(corpus, duration, size):
    root, _ = corpus
    with pytest.raises(ValueError, match="limit"):
        preprocess(
            root / "raw/s0_0.wav",
            root / "out.wav",
            sample_rate=16000,
            max_duration_seconds=duration,
            max_file_bytes=size,
        )


def test_nonfinite_audio_rejected(tmp_path):
    source = tmp_path / "nan.wav"
    sf.write(source, np.array([0.1, np.nan]), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="Nonfinite"):
        preprocess(
            source,
            tmp_path / "out.wav",
            sample_rate=16000,
            max_duration_seconds=1,
            max_file_bytes=100000,
        )


def test_blank_speaker_rejected(corpus):
    root, _ = corpus
    row = json.loads((root / "input.jsonl").read_text().splitlines()[0])
    row["speaker_id"] = " "
    with pytest.raises(ValidationError):
        Sample.model_validate(row)


def test_compose_detector_corpus_filters_synthetic_quality(tmp_path):
    real_root = tmp_path / "real"
    synthetic_root = tmp_path / "synthetic"
    real_root.mkdir()
    synthetic_root.mkdir()
    sf.write(real_root / "real.wav", np.zeros(160), 16000)
    sf.write(synthetic_root / "pass.wav", np.zeros(240), 24000)
    sf.write(synthetic_root / "fail.wav", np.zeros(240), 24000)
    real = Sample(
        sample_id="real_1",
        path="real.wav",
        label="real",
        language="ru",
        speaker_id="real_speaker",
        source_dataset="fixture",
        source_id="real_source",
        license="fixture",
        original_path="real.wav",
        rights_reference="fixture",
        usage_permission="approved",
    )
    base_synthetic = Sample(
        sample_id="synthetic_pass",
        path="pass.wav",
        label="synthetic",
        language="ru",
        speaker_id="voice",
        source_dataset="fixture",
        source_id="synthetic_source",
        generator_id="generator",
        license="fixture",
        original_path="pass.wav",
        rights_reference="fixture",
        usage_permission="approved",
        intended_role="train",
        quality_gate="pass",
        training_eligible=True,
        diagnostic_only=False,
    )
    rejected = base_synthetic.model_copy(
        update={
            "sample_id": "synthetic_fail",
            "path": "fail.wav",
            "source_id": "failed_source",
            "original_path": "fail.wav",
            "quality_gate": "fail",
            "training_eligible": False,
            "diagnostic_only": True,
        }
    )
    write_manifest(real_root / "manifest.jsonl", [real])
    write_manifest(synthetic_root / "manifest.jsonl", [base_synthetic, rejected])

    output = compose_detector_corpus(
        real_root / "manifest.jsonl", synthetic_root / "manifest.jsonl", tmp_path / "staged"
    )
    rows = read_manifest(output / "manifest.jsonl")
    assert {row.sample_id for row in rows} == {"real_1", "synthetic_pass"}
    assert all((output / row.path).is_file() and row.split is None for row in rows)
