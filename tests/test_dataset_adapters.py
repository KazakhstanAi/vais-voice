import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from pydantic import ValidationError

from vais_voice.data.prepare import prepare
from vais_voice.datasets.adapters import adapter_from_config
from vais_voice.datasets.adapters.base import deterministic_id
from vais_voice.datasets.manifest import read_manifest
from vais_voice.datasets.source import SourceMetadata


def tone(path: Path, frequency: int = 220) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    signal = 0.1 * np.sin(2 * np.pi * frequency * np.arange(800) / 8000)
    sf.write(path, signal, 8000)


def source_config(
    tmp_path: Path,
    name: str,
    *,
    adapter: str,
    root: Path,
    language: str,
    role: str = "primary",
    **extra,
) -> Path:
    metadata_extra = extra.pop("metadata_extra", {})
    path = tmp_path / f"{name}.yaml"
    value = {
        "adapter": adapter,
        "local_root": str(root),
        "language": language,
        "metadata": {
            "source_id": name,
            "name": name,
            "version": "fixture-v1",
            "homepage_url": None,
            "download_url": None,
            "license": "test-only fixture terms",
            "license_url": None,
            "citation": None,
            "languages": [language],
            "intended_role": role,
            "notes": "Synthetic technical fixture; not speech.",
            **metadata_extra,
        },
        **extra,
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


@pytest.fixture
def adapter_configs(tmp_path: Path) -> dict[str, Path]:
    ksc = tmp_path / "ksc"
    tone(ksc / "audio/a.wav")
    tone(ksc / "audio/b.wav", 250)
    (ksc / "metadata.csv").write_text(
        "file,speaker,utterance,lang\na.wav,k1,k-a,Kazakh\nb.wav,,k-b,kk_ru\n",
        encoding="utf-8",
    )
    ksc_config = source_config(
        tmp_path,
        "ksc2_fixture",
        adapter="ksc2",
        root=ksc,
        language="kk",
        metadata_file="metadata.csv",
        audio_dir="audio",
        columns={
            "path": "file",
            "speaker_id": "speaker",
            "source_id": "utterance",
            "language": "lang",
        },
        language_map={"Kazakh": "kk"},
    )

    configs = {"ksc": ksc_config}
    for language in ("kk", "ru"):
        fleurs = tmp_path / f"fleurs_{language}"
        # kk deliberately duplicates k-a under another name.
        tone(fleurs / "audio/sample.wav", 220 if language == "kk" else 330)
        (fleurs / "metadata.tsv").write_text(
            "1\tsample.wav\traw text\tnormalized text\tchars\t800\tOTHER\n",
            encoding="utf-8",
        )
        configs[f"fleurs_{language}"] = source_config(
            tmp_path,
            f"fleurs_{language}_fixture",
            adapter="fleurs",
            root=fleurs,
            language=language,
            metadata_file="metadata.tsv",
            audio_dir="audio",
            columns={"path": "file", "speaker_id": "speaker_id", "source_id": "id"},
            metadata_extra={
                "source_config": f"{language}_{'kz' if language == 'kk' else 'ru'}",
                "layout_version": "google_fleurs_tsv_v1",
            },
        )

    asv = tmp_path / "asv"
    tone(asv / "audio/bonafide.wav", 440)
    tone(asv / "audio/spoof.wav", 550)
    (asv / "protocol.txt").write_text(
        "speaker1 bonafide nocodec asvspoof - bonafide notrim eval - - - - -\n"
        "speaker2 spoof mp3m4a asvspoof A07 spoof notrim eval neural - - - -\n",
        encoding="utf-8",
    )
    configs["asv"] = source_config(
        tmp_path,
        "asv_fixture",
        adapter="asvspoof2021",
        root=asv,
        language="other",
        role="external_sanity",
        protocol_file="protocol.txt",
        audio_dir="audio",
    )
    return configs


def test_source_metadata_requires_verified_license() -> None:
    metadata = SourceMetadata(
        source_id="fixture",
        name="Fixture",
        version="v1",
        license=None,
        languages=["kk"],
        intended_role="primary",
    )
    with pytest.raises(ValueError, match="verified license"):
        metadata.require_license()
    with pytest.raises(ValidationError, match="verified or null"):
        metadata.model_copy(update={"license": "TODO"}).model_validate(
            {**metadata.model_dump(), "license": "TODO"}
        )


def test_deterministic_sample_ids() -> None:
    assert deterministic_id("source", "record") == deterministic_id("source", "record")
    assert deterministic_id("source", "record") != deterministic_id("source", "other")


def test_ksc2_real_ingestion_and_conservative_languages(adapter_configs) -> None:
    adapter = adapter_from_config(adapter_configs["ksc"])
    first = adapter.iter_samples()
    second = adapter.iter_samples()
    assert [item.row.sample_id for item in first] == [item.row.sample_id for item in second]
    assert [item.row.language for item in first] == ["kk", "kk_ru"]
    assert all(item.row.label == "real" and item.row.generator_id is None for item in first)
    assert first[0].row.speaker_id == "k1"
    assert first[1].row.missing_metadata == ["speaker_id"]


def test_ksc2_official_sidecar_layout(tmp_path) -> None:
    root = tmp_path / "ISSAI_KSC2"
    tone(root / "Test/crowdsourced/utterance.flac")
    (root / "Test/crowdsourced/utterance.txt").write_text("fixture transcript", encoding="utf-8")
    config = source_config(
        tmp_path,
        "ksc2_sidecar_fixture",
        adapter="ksc2",
        root=root,
        language="kk",
        audio_dir=".",
        metadata_extra={"layout_version": "issai_ksc2_sidecar_v1"},
    )
    rows = [item.row for item in adapter_from_config(config).iter_samples()]
    assert len(rows) == 1
    assert rows[0].original_path == "Test/crowdsourced/utterance.flac"
    assert rows[0].language == "kk" and rows[0].generator_id is None


@pytest.mark.parametrize(("key", "language"), [("fleurs_kk", "kk"), ("fleurs_ru", "ru")])
def test_fleurs_language_ingestion(adapter_configs, key, language) -> None:
    row = adapter_from_config(adapter_configs[key]).iter_samples()[0].row
    assert row.language == language
    assert row.label == "real" and row.generator_id is None
    assert row.speaker_id.startswith("unknown_")
    assert row.missing_metadata == ["speaker_id"]


def test_asvspoof_protocol_mapping_and_role(adapter_configs) -> None:
    adapter = adapter_from_config(adapter_configs["asv"])
    rows = [item.row for item in adapter.iter_samples()]
    assert adapter.config.metadata.intended_role == "external_sanity"
    assert rows[0].label == "real" and rows[0].generator_id is None
    assert rows[1].label == "synthetic"
    assert rows[1].generator_id == rows[1].attack_id == "A07"


def test_verified_source_configs_have_pinned_provenance() -> None:
    root = Path(__file__).parents[1] / "configs/data/sources"
    expected = {
        "ksc2.yaml": ("10.48342/m90y-aj02", None, None),
        "fleurs_kk.yaml": ("10.1109/SLT54892.2023.10023141", "kk_kz", None),
        "fleurs_ru.yaml": ("10.1109/SLT54892.2023.10023141", "ru_ru", None),
        "asvspoof2021.yaml": ("10.5281/zenodo.4835108", None, "DF"),
    }
    for filename, (doi, source_config_name, track) in expected.items():
        metadata = adapter_from_config(root / filename).config.metadata
        assert metadata.license and metadata.citation and metadata.release_revision
        assert metadata.doi == doi
        assert metadata.source_config == source_config_name
        assert metadata.track == track
        assert metadata.layout_version


def test_fixture_build_report_duplicates_and_role_separation(tmp_path, adapter_configs) -> None:
    build = tmp_path / "build.yaml"
    build.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "fixture_real",
                "dataset_version": "v0.1",
                "project_root": ".",
                "output_dir": "built",
                "include_roles": ["primary"],
                "languages": ["kk", "ru", "kk_ru"],
                "duplicate_policy": "report",
                "sample_rate": 16000,
                "max_duration_seconds": 1,
                "max_file_bytes": 100000,
                "sources": [
                    {"config": path.name}
                    for path in [
                        adapter_configs["ksc"],
                        adapter_configs["fleurs_kk"],
                        adapter_configs["fleurs_ru"],
                        adapter_configs["asv"],
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )
    output = prepare(build)
    rows = read_manifest(output / "manifest.parquet")
    report = json.loads((output / "dataset_report.json").read_text(encoding="utf-8"))
    preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))

    assert len(rows) == report["number_of_samples"] == 4
    assert {row.source_dataset for row in rows} == {
        "ksc2_fixture",
        "fleurs_kk_fixture",
        "fleurs_ru_fixture",
    }
    assert report["verified_code_switch_samples"] == 1
    assert report["missing_metadata_counts"] == {
        "codec": 0,
        "duration_sec": 0,
        "sample_rate": 0,
        "source_id": 0,
        "speaker_id": 3,
    }
    assert report["duplicate_content"]["duplicate_sample_count"] == 1
    assert report["invalid_file_count"] == 0
    assert report["manifest_sha256"] == preparation["manifest_hashes"]["manifest.parquet"]
    assert all(row.path.startswith("audio/") for row in rows)
    assert not any(row.source_dataset == "asv_fixture" for row in rows)
