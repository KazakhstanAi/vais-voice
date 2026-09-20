import json
from pathlib import Path

import pytest

from vais_voice.datasets.manifest import read_manifest, write_manifest
from vais_voice.datasets.roles import validate_generator_roles
from vais_voice.datasets.schema import Sample
from vais_voice.datasets.split import assign_splits
from vais_voice.generation.inventory import (
    extract_text_inventory,
    read_inventory,
    write_inventory,
)
from vais_voice.generation.models import GeneratorConfig, GeneratorProvenance
from vais_voice.generation.plan import create_plan, read_jobs
from vais_voice.generation.registry import load_generator_config
from vais_voice.generation.report import build_report
from vais_voice.generation.run import run_plan


def real_row(sample_id: str, language: str, text: str, split: str = "train") -> Sample:
    return Sample(
        sample_id=sample_id,
        path=f"{sample_id}.wav",
        label="real",
        language=language,
        speaker_id=f"speaker_{sample_id}",
        source_dataset=f"source_{language}",
        source_id=sample_id,
        generator_id=None,
        license="fixture",
        original_path=f"{sample_id}.wav",
        rights_reference="fixture",
        usage_permission="approved",
        split=split,
        transcript=text,
        source_record_id=sample_id,
    )


def fake_config(role: str = "train") -> dict:
    return {
        "adapter": "fake_sine",
        "provenance": {
            "generator_id": "infrastructure_fake",
            "display_name": "Fixture only",
            "family": "infrastructure_test_only",
            "provider": "tests",
            "model_id": None,
            "model_revision": "fixture-1",
            "model_version": None,
            "languages": ["kk", "ru"],
            "license": "MIT",
            "license_url": "https://opensource.org/license/mit",
            "homepage_url": None,
            "model_url": None,
            "citation": None,
            "intended_role": role,
            "commercial_use_status": "permissive",
            "notes": "Not speech.",
        },
        "runtime": {},
        "voices": ["fixture"],
        "generation_params": {"duration_sec": 0.03, "sample_rate": 16000},
        "supports_seed": True,
        "license_review_acknowledged": True,
        "verification_status": "infrastructure_test_only",
    }


def pipeline_config(tmp_path: Path, output: str = "plan", role: str = "train") -> Path:
    generator = tmp_path / f"generator_{output}.yaml"
    generator.write_text(json.dumps(fake_config(role)), encoding="utf-8")
    config = tmp_path / f"config_{output}.yaml"
    config.write_text(
        json.dumps(
            {
                "dataset_id": "fixture_synthetic",
                "dataset_version": "v0.1",
                "project_root": ".",
                "text_inventory": "inventory.jsonl",
                "output_dir": output,
                "languages": ["kk", "ru"],
                "max_texts_per_language": 2,
                "allowed_source_datasets": [],
                "allowed_generator_roles": [role],
                "generators": [{"config": generator.name}],
                "seed_strategy": "per_job_hash",
                "base_seed": 42,
                "output_format": "wav",
                "output_sample_rate": None,
                "failure_policy": "stop",
                "existing_output_policy": "skip",
            }
        ),
        encoding="utf-8",
    )
    return config


def make_inventory(tmp_path: Path) -> None:
    manifest = tmp_path / "real.jsonl"
    write_manifest(
        manifest,
        [real_row("kk1", "kk", "Сәлем әлем"), real_row("ru1", "ru", "Привет мир")],
    )
    write_inventory(tmp_path / "inventory.jsonl", extract_text_inventory(manifest))


def synthetic(role: str, split: str = "train") -> Sample:
    return Sample(
        sample_id=f"sample_{role}",
        path="audio.wav",
        label="synthetic",
        language="kk",
        speaker_id="voice",
        source_dataset="fixture",
        source_id=f"source_{role}",
        generator_id=f"generator_{role}",
        license="fixture",
        original_path="audio.wav",
        rights_reference="fixture",
        usage_permission="approved",
        split=split,
        source_text_id="text_1",
        intended_role=role,
    )


def test_generator_provenance_and_license_acknowledgement() -> None:
    provenance = GeneratorProvenance.model_validate(fake_config()["provenance"])
    assert provenance.languages == ["kk", "ru"]
    pending = fake_config()
    pending["adapter"] = "silero"
    pending["verification_status"] = "runtime_unverified"
    pending["provenance"]["license"] = None
    pending["provenance"]["commercial_use_status"] = "needs_review"
    pending["license_review_acknowledged"] = False
    with pytest.raises(ValueError, match="unresolved licensing"):
        GeneratorConfig.model_validate(pending).require_generation_approval()


def test_text_inventory_filtering_deduplication_and_no_code_switch(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        manifest,
        [
            real_row("a", "kk", " Бір  мәтін "),
            real_row("b", "kk", "бір мәтін"),
            real_row("c", "ru", "Текст"),
            real_row("d", "kk_ru", "Аралас"),
        ],
    )
    items = extract_text_inventory(manifest, languages={"kk"})
    assert len(items) == 1
    assert items[0].metadata["source_sample_ids"] == ["a", "b"]
    with pytest.raises(ValueError, match="kk_ru"):
        extract_text_inventory(manifest, languages={"kk_ru"})


def test_text_id_cannot_cross_protected_splits() -> None:
    real = real_row("real1", "kk", "мәтін", "train").model_copy(update={"text_id": "text_shared"})
    generated = synthetic("train", "test").model_copy(update={"source_text_id": "text_shared"})
    with pytest.raises(ValueError, match="text"):
        assign_splits(
            [real, generated],
            seed=1,
            train_fraction=0.6,
            val_fraction=0.2,
            unseen_generators=[],
        )


@pytest.mark.parametrize("role", ["unseen_test", "external_challenge"])
def test_protected_generators_cannot_enter_train(role: str) -> None:
    with pytest.raises(ValueError, match="Generator-role violation"):
        validate_generator_roles([synthetic(role)], "train")


def test_external_challenge_is_not_ordinary_evaluation() -> None:
    with pytest.raises(ValueError, match="Generator-role violation"):
        validate_generator_roles([synthetic("external_challenge", "test")], "protected_test")
    validate_generator_roles([synthetic("external_challenge", "test")], "external_challenge")


@pytest.mark.parametrize(
    ("config_name", "role"),
    [
        ("mms_kk.yaml", "unseen_test"),
        ("mms_ru.yaml", "unseen_test"),
        ("omnivoice.yaml", "unseen_test"),
        ("ait_syn.yaml", "external_challenge"),
    ],
)
def test_configured_protected_generators_are_not_train(config_name: str, role: str) -> None:
    root = Path(__file__).parents[1]
    config = load_generator_config(root / "configs/generation/generators" / config_name)
    assert config.provenance.intended_role == role
    with pytest.raises(ValueError, match="Generator-role violation"):
        validate_generator_roles(
            [synthetic(role).model_copy(update={"generator_id": config.provenance.generator_id})],
            "train",
        )


def test_deterministic_plan_fake_run_resume_manifest_and_report(tmp_path: Path) -> None:
    make_inventory(tmp_path)
    first = create_plan(pipeline_config(tmp_path, "plan_a"))
    second = create_plan(pipeline_config(tmp_path, "plan_b"))
    assert [job.job_id for job in read_jobs(first)] == [job.job_id for job in read_jobs(second)]

    run = run_plan(first)
    rows = read_manifest(run / "manifest.synthetic.jsonl")
    assert len(rows) == 2
    assert all(row.source_text_id and row.generator_family for row in rows)
    assert all(row.generation_seed is not None and row.intended_role == "train" for row in rows)
    original_hashes = {row.sample_id: row.processed_sha256 for row in rows}

    run_plan(first)
    resumed = read_manifest(run / "manifest.synthetic.jsonl")
    assert {row.sample_id: row.processed_sha256 for row in resumed} == original_hashes
    results = [
        json.loads(line) for line in (run / "generation_results.jsonl").read_text().splitlines()
    ]
    assert all(result["status"] == "skipped" for result in results)

    report = json.loads(build_report(run).read_text(encoding="utf-8"))
    assert report["number_of_planned_jobs"] == report["number_skipped"] == 2
    assert report["number_failed"] == 0
    assert report["counts_by"]["language"] == {"kk": 1, "ru": 1}
    assert report["duplicate_hashes"] == {}


def test_existing_output_fail_and_regenerate(tmp_path: Path) -> None:
    make_inventory(tmp_path)
    plan = create_plan(pipeline_config(tmp_path))
    run = run_plan(plan)
    before = {
        row.sample_id: row.processed_sha256
        for row in read_manifest(run / "manifest.synthetic.jsonl")
    }
    with pytest.raises(ValueError, match="policy=fail"):
        run_plan(plan, existing_policy="fail")
    run_plan(plan, existing_policy="regenerate")
    regenerated = read_manifest(run / "manifest.synthetic.jsonl")
    assert len(regenerated) == 2
    assert {row.sample_id: row.processed_sha256 for row in regenerated} == before


def test_duplicate_inventory_records_rejected(tmp_path: Path) -> None:
    make_inventory(tmp_path)
    inventory = tmp_path / "inventory.jsonl"
    line = inventory.read_text(encoding="utf-8").splitlines()[0]
    inventory.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate text_id"):
        read_inventory(inventory)


def test_duplicate_generation_jobs_rejected(tmp_path: Path) -> None:
    make_inventory(tmp_path)
    plan = create_plan(pipeline_config(tmp_path))
    first = plan.read_text(encoding="utf-8").splitlines()[0]
    duplicate = tmp_path / "duplicate_jobs.jsonl"
    duplicate.write_text(f"{first}\n{first}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate jobs"):
        read_jobs(duplicate)
