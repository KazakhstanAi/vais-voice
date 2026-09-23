"""Machine-readable reporting over actual generation results and audio."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from vais_voice.generation.models import (
    GenerationResult,
    GeneratorConfig,
    PronunciationReviewItem,
    TextItem,
)
from vais_voice.generation.plan import read_jobs
from vais_voice.utils.io import contained_path, sha256, write_json


def build_report(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    snapshot = json.loads((run_dir / "run_snapshot.json").read_text(encoding="utf-8"))
    plan_dir = Path(snapshot["plan_path"]).parent
    plan_metadata = json.loads((plan_dir / "generation_plan.json").read_text(encoding="utf-8"))
    jobs_path = plan_dir / "generation_jobs.jsonl"
    if sha256(jobs_path) != snapshot["plan_sha256"]:
        raise ValueError("Generation plan changed after the run")
    jobs = read_jobs(jobs_path)
    results = {
        result.job_id: result
        for result in (
            GenerationResult.model_validate(json.loads(line))
            for line in (run_dir / "generation_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    if len(results) != len(
        [line for line in (run_dir / "generation_results.jsonl").read_text().splitlines() if line]
    ):
        raise ValueError("Duplicate generation results cannot be reported")
    configs = {
        key: GeneratorConfig.model_validate(value)
        for key, value in plan_metadata["generator_configs"].items()
    }
    texts = {
        item.text_id: item for item in map(TextItem.model_validate, plan_metadata["text_items"])
    }
    dimensions = {
        "language": Counter(),
        "generator": Counter(),
        "generator_family": Counter(),
        "voice_id": Counter(),
        "speaker_id": Counter(),
        "speaker_name": Counter(),
        "intended_role": Counter(),
    }
    durations = {key: defaultdict(float) for key in dimensions}
    rates, codecs, errors, hashes = Counter(), Counter(), Counter(), defaultdict(list)
    total_duration = 0.0
    valid_statuses = Counter()
    generated_text_ids = set()
    valid_job_ids = set()
    for job in jobs:
        result = results.get(job.job_id)
        if not result or result.status not in {"completed", "skipped"}:
            if result and result.error:
                errors[result.error] += 1
            continue
        output = contained_path(run_dir, result.output_relative_path)
        if (
            not output.is_file()
            or not result.output_sha256
            or sha256(output) != result.output_sha256
        ):
            errors["invalid_or_missing_output"] += 1
            valid_statuses["failed"] += 1
            continue
        valid_statuses[result.status] += 1
        valid_job_ids.add(job.job_id)
        generated_text_ids.add(job.text_id)
        duration = result.duration_sec or 0.0
        total_duration += duration
        values = {
            "language": job.language,
            "generator": job.generator_id,
            "generator_family": configs[job.generator_id].provenance.family,
            "voice_id": job.voice_id or "default",
            "speaker_id": str(job.speaker_id) if job.speaker_id is not None else "default",
            "speaker_name": job.speaker_name or "default",
            "intended_role": job.intended_role,
        }
        for key, value in values.items():
            dimensions[key][value] += 1
            durations[key][value] += duration
        rates[str(result.sample_rate or "unknown")] += 1
        codecs[result.codec or "unknown"] += 1
        if result.output_sha256:
            hashes[result.output_sha256].append(job.job_id)
    review_path = run_dir / "pronunciation_review.jsonl"
    review_items = [
        PronunciationReviewItem(
            sample_id=f"synth_{job.job_id}",
            language=job.language,
            source_text=texts[job.text_id].text,
            voice=job.voice_id or job.generator_id,
            speaker=job.speaker_name or "single-speaker/default",
            speaker_id=job.speaker_id,
            audio_path=job.output_relative_path,
            training_eligible=configs[job.generator_id].training_eligible,
            diagnostic_only=configs[job.generator_id].diagnostic_only,
            candidate=configs[job.generator_id].candidate,
        )
        for job in jobs
        if job.job_id in valid_job_ids
    ]
    if not review_path.exists():
        with review_path.open("x", encoding="utf-8") as stream:
            for item in review_items:
                stream.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
    else:
        existing_review = [
            PronunciationReviewItem.model_validate(json.loads(line))
            for line in review_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if {item.sample_id for item in existing_review} != {
            item.sample_id for item in review_items
        }:
            raise ValueError("Pronunciation review artifact does not match this run")
        review_items = existing_review

    review_counts = {
        status: sum(item.review_status == status for item in review_items)
        for status in ("pending", "pass", "warning", "fail")
    }
    quality_gate_status = (
        "failed"
        if review_counts["fail"]
        else "pending"
        if review_counts["pending"]
        else "warning"
        if review_counts["warning"]
        else "passed"
    )
    jobs_by_sample_id = {f"synth_{job.job_id}": job for job in jobs}
    reviews_by_generator: dict[str, list[PronunciationReviewItem]] = defaultdict(list)
    for item in review_items:
        reviews_by_generator[jobs_by_sample_id[item.sample_id].generator_id].append(item)
    quality_gate_by_generator = {}
    for generator_id, items in sorted(reviews_by_generator.items()):
        status_counts = {
            status: sum(item.review_status == status for item in items)
            for status in ("pending", "pass", "warning", "fail")
        }
        status = (
            "failed"
            if status_counts["fail"]
            else "pending"
            if status_counts["pending"]
            else "warning"
            if status_counts["warning"]
            else "passed"
        )
        quality_gate_by_generator[generator_id] = {
            "status": status,
            "training_eligible": bool(items) and all(item.training_eligible for item in items),
            "diagnostic_only": bool(items) and all(item.diagnostic_only for item in items),
            "candidate": any(item.candidate for item in items),
            "quality_tier_counts": dict(
                sorted(Counter(item.quality_tier for item in items if item.quality_tier).items())
            ),
            "status_counts": status_counts,
        }
    report = {
        "dataset_id": plan_metadata["dataset_id"],
        "dataset_version": plan_metadata["dataset_version"],
        "plan_hash": snapshot["plan_sha256"],
        "text_inventory_hash": snapshot["text_inventory_sha256"],
        "generator_configs": plan_metadata["generator_configs"],
        "generator_config_hashes": snapshot["generator_config_hashes"],
        "runtime": {key: value.runtime for key, value in sorted(configs.items())},
        "voice_models": {
            key: {
                "voice_id": value.voice_id,
                "model_id": value.provenance.model_id,
                "model_revision": value.provenance.model_revision,
                "model_sha256": value.runtime.get("model_sha256"),
                "base_model_sha256": value.runtime.get("base_model_sha256"),
                "lora_sha256": value.runtime.get("lora_sha256"),
                "speech_tokenizer_sha256": value.runtime.get("speech_tokenizer_sha256"),
                "model_config_sha256": value.runtime.get("model_config_sha256"),
            }
            for key, value in sorted(configs.items())
        },
        "generation_parameters": {
            key: value.generation_params for key, value in sorted(configs.items())
        },
        "number_of_planned_jobs": len(jobs),
        "number_completed": valid_statuses["completed"],
        "number_skipped": valid_statuses["skipped"],
        "number_failed": sum(r.status == "failed" for r in results.values())
        + valid_statuses["failed"],
        "duration_generated_sec": total_duration,
        "counts_by": {key: dict(sorted(value.items())) for key, value in dimensions.items()},
        "duration_by": {key: dict(sorted(value.items())) for key, value in durations.items()},
        "output_sample_rate_distribution": dict(sorted(rates.items())),
        "output_codec_distribution": dict(sorted(codecs.items())),
        "failure_reasons": dict(sorted(errors.items())),
        "duplicate_hashes": {digest: ids for digest, ids in sorted(hashes.items()) if len(ids) > 1},
        "output_hashes": {
            job.job_id: results[job.job_id].output_sha256
            for job in jobs
            if job.job_id in valid_job_ids
        },
        "provenance_snapshot": snapshot,
        "source_text_count": len(generated_text_ids),
        "pronunciation_review": {
            "path": review_path.name,
            "sha256": sha256(review_path),
            "status_counts": review_counts,
        },
        "quality_gate": {
            "status": quality_gate_status,
            "training_eligible": bool(review_items)
            and all(item.training_eligible for item in review_items),
            "diagnostic_only": bool(review_items)
            and all(item.diagnostic_only for item in review_items),
        },
        "quality_gate_by_generator": quality_gate_by_generator,
    }
    path = run_dir / "generation_report.json"
    if path.exists():
        path.unlink()
    write_json(path, report)
    return path
