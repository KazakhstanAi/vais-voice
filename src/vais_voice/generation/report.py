"""Machine-readable reporting over actual generation results and audio."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from vais_voice.generation.models import GenerationResult, GeneratorConfig
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
    dimensions = {
        "language": Counter(),
        "generator": Counter(),
        "generator_family": Counter(),
        "voice_id": Counter(),
        "intended_role": Counter(),
    }
    durations = {key: defaultdict(float) for key in dimensions}
    rates, codecs, errors, hashes = Counter(), Counter(), Counter(), defaultdict(list)
    total_duration = 0.0
    valid_statuses = Counter()
    generated_text_ids = set()
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
        generated_text_ids.add(job.text_id)
        duration = result.duration_sec or 0.0
        total_duration += duration
        values = {
            "language": job.language,
            "generator": job.generator_id,
            "generator_family": configs[job.generator_id].provenance.family,
            "voice_id": job.voice_id or "default",
            "intended_role": job.intended_role,
        }
        for key, value in values.items():
            dimensions[key][value] += 1
            durations[key][value] += duration
        rates[str(result.sample_rate or "unknown")] += 1
        codecs[result.codec or "unknown"] += 1
        if result.output_sha256:
            hashes[result.output_sha256].append(job.job_id)
    report = {
        "dataset_id": plan_metadata["dataset_id"],
        "dataset_version": plan_metadata["dataset_version"],
        "plan_hash": snapshot["plan_sha256"],
        "text_inventory_hash": snapshot["text_inventory_sha256"],
        "generator_configs": plan_metadata["generator_configs"],
        "generator_config_hashes": snapshot["generator_config_hashes"],
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
        "provenance_snapshot": snapshot,
        "source_text_count": len(generated_text_ids),
    }
    path = run_dir / "generation_report.json"
    if path.exists():
        path.unlink()
    write_json(path, report)
    return path
