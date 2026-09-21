"""Execute a generation plan with validated, resumable output handling."""

import json
from pathlib import Path

import soundfile as sf

from vais_voice.datasets.manifest import write_manifest
from vais_voice.datasets.schema import Sample
from vais_voice.generation.inventory import TextItem
from vais_voice.generation.models import GenerationResult, GeneratorConfig
from vais_voice.generation.plan import read_jobs
from vais_voice.generation.registry import adapter_from_config
from vais_voice.utils.io import contained_path, sha256, write_json
from vais_voice.utils.tracking import snapshot


def _read_results(path: Path) -> dict[str, GenerationResult]:
    if not path.exists():
        return {}
    records = [
        GenerationResult.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({record.job_id for record in records}) != len(records):
        raise ValueError("Duplicate generation result records detected")
    return {record.job_id: record for record in records}


def _valid_output(path: Path, result: GenerationResult | None) -> bool:
    if not path.is_file() or result is None or not result.output_sha256:
        return False
    try:
        info = sf.info(path)
    except (RuntimeError, OSError):
        return False
    return info.frames > 0 and sha256(path) == result.output_sha256


def _write_results(path: Path, results: dict[str, GenerationResult]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for job_id in sorted(results):
            stream.write(
                json.dumps(results[job_id].model_dump(mode="json"), ensure_ascii=False) + "\n"
            )
    temporary.replace(path)


def _manifest_rows(
    jobs, results, texts: dict[str, TextItem], configs: dict[str, GeneratorConfig]
) -> list[Sample]:
    rows = []
    for job in jobs:
        result = results.get(job.job_id)
        if not result or result.status not in {"completed", "skipped"} or not result.output_sha256:
            continue
        item = texts[job.text_id]
        config = configs[job.generator_id]
        provenance = config.provenance
        version = provenance.model_revision or provenance.model_version
        rows.append(
            Sample(
                sample_id=f"synth_{job.job_id}",
                path=job.output_relative_path,
                label="synthetic",
                language=job.language,
                speaker_id=job.speaker_name or job.voice_id or f"{job.generator_id}_default",
                source_dataset=item.source_dataset,
                source_id=job.job_id,
                generator_id=job.generator_id,
                license=provenance.license or "unverified; explicit acknowledgement recorded",
                original_path=job.output_relative_path,
                rights_reference=provenance.license_url
                or provenance.model_url
                or provenance.homepage_url
                or "generator config",
                usage_permission="approved",
                codec=result.codec or "unknown",
                split=item.split,
                source_sha256=result.output_sha256,
                processed_sha256=result.output_sha256,
                sample_rate=result.sample_rate,
                duration_sec=result.duration_sec,
                transcript=item.text,
                normalized_text=item.normalized_text,
                text_id=item.text_id,
                source_text_id=item.text_id,
                generator_family=provenance.family,
                generator_version=version,
                generator_model=provenance.generator_model or provenance.model_version,
                generator_runtime=provenance.generator_runtime,
                voice_id=job.voice_id,
                generator_speaker_id=job.speaker_id,
                generator_speaker_name=job.speaker_name,
                generation_seed=job.seed,
                generation_params=job.generation_params,
                intended_role=job.intended_role,
                quality_gate=config.quality_gate,
                training_eligible=config.training_eligible,
                diagnostic_only=config.diagnostic_only,
                candidate=config.candidate,
            )
        )
    return rows


def run_plan(
    plan_path: Path,
    *,
    run_dir: Path | None = None,
    existing_policy: str | None = None,
) -> Path:
    plan_path = plan_path.resolve()
    metadata_path = plan_path.parent / "generation_plan.json"
    if (
        not metadata_path.is_file()
        or sha256(plan_path)
        != json.loads(metadata_path.read_text(encoding="utf-8"))["generation_jobs_sha256"]
    ):
        raise ValueError("Generation plan metadata is missing or plan checksum changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy = existing_policy or metadata["config"]["existing_output_policy"]
    if policy not in {"skip", "fail", "regenerate"}:
        raise ValueError(f"Unknown existing-output policy: {policy}")
    run_dir = (run_dir or plan_path.parent / "run").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = read_jobs(plan_path)
    texts = {item.text_id: item for item in map(TextItem.model_validate, metadata["text_items"])}
    configs = {
        key: GeneratorConfig.model_validate(value)
        for key, value in metadata["generator_configs"].items()
    }
    results_path = run_dir / "generation_results.jsonl"
    results = _read_results(results_path)
    unknown_results = set(results) - {job.job_id for job in jobs}
    if unknown_results:
        raise ValueError(f"Results contain jobs outside this plan: {sorted(unknown_results)}")
    adapters = {}
    stopped_error: RuntimeError | None = None
    for job in jobs:
        config = configs[job.generator_id]
        config.require_generation_approval()
        output = contained_path(run_dir, job.output_relative_path)
        previous = results.get(job.job_id)
        valid = _valid_output(output, previous)
        if output.exists():
            if policy == "fail":
                raise ValueError(f"Output exists for {job.job_id}; policy=fail")
            if policy == "skip" and valid:
                results[job.job_id] = previous.model_copy(update={"status": "skipped"})
                _write_results(results_path, results)
                continue
            if policy == "skip":
                raise ValueError(f"Existing output for {job.job_id} is not valid for this plan")
            output.unlink()
        adapter = adapters.setdefault(job.generator_id, adapter_from_config(config))
        try:
            adapter.validate_runtime()
            runtime_voice = job.voice_id
            if config.adapter == "piper":
                runtime_voice = str(job.speaker_id) if job.speaker_id is not None else None
            generated = adapter.synthesize(
                texts[job.text_id],
                output,
                runtime_voice,
                job.seed,
                job.generation_params,
            )
            completed = generated.model_copy(
                update={"job_id": job.job_id, "output_relative_path": job.output_relative_path}
            )
            if not _valid_output(output, completed):
                raise RuntimeError("Adapter returned output metadata that does not match its audio")
            results[job.job_id] = completed
        except Exception as exc:
            output.unlink(missing_ok=True)
            results[job.job_id] = GenerationResult(
                job_id=job.job_id,
                status="failed",
                output_relative_path=job.output_relative_path,
                error=f"{type(exc).__name__}: {exc}",
            )
            _write_results(results_path, results)
            if metadata["config"]["failure_policy"] == "stop":
                stopped_error = RuntimeError(f"Generation stopped after {job.job_id}: {exc}")
                break
            continue
        _write_results(results_path, results)

    rows = _manifest_rows(jobs, results, texts, configs)
    manifest_hashes = {}
    if rows:
        write_manifest(run_dir / "manifest.synthetic.jsonl", rows, overwrite=True)
        write_manifest(run_dir / "manifest.synthetic.parquet", rows, overwrite=True)
        manifest_hashes = {
            name: sha256(run_dir / name)
            for name in ("manifest.synthetic.jsonl", "manifest.synthetic.parquet")
        }
    run_snapshot = {
        **snapshot(Path(metadata["text_inventory_path"]).parent),
        "status": (
            "incomplete"
            if len(results) != len(jobs)
            else "complete_with_failures"
            if any(result.status == "failed" for result in results.values())
            else "complete"
        ),
        "plan_sha256": sha256(plan_path),
        "plan_path": str(plan_path),
        "text_inventory_sha256": metadata["text_inventory_sha256"],
        "config_sha256": metadata["config_sha256"],
        "generator_config_hashes": metadata["generator_config_hashes"],
        "manifest_hashes": manifest_hashes,
        "dataset_version": metadata["dataset_version"],
        "existing_output_policy": policy,
    }
    report_path = run_dir / "run_snapshot.json"
    if report_path.exists():
        report_path.unlink()
    write_json(report_path, run_snapshot)
    if stopped_error:
        raise stopped_error
    return run_dir
