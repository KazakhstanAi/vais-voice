from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .benchmark import ROOT, dataset_hash, load_benchmark


DB_PATH = ROOT / "data" / "enterprise.sqlite3"
ARTIFACTS = ROOT / "artifacts" / "runs"


def now() -> str:
    return datetime.now(UTC).isoformat()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialise() -> None:
    with connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS evaluation_targets (id TEXT PRIMARY KEY, name TEXT NOT NULL, adapter_type TEXT NOT NULL, endpoint TEXT, version TEXT NOT NULL, languages_json TEXT NOT NULL, visibility TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS benchmark_runs (id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, benchmark_version TEXT NOT NULL, target_id TEXT NOT NULL, target_version TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT, config_hash TEXT NOT NULL, dataset_hash TEXT NOT NULL, result_hash TEXT, evaluator_version TEXT NOT NULL, git_commit TEXT NOT NULL, visibility TEXT NOT NULL, limitations_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS case_results (run_id TEXT NOT NULL, case_id TEXT NOT NULL, result_json TEXT NOT NULL, PRIMARY KEY (run_id, case_id));
        CREATE TABLE IF NOT EXISTS metric_results (run_id TEXT NOT NULL, metric_name TEXT NOT NULL, value REAL, metric_source TEXT NOT NULL, metadata_json TEXT NOT NULL, PRIMARY KEY (run_id, metric_name));
        """)
        for ident, name, version in [("mock-a", "MockTarget A", "mock-a"), ("mock-b", "MockTarget B", "mock-b")]:
            conn.execute("INSERT OR IGNORE INTO evaluation_targets VALUES (?, ?, 'mock', NULL, ?, '[\"kk\", \"ru\"]', 'public', ?)", (ident, name, version, now()))


def targets() -> list[dict[str, Any]]:
    with connection() as conn:
        return [{**dict(row), "languages": json.loads(row["languages_json"])} for row in conn.execute("SELECT * FROM evaluation_targets ORDER BY created_at")]


def target(identifier: str) -> dict[str, Any] | None:
    return next((item for item in targets() if item["id"] == identifier), None)


def add_target(payload: dict[str, Any]) -> dict[str, Any]:
    item = {"id": f"target-{uuid.uuid4().hex[:12]}", **payload, "created_at": now()}
    with connection() as conn:
        conn.execute("INSERT INTO evaluation_targets VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (item["id"], item["name"], item["adapter_type"], item.get("endpoint"), item["version"], json.dumps(item["languages"]), item["visibility"], item["created_at"]))
    return item


def create_run(target_item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    benchmark = load_benchmark()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    item = {"id": str(uuid.uuid4()), "benchmark_id": benchmark["id"], "benchmark_version": benchmark["version"], "target_id": target_item["id"], "target_version": target_item["version"], "status": "running", "started_at": now(), "finished_at": None, "config_hash": config_hash, "dataset_hash": dataset_hash(), "result_hash": None, "evaluator_version": "0.1.0", "git_commit": git_commit(), "visibility": "private", "limitations_json": json.dumps(["Synthetic public demo dataset", "Retrieval metrics are not evaluated in v0.1"])}
    with connection() as conn:
        conn.execute("INSERT INTO benchmark_runs VALUES (:id,:benchmark_id,:benchmark_version,:target_id,:target_version,:status,:started_at,:finished_at,:config_hash,:dataset_hash,:result_hash,:evaluator_version,:git_commit,:visibility,:limitations_json)", item)
    return item


def finish_run(run: dict[str, Any], results: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    result_bytes = "\n".join(json.dumps(result, ensure_ascii=False, sort_keys=True) for result in results).encode()
    result_hash = hashlib.sha256(result_bytes).hexdigest()
    finished_at = now()
    with connection() as conn:
        conn.execute("UPDATE benchmark_runs SET status='completed', finished_at=?, result_hash=? WHERE id=? AND status='running'", (finished_at, result_hash, run["id"]))
        for result in results:
            conn.execute("INSERT INTO case_results VALUES (?, ?, ?)", (run["id"], result["case_id"], json.dumps(result, ensure_ascii=False)))
        for name, metric in metrics.items():
            conn.execute("INSERT INTO metric_results VALUES (?, ?, ?, ?, ?)", (run["id"], name, metric["value"], metric["metric_source"], "{}"))
    run.update(status="completed", finished_at=finished_at, result_hash=result_hash)
    write_artifacts(run, results, metrics)
    return run


def write_artifacts(run: dict[str, Any], results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    path = ARTIFACTS / run["id"]
    path.mkdir(parents=True, exist_ok=True)
    files = {"run.json": run, "metrics.json": metrics, "manifest.json": {"benchmark_version": run["benchmark_version"], "dataset_hash": run["dataset_hash"], "result_hash": run["result_hash"], "evaluator_version": run["evaluator_version"], "timestamp": run["finished_at"]}}
    for name, value in files.items():
        (path / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    (path / "results.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in results) + "\n", encoding="utf-8")
    (path / "sha256.txt").write_text("\n".join(f"{hashlib.sha256((path / name).read_bytes()).hexdigest()}  {name}" for name in [*files, "results.jsonl"]) + "\n", encoding="utf-8")


def runs() -> list[dict[str, Any]]:
    with connection() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM benchmark_runs ORDER BY started_at DESC")]


def run(identifier: str) -> dict[str, Any] | None:
    return next((item for item in runs() if item["id"] == identifier), None)


def results(identifier: str) -> list[dict[str, Any]]:
    with connection() as conn:
        return [json.loads(row["result_json"]) for row in conn.execute("SELECT result_json FROM case_results WHERE run_id=? ORDER BY case_id", (identifier,))]


def metrics(identifier: str) -> dict[str, Any]:
    with connection() as conn:
        return {row["metric_name"]: {"value": row["value"], "metric_source": row["metric_source"]} for row in conn.execute("SELECT * FROM metric_results WHERE run_id=?", (identifier,))}
