from __future__ import annotations

import ipaddress
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .adapters import adapter_for
from .benchmark import ROOT, load_benchmark, load_cases
from .evaluator import aggregate, evaluate_case
from .storage import add_target, create_run, finish_run, initialise, metrics, results, run, target, targets, runs


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialise()
    yield


app = FastAPI(title="VAIS Enterprise", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["https://enterprise.vaislabs.com", "http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


class TargetInput(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    adapter_type: str
    endpoint: str | None = None
    version: str = Field(min_length=1, max_length=80)
    languages: list[str] = Field(min_length=1)
    visibility: str = "private"
    api_key: str | None = Field(default=None, exclude=True)


class RunInput(BaseModel):
    target_id: str
    config: dict = Field(default_factory=dict)


def public_endpoint(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    try:
        return not ipaddress.ip_address(parsed.hostname).is_private and not ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return parsed.hostname.lower() not in {"localhost", "localhost.localdomain"}


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "offline-demo"}


@app.get("/api/benchmarks")
def benchmarks():
    return [load_benchmark()]


@app.get("/api/benchmarks/{identifier}")
def benchmark(identifier: str):
    item = load_benchmark()
    if identifier != item["id"]:
        raise HTTPException(404, "Benchmark not found")
    return item


@app.get("/api/targets")
def get_targets():
    return targets()


@app.post("/api/targets", status_code=201)
def post_target(payload: TargetInput):
    if payload.adapter_type not in {"openai_compatible", "generic_http_json"}:
        raise HTTPException(422, "Only external adapter metadata can be registered here")
    if not public_endpoint(payload.endpoint):
        raise HTTPException(422, "Endpoint must be a public HTTPS address; private and localhost ranges are blocked")
    return add_target(payload.model_dump(exclude={"api_key"}))


@app.get("/api/targets/{identifier}")
def get_target(identifier: str):
    item = target(identifier)
    if not item:
        raise HTTPException(404, "Target not found")
    return item


def execute(payload: RunInput):
    target_item = target(payload.target_id)
    if not target_item:
        raise HTTPException(404, "Target not found")
    if target_item["adapter_type"] != "mock":
        raise HTTPException(501, "External target execution is planned; credential persistence is unsupported")
    item = create_run(target_item, payload.config)
    adapter = adapter_for(target_item)
    results = [evaluate_case(case, adapter.answer(case, [], payload.config)) for case in load_cases()]
    return finish_run(item, results, aggregate(results))


@app.post("/api/runs", status_code=201)
def post_run(payload: RunInput):
    return execute(payload)


@app.post("/api/demo/run", status_code=201)
def demo_run(target_id: str = "mock-a"):
    return execute(RunInput(target_id=target_id))


@app.get("/api/runs")
def get_runs():
    return runs()


@app.get("/api/runs/{identifier}")
def get_run(identifier: str):
    item = run(identifier)
    if not item:
        raise HTTPException(404, "Run not found")
    return {**item, "metrics": metrics(identifier)}


@app.get("/api/runs/{identifier}/results")
def get_results(identifier: str):
    if not run(identifier):
        raise HTTPException(404, "Run not found")
    return results(identifier)


@app.get("/api/compare")
def compare(runs: str):
    identifiers = [part for part in runs.split(",") if part]
    if len(identifiers) != 2:
        raise HTTPException(422, "Supply exactly two comma-separated run IDs")
    selected = [run(identifier) for identifier in identifiers]
    if not all(selected):
        raise HTTPException(404, "Run not found")
    first, second = (results(identifier) for identifier in identifiers)
    return {"runs": [{**selected[index], "metrics": metrics(identifier)} for index, identifier in enumerate(identifiers)], "failure_cases": [item for item, other in zip(first, second) if item["metrics"]["answer_correctness"] != other["metrics"]["answer_correctness"]]}


@app.get("/api/methodology")
def methodology():
    return {"deterministic": ["normalised answer match", "citation coverage", "unsupported claims", "latency"], "human_review": "Not available", "llm_judge": "Not evaluated", "disclaimer": "VAIS Enterprise does not certify that an AI system is safe or production-ready. Benchmark results describe performance under the published evaluation protocol."}


# A built frontend is served by the same localhost-only origin behind the tunnel.
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
