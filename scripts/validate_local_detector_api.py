"""Run a secret-free controlled smoke against the local FastAPI detector boundary."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--long-audio", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cuda")
    args = parser.parse_args()

    voice_repository = Path(__file__).resolve().parents[1]
    service = args.web_repository.resolve() / "services" / "ml-api"
    os.environ.update(
        {
            "VAIS_ML_MOCK_MODE": "false",
            "VAIS_ML_VOICE_REPO": str(voice_repository),
            "VAIS_ML_DETECTOR_CHECKPOINT": str(args.checkpoint.resolve()),
            "VAIS_ML_DETECTOR_DEVICE": args.device,
            "VAIS_ML_API_TOKEN": "",
        }
    )
    sys.path.insert(0, str(service))
    module = importlib.import_module("app.main")
    client = TestClient(module.app)

    started = time.time()
    health = client.get("/health").json()
    model = client.get("/model").json()
    payload = args.audio.read_bytes()
    latencies: list[float] = []
    responses: list[dict[str, object]] = []
    for _ in range(10):
        before = time.perf_counter()
        response = client.post(
            "/detect", files={"audio": (args.audio.name, payload, "audio/wav")}
        )
        latencies.append((time.perf_counter() - before) * 1000)
        responses.append({"status": response.status_code, "body": response.json()})

    negative = {}
    for name, filename, data in (
        ("empty", "empty.wav", b""),
        ("invalid_extension", "sample.exe", payload),
        ("malformed", "malformed.wav", b"not a wave file"),
    ):
        response = client.post("/detect", files={"audio": (filename, data)})
        negative[name] = {"status": response.status_code, "detail": response.json().get("detail")}

    oversized = b"0" * (26_214_400 + 1)
    response = client.post("/detect", files={"audio": ("oversized.wav", oversized)})
    negative["oversized"] = {
        "status": response.status_code,
        "detail": response.json().get("detail"),
    }

    short_buffer = io.BytesIO()
    with wave.open(short_buffer, "wb") as short_wav:
        short_wav.setnchannels(1)
        short_wav.setsampwidth(2)
        short_wav.setframerate(16_000)
        short_wav.writeframes(b"\0\0" * 1600)
    response = client.post(
        "/detect", files={"audio": ("short.wav", short_buffer.getvalue(), "audio/wav")}
    )
    duration_cases = {"short": {"status": response.status_code, "body": response.json()}}
    if args.long_audio:
        response = client.post(
            "/detect",
            files={
                "audio": (
                    args.long_audio.name,
                    args.long_audio.read_bytes(),
                    "audio/wav",
                )
            },
        )
        duration_cases["long"] = {"status": response.status_code, "body": response.json()}

    memory = None
    try:
        import torch

        if torch.cuda.is_available():
            memory = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
    except ImportError:
        pass
    print(
        json.dumps(
            {
                "started_unix": started,
                "health": health,
                "model": model,
                "requests": {
                    "count": len(responses),
                    "success": sum(item["status"] == 200 for item in responses),
                    "latency_ms": {
                        "min": min(latencies),
                        "mean": sum(latencies) / len(latencies),
                        "max": max(latencies),
                    },
                    "unique_model_ids": sorted(
                        {str(item["body"].get("model_id")) for item in responses}
                    ),
                    "unique_checkpoint_hashes": sorted(
                        {str(item["body"].get("checkpoint_sha256")) for item in responses}
                    ),
                },
                "negative": negative,
                "duration_cases": duration_cases,
                "backend_timeout": "not_evaluated_in_process_test_client",
                "gpu_memory": memory,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
