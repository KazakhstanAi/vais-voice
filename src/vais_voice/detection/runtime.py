"""Checkpoint-backed detector runtime shared by CLI and FastAPI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import torch

from vais_voice.detection.dataset import load_waveform
from vais_voice.detection.models import ModelSpec
from vais_voice.detection.training import load_checkpoint
from vais_voice.utils.io import sha256


class CheckpointDetector:
    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        self.checkpoint = checkpoint.resolve()
        if not self.checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {self.checkpoint}")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        self.model, self.payload = load_checkpoint(self.checkpoint, self.device)
        self.spec = ModelSpec(**self.payload["model_spec"])
        self.lock = threading.Lock()

    def predict_score(self, audio_path: Path) -> float:
        waveform = (
            load_waveform(audio_path, self.spec.sample_rate, self.spec.max_samples)
            .unsqueeze(0)
            .to(self.device)
        )
        with self.lock, torch.inference_mode():
            score = torch.sigmoid(self.model(waveform))[0]
        return float(score.cpu())

    @property
    def threshold(self) -> float:
        return float(self.payload["threshold"])

    def metadata(self) -> dict[str, Any]:
        model_sidecar = self.checkpoint.with_name("model.json")
        sidecar = (
            json.loads(model_sidecar.read_text(encoding="utf-8")) if model_sidecar.is_file() else {}
        )
        return {
            **sidecar,
            "model_id": self.payload["model_id"],
            "architecture": self.spec.architecture,
            "threshold": self.threshold,
            "checkpoint_sha256": sha256(self.checkpoint),
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
        }
