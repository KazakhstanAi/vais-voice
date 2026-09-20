"""Deterministic audio fixture for offline infrastructure tests only."""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class FakeSineGenerator(GeneratorAdapter):
    def validate_runtime(self) -> None:
        if self.config.verification_status != "infrastructure_test_only":
            raise ValueError("FakeSineGenerator must be marked infrastructure_test_only")

    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        if output_path.exists():
            raise FileExistsError(output_path)
        rate = int(generation_params.get("sample_rate", 16000))
        duration = float(generation_params.get("duration_sec", 0.08))
        identity = f"{text_item.text_id}\0{voice_id}\0{seed}".encode()
        frequency = 220 + int.from_bytes(hashlib.sha256(identity).digest()[:2], "big") % 440
        samples = np.arange(max(1, round(rate * duration)), dtype=np.float64) / rate
        audio = (0.05 * np.sin(2 * np.pi * frequency * samples)).astype(np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, rate, subtype="PCM_16")
        info = sf.info(output_path)
        return GenerationResult(
            job_id="pending",
            status="completed",
            output_relative_path=output_path.name,
            output_sha256=sha256(output_path),
            sample_rate=info.samplerate,
            duration_sec=info.duration,
            codec=output_path.suffix.lstrip("."),
            metadata={"fixture": True, "frequency_hz": frequency},
        )
