"""Pinned local Silero subprocess integration."""

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class SileroGenerator(GeneratorAdapter):
    def _runtime(self) -> tuple[Path, Path, Path]:
        paths = tuple(
            Path(str(self.config.runtime.get(key, "")))
            for key in ("executable", "script", "model_path")
        )
        if not all(path.is_file() for path in paths):
            raise RuntimeError("Silero executable, runner script, or model is missing")
        for path, key in zip(
            paths, ("executable_sha256", "script_sha256", "model_sha256"), strict=True
        ):
            expected = self.config.runtime.get(key)
            if expected and sha256(path) != expected:
                raise RuntimeError(f"Pinned Silero artifact checksum mismatch: {path}")
        return paths

    def validate_runtime(self) -> None:
        executable, _, _ = self._runtime()
        completed = subprocess.run(
            [str(executable), "-c", "import torch; print(torch.__version__)"],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
            check=False,
        )
        expected = str(self.config.runtime.get("torch_version"))
        if completed.returncode or completed.stdout.strip() != expected:
            actual = completed.stdout.strip()
            raise RuntimeError(f"Silero torch version mismatch: expected {expected}, got {actual}")

    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        if seed is not None:
            raise ValueError("Pinned Silero pilot does not accept a seed")
        executable, script, model = self._runtime()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", dir=output_path.parent, delete=False
        ) as stream:
            stream.write(text_item.text + "\n")
            input_path = Path(stream.name)
        command = [
            str(executable),
            str(script),
            "--model",
            str(model),
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--speaker",
            str(voice_id),
            "--sample-rate",
            str(generation_params["sample_rate"]),
        ]
        try:
            completed = subprocess.run(
                command, text=True, encoding="utf-8", capture_output=True, timeout=600, check=False
            )
        finally:
            input_path.unlink(missing_ok=True)
        if completed.returncode:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Silero failed: {completed.stderr.strip()}")
        info = sf.info(output_path)
        return GenerationResult(
            job_id="pending",
            status="completed",
            output_relative_path=output_path.name,
            output_sha256=sha256(output_path),
            sample_rate=info.samplerate,
            duration_sec=info.duration,
            codec="wav",
            metadata={
                "runtime": "silero_torch_package",
                "torch_version": self.config.runtime.get("torch_version"),
                "model_sha256": self.config.runtime.get("model_sha256"),
            },
        )
