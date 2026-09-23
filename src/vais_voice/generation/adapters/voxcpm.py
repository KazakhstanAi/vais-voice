"""Pinned local VoxCPM subprocess integration."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class VoxCPMGenerator(GeneratorAdapter):
    def _runtime(self) -> dict[str, Path]:
        paths = {
            key: Path(str(self.config.runtime.get(key, "")))
            for key in (
                "executable",
                "script",
                "upstream_path",
                "base_model_path",
                "lora_path",
            )
        }
        if not paths["executable"].is_file() or not paths["script"].is_file():
            raise RuntimeError("VoxCPM executable or runner script is missing")
        model_dirs = ("upstream_path", "base_model_path", "lora_path")
        if not all(paths[key].is_dir() for key in model_dirs):
            raise RuntimeError("Pinned VoxCPM source, base model, or LoRA directory is missing")
        artifacts = {
            paths["executable"]: "executable_sha256",
            paths["script"]: "script_sha256",
            paths["base_model_path"] / "model.safetensors": "base_model_sha256",
            paths["base_model_path"] / "audiovae.pth": "audiovae_sha256",
            paths["lora_path"] / "lora_weights.safetensors": "lora_sha256",
        }
        for path, key in artifacts.items():
            expected = self.config.runtime.get(key)
            if not path.is_file() or not expected or sha256(path) != expected:
                raise RuntimeError(f"Pinned VoxCPM artifact checksum mismatch: {path}")
        return paths

    def validate_runtime(self) -> None:
        executable = self._runtime()["executable"]
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                "import torch; print(torch.__version__); print(torch.cuda.is_available())",
            ],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=30,
            check=False,
        )
        lines = completed.stdout.splitlines()
        expected = str(self.config.runtime.get("torch_version"))
        if completed.returncode or lines != [expected, "True"]:
            raise RuntimeError(f"VoxCPM CUDA runtime mismatch: {completed.stderr or lines}")

    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        if seed is None:
            raise ValueError("Pinned VoxCPM pilot requires a deterministic seed")
        paths = self._runtime()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".txt", dir=output_path.parent, delete=False
        ) as stream:
            stream.write(text_item.text + "\n")
            input_path = Path(stream.name)
        command = [
            str(paths["executable"]),
            str(paths["script"]),
            "--upstream-root",
            str(paths["upstream_path"]),
            "--base-model",
            str(paths["base_model_path"]),
            "--lora",
            str(paths["lora_path"]),
            "--input-file",
            str(input_path),
            "--output-file",
            str(output_path),
            "--seed",
            str(seed),
            "--cfg-value",
            str(generation_params["cfg_value"]),
            "--inference-timesteps",
            str(generation_params["inference_timesteps"]),
            "--max-len",
            str(generation_params["max_len"]),
        ]
        expected_keys = self.config.runtime.get("expected_lora_keys")
        if expected_keys is not None:
            command.extend(("--expected-lora-keys", str(expected_keys)))
        try:
            completed = subprocess.run(command, capture_output=True, timeout=1800, check=False)
        finally:
            input_path.unlink(missing_ok=True)
        if completed.returncode:
            output_path.unlink(missing_ok=True)
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"VoxCPM failed: {stderr}")
        stdout = completed.stdout.decode("utf-8")
        runner_metadata = json.loads(stdout.splitlines()[-1])
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
                "runtime": "voxcpm-kazakh-lora",
                "base_model_sha256": self.config.runtime["base_model_sha256"],
                "lora_sha256": self.config.runtime["lora_sha256"],
                **runner_metadata,
            },
        )
