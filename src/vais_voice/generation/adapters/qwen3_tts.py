"""Pinned local Qwen3-TTS subprocess integration."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class Qwen3TTSGenerator(GeneratorAdapter):
    def _runtime(self) -> tuple[Path, Path, Path]:
        paths = tuple(
            Path(str(self.config.runtime.get(key, "")))
            for key in ("executable", "script", "model_path")
        )
        if not paths[0].is_file() or not paths[1].is_file() or not paths[2].is_dir():
            raise RuntimeError("Qwen3-TTS executable, runner, or model directory is missing")
        artifacts = {
            paths[0]: "executable_sha256",
            paths[1]: "script_sha256",
            paths[2] / "model.safetensors": "model_sha256",
            paths[2] / "speech_tokenizer" / "model.safetensors": "speech_tokenizer_sha256",
        }
        for path, key in artifacts.items():
            expected = self.config.runtime.get(key)
            if not path.is_file() or not expected or sha256(path) != expected:
                raise RuntimeError(f"Pinned Qwen3-TTS artifact checksum mismatch: {path}")
        return paths

    def validate_runtime(self) -> None:
        executable, _, _ = self._runtime()
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                "import torch,qwen_tts; print(torch.__version__); print(torch.cuda.is_available())",
            ],
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=60,
            check=False,
        )
        lines = completed.stdout.splitlines()
        expected = str(self.config.runtime.get("torch_version"))
        if completed.returncode or lines[-2:] != [expected, "True"]:
            raise RuntimeError(f"Qwen3-TTS CUDA runtime mismatch: {completed.stderr or lines}")

    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        if seed is None or voice_id is None:
            raise ValueError("Pinned Qwen3-TTS pilot requires a seed and speaker")
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
            voice_id,
            "--language",
            str(generation_params["language"]),
            "--instruct",
            str(generation_params["instruct"]),
            "--seed",
            str(seed),
            "--top-k",
            str(generation_params["top_k"]),
            "--top-p",
            str(generation_params["top_p"]),
            "--temperature",
            str(generation_params["temperature"]),
            "--repetition-penalty",
            str(generation_params["repetition_penalty"]),
            "--max-new-tokens",
            str(generation_params["max_new_tokens"]),
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, timeout=1800, check=False
            )
        finally:
            input_path.unlink(missing_ok=True)
        if completed.returncode:
            output_path.unlink(missing_ok=True)
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Qwen3-TTS failed: {stderr}")
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
                "runtime": "qwen-tts-package",
                "model_sha256": self.config.runtime["model_sha256"],
                "speech_tokenizer_sha256": self.config.runtime["speech_tokenizer_sha256"],
                **runner_metadata,
            },
        )
