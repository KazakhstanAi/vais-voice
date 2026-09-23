"""Local Piper CLI integration with all executable/model details supplied by config."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class PiperGenerator(GeneratorAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._paths: tuple[str, Path] | None = None

    def _runtime(self) -> tuple[str, Path]:
        if self._paths is not None:
            return self._paths
        executable = str(self.config.runtime.get("executable", "piper"))
        executable_path = Path(executable)
        resolved = str(executable_path) if executable_path.is_file() else shutil.which(executable)
        if not resolved:
            raise RuntimeError(f"Piper executable not found: {executable}")
        model = Path(str(self.config.runtime.get("model_path", ""))).expanduser()
        if not model.is_file():
            raise RuntimeError("Piper model_path is not configured or does not exist")
        config_path = model.with_suffix(model.suffix + ".json")
        if not config_path.is_file():
            raise RuntimeError(f"Piper model config not found: {config_path}")
        expected = {
            Path(resolved): self.config.runtime.get("executable_sha256"),
            model: self.config.runtime.get("model_sha256"),
            config_path: self.config.runtime.get("model_config_sha256"),
        }
        for path, digest in expected.items():
            if digest and sha256(path) != digest:
                raise RuntimeError(f"Pinned Piper artifact checksum mismatch: {path}")
        self._paths = (resolved, model)
        return self._paths

    def _command_prefix(self, executable: str) -> list[str]:
        if self.config.runtime.get("entrypoint") == "python_module":
            return [executable, "-m", str(self.config.runtime.get("module", "piper"))]
        return [executable]

    def validate_runtime(self) -> None:
        executable, _ = self._runtime()
        expected_version = self.config.runtime.get("runtime_version")
        if expected_version:
            if self.config.runtime.get("entrypoint") == "python_module":
                version_command = [
                    executable,
                    "-c",
                    "import importlib.metadata; print(importlib.metadata.version('piper-tts'))",
                ]
            else:
                version_command = [executable, "--version"]
            completed = subprocess.run(
                version_command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
                check=False,
            )
            actual = (completed.stdout or completed.stderr).strip()
            if completed.returncode or actual != expected_version:
                raise RuntimeError(
                    f"Piper runtime version mismatch: expected {expected_version}, got {actual}"
                )

    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        if seed is not None or self.config.supports_seed:
            raise ValueError("Piper adapter does not support deterministic seed input")
        if output_path.exists():
            raise FileExistsError(output_path)
        executable, model = self._runtime()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *self._command_prefix(executable),
            "--model",
            str(model),
            "--output_file",
            str(output_path),
        ]
        if voice_id is not None:
            command += ["--speaker", voice_id]
        input_text: str | None = None
        input_path: Path | None = None
        if self.config.runtime.get("entrypoint") == "python_module":
            cli_parameters = {
                "noise_scale": "--noise-scale",
                "length_scale": "--length-scale",
                "noise_w_scale": "--noise-w-scale",
                "sentence_silence": "--sentence-silence",
                "volume": "--volume",
            }
            for name, flag in cli_parameters.items():
                if name in generation_params:
                    command += [flag, str(generation_params[name])]
            if generation_params.get("normalize_audio") is False:
                command.append("--no-normalize")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                dir=output_path.parent,
                delete=False,
            ) as stream:
                stream.write(text_item.text + "\n")
                input_path = Path(stream.name)
            command += ["--input-file", str(input_path)]
        else:
            cli_parameters = {
                "noise_scale": "--noise_scale",
                "length_scale": "--length_scale",
                "noise_w_scale": "--noise_w",
                "sentence_silence": "--sentence_silence",
            }
            for name, flag in cli_parameters.items():
                if name in generation_params:
                    command += [flag, str(generation_params[name])]
            input_text = text_item.text + "\n"
        try:
            completed = subprocess.run(
                command,
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=600,
                check=False,
            )
        finally:
            if input_path is not None:
                input_path.unlink(missing_ok=True)
        if completed.returncode:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Piper failed: {completed.stderr.strip()}")
        info = sf.info(output_path)
        return GenerationResult(
            job_id="pending",
            status="completed",
            output_relative_path=output_path.name,
            output_sha256=sha256(output_path),
            sample_rate=info.samplerate,
            duration_sec=info.duration,
            codec=output_path.suffix.lstrip("."),
            metadata={
                "runtime": "piper_cli",
                "runtime_version": self.config.runtime.get("runtime_version"),
                "runtime_sha256": self.config.runtime.get("executable_sha256"),
                "model_sha256": self.config.runtime.get("model_sha256"),
                "model_config_sha256": self.config.runtime.get("model_config_sha256"),
            },
        )
