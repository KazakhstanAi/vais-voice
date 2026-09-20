"""Local Piper CLI integration with all executable/model details supplied by config."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class PiperGenerator(GeneratorAdapter):
    def _runtime(self) -> tuple[str, Path]:
        executable = str(self.config.runtime.get("executable", "piper"))
        resolved = shutil.which(executable)
        if not resolved:
            raise RuntimeError(f"Piper executable not found: {executable}")
        model = Path(str(self.config.runtime.get("model_path", ""))).expanduser()
        if not model.is_file():
            raise RuntimeError("Piper model_path is not configured or does not exist")
        return resolved, model

    def validate_runtime(self) -> None:
        self._runtime()

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
        command = [executable, "--model", str(model), "--output_file", str(output_path)]
        if voice_id is not None:
            command += ["--speaker", voice_id]
        if generation_params:
            command += ["--json-input"]
            input_text = json.dumps({"text": text_item.text, **generation_params}) + "\n"
        else:
            input_text = text_item.text + "\n"
        completed = subprocess.run(
            command, input=input_text, text=True, capture_output=True, timeout=600, check=False
        )
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
            metadata={"runtime": "piper_cli"},
        )
