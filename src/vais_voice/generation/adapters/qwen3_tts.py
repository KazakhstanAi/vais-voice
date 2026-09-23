"""Pinned local Qwen3-TTS subprocess integration."""

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import soundfile as sf

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem
from vais_voice.utils.io import sha256


class Qwen3TTSGenerator(GeneratorAdapter):
    def __init__(self, config):
        super().__init__(config)
        self._paths: tuple[Path, Path, Path] | None = None
        self._worker: subprocess.Popen[str] | None = None

    def _runtime(self) -> tuple[Path, Path, Path]:
        if self._paths is not None:
            return self._paths
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
        self._paths = paths
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
        self._runtime()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        worker = self._ensure_worker()
        assert worker.stdin is not None and worker.stdout is not None
        effective_text = text_item.text.strip()
        capitalization_added = False
        if generation_params.get("capitalize_initial"):
            for index, character in enumerate(effective_text):
                if character.isalpha():
                    capitalized = character.upper()
                    capitalization_added = capitalized != character
                    effective_text = (
                        effective_text[:index] + capitalized + effective_text[index + 1 :]
                    )
                    break
        punctuation_added = False
        if generation_params.get("append_terminal_punctuation") and not effective_text.endswith(
            (".", "!", "?", "…")
        ):
            effective_text += "."
            punctuation_added = True
        configured_max_tokens = int(generation_params["max_new_tokens"])
        tokens_per_character = float(generation_params.get("max_tokens_per_character", 0))
        token_reserve = int(generation_params.get("max_token_reserve", 0))
        minimum_tokens = int(generation_params.get("min_new_tokens", 1))
        effective_max_tokens = configured_max_tokens
        if tokens_per_character > 0:
            estimated_limit = math.ceil(len(effective_text) * tokens_per_character) + token_reserve
            effective_max_tokens = min(configured_max_tokens, max(minimum_tokens, estimated_limit))
        request = {
            "text": effective_text,
            "output_file": str(output_path.resolve()),
            "speaker": voice_id,
            "language": str(generation_params["language"]),
            "instruct": str(generation_params["instruct"]),
            "top_k": generation_params["top_k"],
            "top_p": generation_params["top_p"],
            "temperature": generation_params["temperature"],
            "repetition_penalty": generation_params["repetition_penalty"],
            "max_new_tokens": effective_max_tokens,
        }
        nominal_token_rate = float(generation_params.get("nominal_audio_token_rate_hz", 12))
        runaway_threshold = float(generation_params.get("runaway_limit_fraction", 0.9))
        maximum_expected_duration = effective_max_tokens / nominal_token_rate
        retry_attempts = int(generation_params.get("runaway_retry_attempts", 1))
        retry_stride = int(generation_params.get("retry_seed_stride", 1))
        if retry_attempts < 1:
            raise ValueError("runaway_retry_attempts must be at least one")
        runner_metadata: dict[str, Any] | None = None
        info = None
        runaway_durations: list[float] = []
        for attempt in range(retry_attempts):
            effective_seed = (seed + attempt * retry_stride) % (2**64)
            request["seed"] = effective_seed
            worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            worker.stdin.flush()
            response = self._read_protocol_message(worker, {"completed", "failed"})
            if response.get("status") != "completed":
                output_path.unlink(missing_ok=True)
                raise RuntimeError(f"Qwen3-TTS failed: {response.get('error', 'unknown error')}")
            info = sf.info(output_path)
            if info.duration < maximum_expected_duration * runaway_threshold:
                runner_metadata = response["metadata"]
                runner_metadata["effective_seed"] = effective_seed
                runner_metadata["generation_attempt"] = attempt + 1
                break
            runaway_durations.append(info.duration)
            output_path.unlink(missing_ok=True)
        if runner_metadata is None or info is None:
            formatted = ", ".join(f"{duration:.2f}s" for duration in runaway_durations)
            raise RuntimeError(
                "Qwen3-TTS exhausted the deterministic runaway retries "
                f"({formatted}; max_new_tokens={effective_max_tokens})"
            )
        runner_metadata["initial_capitalization_added"] = capitalization_added
        runner_metadata["terminal_punctuation_added"] = punctuation_added
        runner_metadata["rejected_runaway_durations_sec"] = runaway_durations
        runner_metadata["effective_max_new_tokens"] = effective_max_tokens
        runner_metadata["runaway_duration_limit_sec"] = (
            maximum_expected_duration * runaway_threshold
        )
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

    def _read_protocol_message(
        self, worker: subprocess.Popen[str], expected_statuses: set[str]
    ) -> dict[str, Any]:
        """Ignore third-party stdout noise and return the next worker protocol record."""
        assert worker.stdout is not None
        while True:
            line = worker.stdout.readline()
            if not line:
                return_code = worker.poll()
                self._worker = None
                raise RuntimeError(f"Qwen3-TTS worker stopped unexpectedly (exit={return_code})")
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("status") in expected_statuses:
                return message

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        executable, script, model = self._runtime()
        self._worker = subprocess.Popen(
            [str(executable), str(script), "--worker", "--model", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._read_protocol_message(self._worker, {"ready"})
        return self._worker

    def close(self) -> None:
        worker, self._worker = self._worker, None
        if worker is None or worker.poll() is not None:
            return
        try:
            if worker.stdin is not None:
                worker.stdin.write('{"command":"close"}\n')
                worker.stdin.flush()
            worker.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
