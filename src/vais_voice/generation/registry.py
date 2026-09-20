"""Config-driven generator registry."""

from pathlib import Path

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GeneratorConfig
from vais_voice.utils.io import load_yaml


def load_generator_config(path: Path) -> GeneratorConfig:
    config = GeneratorConfig.model_validate(load_yaml(path))
    runtime = dict(config.runtime)
    for key in ("executable", "model_path"):
        value = str(runtime.get(key, ""))
        candidate = Path(value)
        if value and not candidate.is_absolute() and ("/" in value or "\\" in value):
            runtime[key] = str((path.parent / candidate).resolve())
    return config.model_copy(update={"runtime": runtime})


def adapter_from_config(config: GeneratorConfig) -> GeneratorAdapter:
    if config.adapter == "fake_sine":
        from vais_voice.generation.adapters.fake import FakeSineGenerator

        return FakeSineGenerator(config)
    if config.adapter == "piper":
        from vais_voice.generation.adapters.piper import PiperGenerator

        return PiperGenerator(config)
    if config.adapter in {"silero", "voxcpm", "qwen3_tts", "mms", "omnivoice", "ait_syn"}:
        from vais_voice.generation.adapters.unverified import RuntimeUnverifiedGenerator

        return RuntimeUnverifiedGenerator(config)
    raise ValueError(f"Unknown generator adapter: {config.adapter}")
