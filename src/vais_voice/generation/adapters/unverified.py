"""Explicit boundaries for heavyweight runtimes not yet verified in this repository."""

from pathlib import Path
from typing import Any

from vais_voice.generation.base import GeneratorAdapter
from vais_voice.generation.models import GenerationResult, TextItem


class RuntimeUnverifiedGenerator(GeneratorAdapter):
    def validate_runtime(self) -> None:
        name = self.config.provenance.display_name
        raise RuntimeError(f"{name} adapter configured but runtime not installed/verified.")

    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        self.validate_runtime()
        raise AssertionError("unreachable")
