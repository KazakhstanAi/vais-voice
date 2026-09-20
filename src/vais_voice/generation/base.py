"""Generator adapter contract; importing it never loads a model."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from vais_voice.generation.models import GenerationResult, GeneratorConfig, TextItem


class GeneratorAdapter(ABC):
    def __init__(self, config: GeneratorConfig):
        self.config = config

    @property
    def supported_languages(self) -> frozenset[str]:
        return frozenset(self.config.provenance.languages)

    @property
    def available_voices(self) -> tuple[str, ...]:
        return tuple(self.config.voices)

    @abstractmethod
    def validate_runtime(self) -> None:
        """Fail before synthesis when the configured runtime cannot be used."""

    @abstractmethod
    def synthesize(
        self,
        text_item: TextItem,
        output_path: Path,
        voice_id: str | None,
        seed: int | None,
        generation_params: dict[str, Any],
    ) -> GenerationResult:
        """Synthesize one item without silently replacing an existing output."""
