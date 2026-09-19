from pathlib import Path
from typing import Protocol


class Detector(Protocol):
    def predict_score(self, audio_path: Path) -> float: ...


class DummyDetector:
    """Constant score; infrastructure testing only, never an authenticity detector."""

    def predict_score(self, audio_path: Path) -> float:
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        return 0.5
