"""Manifest-backed waveform dataset used identically by train and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch import Tensor
from torch.utils.data import Dataset

from vais_voice.datasets.schema import Sample
from vais_voice.utils.io import contained_path


def load_waveform(path: Path, sample_rate: int, max_samples: int) -> Tensor:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = audio.mean(axis=1)
    if rate != sample_rate:
        from math import gcd

        divisor = gcd(rate, sample_rate)
        waveform = resample_poly(waveform, sample_rate // divisor, rate // divisor).astype(
            "float32"
        )
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    if len(waveform) > max_samples:
        start = (len(waveform) - max_samples) // 2
        waveform = waveform[start : start + max_samples]
    elif len(waveform) < max_samples:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))
    return torch.from_numpy(np.asarray(waveform, dtype=np.float32))


class ManifestAudioDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        rows: list[Sample],
        audio_root: Path,
        sample_rate: int,
        max_samples: int,
    ) -> None:
        self.rows = rows
        self.audio_root = audio_root
        self.sample_rate = sample_rate
        self.max_samples = max_samples

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        row = self.rows[index]
        waveform = load_waveform(
            contained_path(self.audio_root, row.path), self.sample_rate, self.max_samples
        )
        label = torch.tensor(float(row.label == "synthetic"), dtype=torch.float32)
        return waveform, label, row.sample_id
