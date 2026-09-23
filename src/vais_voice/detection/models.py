"""Small detector baselines with a shared waveform-to-logit contract."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    sample_rate: int = 16_000
    max_duration_seconds: float = 4.0
    n_mels: int = 64
    hidden_dim: int = 192
    ssl_bundle: str = "WAV2VEC2_BASE"

    @property
    def max_samples(self) -> int:
        return round(self.sample_rate * self.max_duration_seconds)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _hz_to_mel(value: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + value / 700.0)


def _mel_to_hz(value: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, value / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> Tensor:
    frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    low = _hz_to_mel(torch.tensor(20.0))
    high = _hz_to_mel(torch.tensor(sample_rate / 2.0))
    points = _mel_to_hz(torch.linspace(low, high, n_mels + 2))
    lower, center, upper = points[:-2], points[1:-1], points[2:]
    up = (frequencies[None, :] - lower[:, None]) / (center - lower)[:, None]
    down = (upper[:, None] - frequencies[None, :]) / (upper - center)[:, None]
    return torch.clamp(torch.minimum(up, down), min=0.0)


class LogMel(nn.Module):
    def __init__(self, sample_rate: int, n_mels: int) -> None:
        super().__init__()
        self.n_fft = 512
        self.hop_length = 160
        self.win_length = 400
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer(
            "filterbank",
            mel_filterbank(sample_rate, self.n_fft, n_mels),
            persistent=False,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        spectrum = (
            torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
            .abs()
            .square()
        )
        features = torch.matmul(self.filterbank, spectrum)
        features = torch.log(features.clamp_min(1e-6))
        mean = features.mean(dim=(-2, -1), keepdim=True)
        std = features.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (features - mean) / std


class LogMelCnn(nn.Module):
    """Compact baseline A; intentionally simple enough for an 8 GB GPU."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.frontend = LogMel(spec.sample_rate, spec.n_mels)
        channels = (1, 24, 48, 96, 128)
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            groups = math.gcd(output_channels, 8)
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(groups, output_channels),
                    nn.GELU(),
                    nn.MaxPool2d(2, ceil_mode=True),
                ]
            )
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(channels[-1], 1),
        )

    def forward(self, waveform: Tensor) -> Tensor:
        features = self.frontend(waveform).unsqueeze(1)
        return self.classifier(self.encoder(features)).squeeze(-1)


class FrozenWav2VecMlp(nn.Module):
    """Baseline B: official torchaudio SSL encoder, frozen, with a trainable MLP head."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        import torchaudio

        if spec.ssl_bundle != "WAV2VEC2_BASE":
            raise ValueError(f"Unsupported SSL bundle: {spec.ssl_bundle}")
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        if bundle.sample_rate != spec.sample_rate:
            raise ValueError("SSL bundle and detector sample rates differ")
        self.encoder = bundle.get_model()
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, spec.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(spec.hidden_dim, 1),
        )

    def train(self, mode: bool = True) -> FrozenWav2VecMlp:
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, waveform: Tensor) -> Tensor:
        with torch.no_grad():
            features, _ = self.encoder.extract_features(waveform)
            encoded = features[-1]
        pooled = torch.cat([encoded.mean(dim=1), encoded.std(dim=1)], dim=-1)
        return self.classifier(pooled).squeeze(-1)


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.architecture == "logmel_cnn":
        return LogMelCnn(spec)
    if spec.architecture == "wav2vec2_frozen_mlp":
        return FrozenWav2VecMlp(spec)
    raise ValueError(f"Unknown detector architecture: {spec.architecture}")
