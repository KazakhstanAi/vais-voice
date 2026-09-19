"""Bounded local decoding and mono resampling, identical for both labels."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SUPPORTED = {".wav", ".flac", ".mp3", ".ogg", ".opus"}


def preprocess(
    source: Path,
    destination: Path,
    *,
    sample_rate: int,
    max_duration_seconds: float,
    max_file_bytes: int,
) -> dict[str, float | int]:
    if source.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported audio extension: {source.suffix}")
    if source.stat().st_size > max_file_bytes:
        raise ValueError("Audio exceeds file-size limit")
    info = sf.info(source)
    if info.frames <= 0 or info.samplerate <= 0 or not 1 <= info.channels <= 8:
        raise ValueError("Empty/invalid audio or unsupported channel count")
    if info.duration > max_duration_seconds:
        raise ValueError("Audio exceeds duration limit")
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("Nonfinite audio samples")
    audio = audio.mean(axis=1)
    if rate != sample_rate:
        divisor = gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor).astype("float32")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Invalid resampled audio")
    # Float WAV preserves amplitude; no class-dependent normalization or silent clipping.
    with destination.open("xb") as stream:
        sf.write(stream, audio, sample_rate, format="WAV", subtype="FLOAT")
    return {"sample_rate": sample_rate, "duration_sec": len(audio) / sample_rate}
