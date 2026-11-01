"""Audio loading and normalisation (Eq. 3.1)."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EPS = 1e-8


def fix_length(wave: np.ndarray, n_samples: int) -> np.ndarray:
    """Zero-pad or truncate to exactly n_samples."""
    if len(wave) >= n_samples:
        return wave[:n_samples]
    return np.pad(wave, (0, n_samples - len(wave)))


def normalise(wave: np.ndarray) -> np.ndarray:
    """Peak amplitude normalisation: w / (max|w| + eps)."""
    return wave / (np.max(np.abs(wave)) + EPS)


def load_raw(path: "str | Path", sample_rate: int) -> np.ndarray:
    """Load a WAV as mono float32 at ``sample_rate`` (no length fixing)."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)  # multichannel -> mono
    if sr != sample_rate:
        g = gcd(int(sr), int(sample_rate))
        wave = resample_poly(wave, sample_rate // g, sr // g).astype(np.float32)
    return wave


def load_audio(path: "str | Path", sample_rate: int, n_samples: int) -> np.ndarray:
    """Load -> mono -> resample -> pad/truncate -> peak-normalise."""
    wave = fix_length(load_raw(path, sample_rate), n_samples)
    return normalise(wave).astype(np.float32)
