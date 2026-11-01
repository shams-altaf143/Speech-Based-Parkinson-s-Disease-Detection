"""Gammatone filter bank and cochleagram front-end (Eqs. 2.2-2.4, 3.2-3.4).

The filters are fixed (no learnable parameters) and stored as a buffer, so the
module moves with ``.to(device)`` and is invisible to the optimiser.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def erb(f):
    """Moore-Glasberg ERB in Hz: 24.7 * (4.37 f/1000 + 1)  (Eq. 2.3)."""
    return 24.7 * (4.37 * np.asarray(f, dtype=np.float64) / 1000.0 + 1.0)


def erb_rate(f):
    return 21.4 * np.log10(4.37 * np.asarray(f, dtype=np.float64) / 1000.0 + 1.0)


def centre_frequencies(n_channels: int, f_min: float, f_max: float) -> np.ndarray:
    """Centre frequencies uniformly spaced on the ERB-rate scale (Eq. 2.4)."""
    e = np.linspace(erb_rate(f_min), erb_rate(f_max), n_channels)
    return (10 ** (e / 21.4) - 1.0) / 4.37 * 1000.0


def build_filterbank(
    sample_rate: int = 16000,
    n_channels: int = 64,
    f_min: float = 80.0,
    f_max: float = 8000.0,
    filter_len: int = 400,
) -> np.ndarray:
    """4th-order peak-normalised Gammatone impulse responses, shape (C, L) (Eq. 3.2)."""
    fc = centre_frequencies(n_channels, f_min, f_max)
    b = 1.019 * erb(fc)
    t = np.arange(filter_len) / sample_rate
    g = np.empty((n_channels, filter_len), dtype=np.float64)
    for i in range(n_channels):
        g[i] = (
            (2 * np.pi * b[i]) ** 4 / 6.0
            * t ** 3
            * np.exp(-2 * np.pi * b[i] * t)
            * np.cos(2 * np.pi * fc[i] * t)
        )
        g[i] /= np.max(np.abs(g[i])) + EPS
    return g.astype(np.float32)


class Cochleagram(nn.Module):
    """waveform (B, n) -> cochleagram (B, 1, C, n_frames), min-max scaled to [0, 1]."""

    def __init__(self, sample_rate=16000, n_channels=64, f_min=80.0, f_max=8000.0,
                 filter_len=400, n_frames=128):
        super().__init__()
        self.n_frames = n_frames
        self.filter_len = filter_len
        fb = build_filterbank(sample_rate, n_channels, f_min, f_max, filter_len)
        # conv1d is cross-correlation -> flip for a true convolution (Eq. 3.3).
        self.register_buffer("weight", torch.from_numpy(fb).flip(-1).unsqueeze(1), persistent=False)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        x = wave.unsqueeze(1)  # (B, 1, n)
        c = F.conv1d(F.pad(x, (self.filter_len - 1, 0)), self.weight)  # causal, (B, C, n)
        c = torch.log1p(c.abs())  # compressive non-linearity (Eq. 3.4)
        c = F.adaptive_avg_pool1d(c, self.n_frames)  # (B, C, n_frames)
        lo = c.amin(dim=(1, 2), keepdim=True)
        hi = c.amax(dim=(1, 2), keepdim=True)
        c = (c - lo) / (hi - lo + EPS)  # per-utterance min-max
        return c.unsqueeze(1)

    @classmethod
    def from_config(cls, cfg) -> "Cochleagram":
        return cls(cfg.sample_rate, cfg.n_channels, cfg.f_min, cfg.f_max, cfg.filter_len, cfg.n_frames)
