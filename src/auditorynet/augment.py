"""Cochleagram augmentation: SpecAugment masks (Eqs. 3.5-3.6) plus optional noise / time shift."""
from __future__ import annotations

import torch


def spec_augment(c: torch.Tensor, p: float = 0.5, max_t: int = 10, max_f: int = 8,
                 noise_p: float = 0.0, noise_std: float = 0.02,
                 shift_p: float = 0.0, max_shift: int = 8) -> torch.Tensor:
    """c: (1, F, T). Returns an augmented copy. Uses torch RNG (DataLoader-worker safe)."""
    c = c.clone()
    _, n_f, n_t = c.shape
    if shift_p > 0 and torch.rand(1).item() < shift_p:
        s = int(torch.randint(-max_shift, max_shift + 1, (1,)))
        c = torch.roll(c, shifts=s, dims=-1)
    if noise_p > 0 and torch.rand(1).item() < noise_p:
        c = (c + noise_std * torch.randn_like(c)).clamp_(0.0, 1.0)
    if torch.rand(1).item() < p:
        m = int(torch.randint(1, max_t + 1, (1,)))
        t0 = int(torch.randint(0, max(1, n_t - max_t), (1,)))
        c[:, :, t0 : t0 + m] = 0.0
    if torch.rand(1).item() < p:
        m = int(torch.randint(1, max_f + 1, (1,)))
        f0 = int(torch.randint(0, max(1, n_f - max_f), (1,)))
        c[:, f0 : f0 + m, :] = 0.0
    return c
