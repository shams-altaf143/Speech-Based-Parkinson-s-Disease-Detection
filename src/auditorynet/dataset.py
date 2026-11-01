"""Dataset discovery, cochleagram extraction and torch Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import load_audio
from .augment import spec_augment
from .config import Config
from .discovery import CLASS_NAMES, TaskFiles, discover_files  # noqa: F401 (re-exported)
from .gammatone import Cochleagram
from .utils import get_logger

log = get_logger()

@torch.no_grad()
def extract_features(paths: List[Path], cfg: Config, device: torch.device,
                     batch: int = 32) -> Tuple[torch.Tensor, List[int]]:
    """Load audio and compute cochleagrams. Returns (N,1,C,T) on CPU and kept indices
    (unreadable files are skipped with a warning)."""
    coch = Cochleagram.from_config(cfg).to(device)
    feats, kept, buf, buf_idx = [], [], [], []

    def flush():
        if buf:
            w = torch.from_numpy(np.stack(buf)).to(device)
            feats.append(coch(w).cpu())
            kept.extend(buf_idx)
            buf.clear()
            buf_idx.clear()

    for i, p in enumerate(paths):
        try:
            buf.append(load_audio(p, cfg.sample_rate, cfg.n_samples))
            buf_idx.append(i)
        except Exception as e:  # corrupt / unsupported file
            log.warning("Skipping unreadable file %s (%s)", p, e)
        if len(buf) == batch:
            flush()
    flush()
    if not feats:
        raise RuntimeError("No audio could be loaded.")
    return torch.cat(feats), kept


class CochleagramDataset(Dataset):
    def __init__(self, feats: torch.Tensor, labels: np.ndarray, train: bool, cfg: Config):
        self.feats, self.labels, self.train, self.cfg = feats, labels, train, cfg

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        x = self.feats[i]
        if self.train:
            c = self.cfg
            x = spec_augment(x, c.aug_prob, c.time_mask_max, c.freq_mask_max,
                             c.noise_prob, c.noise_std, c.shift_prob, c.max_shift)
        return x, int(self.labels[i])
