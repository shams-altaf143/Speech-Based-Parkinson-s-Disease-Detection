from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class Config:
    data_dir: str = "data"
    output_dir: str = "outputs"
    tasks: List[str] = field(default_factory=lambda: ["A", "E", "I", "O", "U", "petaka", "apto"])

    sample_rate: int = 16000
    duration: float = 3.0
    n_channels: int = 64
    f_min: float = 80.0
    f_max: float = 8000.0
    filter_ms: float = 25.0
    n_frames: int = 128

    batch_size: int = 16
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.4
    grad_clip: float = 1.0
    n_folds: int = 5
    seed: int = 42
    aug_prob: float = 0.5
    noise_prob: float = 0.3
    noise_std: float = 0.02
    shift_prob: float = 0.3
    max_shift: int = 8
    label_smoothing: float = 0.05
    val_fraction: float = 0.15
    patience: int = 8
    time_mask_max: int = 10
    freq_mask_max: int = 8

    device: str = "auto"
    num_workers: int = 0
    amp: bool = False
    speaker_regex: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: "str | Path | None") -> "Config":
        if path is None:
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.duration))

    @property
    def filter_len(self) -> int:
        return int(self.sample_rate * self.filter_ms / 1000)
