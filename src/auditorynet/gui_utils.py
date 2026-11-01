"""Torch-free helpers used by the GUI (unit-testable)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from .config import Config
from .discovery import discover_files

MIN_PER_CLASS = 10  # same threshold train.run_task uses to skip a task


def scan_dataset(data_dir: str, tasks: List[str]) -> List[Dict]:
    rows = []
    for t in tasks:
        tf = discover_files(data_dir, t, quiet=True)
        if tf is None:
            rows.append({"Task": t, "PD files": 0, "Healthy files": 0, "Status": "Missing / no WAV files"})
            continue
        pd_n, hc_n = int((tf.labels == 1).sum()), int((tf.labels == 0).sum())
        ok = min(pd_n, hc_n) >= MIN_PER_CLASS
        status = "OK" if ok else f"Too few files (need >= {MIN_PER_CLASS} per class)"
        rows.append({"Task": t, "PD files": pd_n, "Healthy files": hc_n, "Status": status})
    return rows


def validate_regex(pattern: str) -> Optional[str]:
    """Return an error message, or None if empty/valid (must contain a capture group)."""
    if not pattern.strip():
        return None
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Invalid regex: {e}"
    if rx.groups < 1:
        return "Regex needs one capture group, e.g. ^(spk\\d+)_"
    return None


def write_config(settings: Dict, path: Path) -> Config:
    cfg = Config()
    valid = set(cfg.to_dict())
    for k, v in settings.items():
        if k not in valid:
            raise KeyError(f"Unknown setting: {k}")
        setattr(cfg, k, v)
    if not cfg.speaker_regex:
        cfg.speaker_regex = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return cfg


def tail_file(path: Optional[Path], n_chars: int = 6000) -> str:
    if not path or not Path(path).exists():
        return ""
    data = Path(path).read_bytes()[-n_chars:]
    return data.decode("utf-8", errors="replace")


_FOLD = re.compile(r"Task (\S+) \| fold (\d+)/(\d+)")
_EP = re.compile(r"ep (\d+)/(\d+)")


def parse_progress(log: str, tasks: List[str], n_folds: int, epochs: int):
    """Approximate overall progress (0..1) and a label from the training log."""
    if "Done in" in log:
        return 1.0, "Finished"
    last = None
    for last in _FOLD.finditer(log):
        pass
    if last is None:
        return 0.0, "Preparing data / extracting cochleagrams..."
    task, fold, nf = last.group(1), int(last.group(2)), int(last.group(3))
    ep, total_ep = 0, epochs
    for m in _EP.finditer(log[last.end():]):
        ep, total_ep = int(m.group(1)), int(m.group(2))
    ti = tasks.index(task) if task in tasks else 0
    frac = ((ti * nf + (fold - 1)) * total_ep + ep) / (max(1, len(tasks)) * nf * total_ep)
    return float(min(frac, 0.99)), f"Task {task} - fold {fold}/{nf} - epoch {ep}/{total_ep}"


_CKPT = re.compile(r"^(.+)_fold(\d+)\.pt$")


def available_checkpoint_tasks(ckpt_dir: Path) -> List[str]:
    if not Path(ckpt_dir).is_dir():
        return []
    return sorted({m.group(1) for f in Path(ckpt_dir).glob("*.pt") if (m := _CKPT.match(f.name))})


def read_summary(output_dir: Path):
    import pandas as pd

    p = Path(output_dir) / "summary.csv"
    return pd.read_csv(p) if p.exists() else None
