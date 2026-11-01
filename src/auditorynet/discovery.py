"""Dataset discovery (no torch dependency, so the GUI can scan data instantly)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger("auditorynet")

CLASS_NAMES = ("Healthy", "PD")  # label 0 = HC, label 1 = PD
_PD_DIRS = {"pd", "parkinson", "parkinsons"}
_HC_DIRS = {"healthy", "hc", "control", "controls"}


@dataclass
class TaskFiles:
    paths: List[Path]
    labels: np.ndarray
    groups: List[str]


def _find_task_dir(root: Path, task: str) -> Optional[Path]:
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.is_dir() and d.name.lower() == task.lower():
            return d
    return None


def discover_files(data_dir: str, task: str, speaker_regex: Optional[str] = None,
                   quiet: bool = False) -> Optional[TaskFiles]:
    """Collect WAVs recursively from <data_dir>/<task>/{PD,Healthy}/ (case-insensitive)."""
    warn = (lambda *a: None) if quiet else log.warning
    task_dir = _find_task_dir(Path(data_dir), task)
    if task_dir is None:
        warn("Task folder '%s' not found under %s - skipping.", task, data_dir)
        return None
    rx = re.compile(speaker_regex) if speaker_regex else None
    paths, labels, groups = [], [], []
    for sub in sorted(task_dir.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        label = 1 if name in _PD_DIRS else 0 if name in _HC_DIRS else None
        if label is None:
            continue
        for f in sorted(sub.rglob("*")):
            if f.is_file() and f.suffix.lower() == ".wav":
                paths.append(f)
                labels.append(label)
                if rx:
                    m = rx.search(f.stem)
                    groups.append(f"{label}:{m.group(1) if m else f.stem}")
                else:
                    groups.append(str(f))
    if not paths:
        warn("No WAV files found for task '%s' - skipping.", task)
        return None
    return TaskFiles(paths, np.asarray(labels, dtype=np.int64), groups)
