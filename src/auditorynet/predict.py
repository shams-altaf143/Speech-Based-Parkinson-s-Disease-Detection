"""Inference: average the probabilities of all fold checkpoints of a task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .audio import fix_length, load_raw, normalise
from .config import Config
from .discovery import CLASS_NAMES
from .gammatone import Cochleagram
from .model import AuditoryNet
from .utils import get_device

DISCLAIMER = ("Research prototype - NOT a medical device. Output is a screening-style score "
              "and must not be used for diagnosis.")


def load_ensemble(ckpt_paths: List[Path], device: torch.device):
    models, cfg = [], None
    for p in ckpt_paths:
        ck = torch.load(p, map_location=device)
        cfg = Config(**ck["config"])
        m = AuditoryNet(cfg.dropout).to(device)
        m.load_state_dict(ck["state_dict"])
        models.append(m.eval())
    return models, cfg


@torch.no_grad()
def predict_file(wav: str, models, cfg: Config, device: torch.device, threshold: float = 0.5) -> dict:
    """Split the recording into non-overlapping windows of cfg.duration (last window kept
    if >= 50 % full), score each with every fold model, and average P(PD)."""
    wave = load_raw(wav, cfg.sample_rate)
    n = cfg.n_samples
    starts = list(range(0, max(1, len(wave) - n // 2), n))
    windows = np.stack([normalise(fix_length(wave[s : s + n], n)) for s in starts]).astype(np.float32)
    coch = Cochleagram.from_config(cfg).to(device)
    x = coch(torch.from_numpy(windows).to(device))
    probs = torch.stack([torch.softmax(m(x), 1)[:, 1] for m in models]).mean(0).cpu().numpy()
    p = float(probs.mean())
    return {"file": str(wav), "n_windows": len(starts), "p_pd": p,
            "window_p_pd": [float(v) for v in probs],
            "prediction": CLASS_NAMES[int(p >= threshold)], "threshold": threshold,
            "disclaimer": DISCLAIMER}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Score WAV files with a trained AuditoryNet ensemble.")
    ap.add_argument("wav", nargs="+")
    ap.add_argument("--task", required=True, help="Task name used in training, e.g. E or petaka")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)

    ckpts = sorted(Path(a.ckpt_dir).glob(f"{a.task}_fold*.pt"))
    if not ckpts:
        raise SystemExit(f"No checkpoints for task '{a.task}' in {a.ckpt_dir}")
    device = get_device(a.device)
    models, cfg = load_ensemble(ckpts, device)
    for w in a.wav:
        print(json.dumps(predict_file(w, models, cfg, device, a.threshold), indent=2))


if __name__ == "__main__":
    main()
