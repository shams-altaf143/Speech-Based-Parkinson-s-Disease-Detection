"""Nested-selection 5-fold stratified CV training - CLI entry point.

For every outer fold the training part is split again into train / inner-validation.
The inner-validation set picks the best epoch (early stopping); the outer test fold is
only used once for reporting, so scores are NOT biased by model selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import Config
from .dataset import CochleagramDataset, discover_files, extract_features
from .metrics import safe_auc, summarise
from .model import AuditoryNet, count_parameters
from .plots import plot_cross_task, plot_history, plot_task_dashboard
from .utils import get_device, get_logger, seed_everything

log = get_logger()


@torch.no_grad()
def predict_probs(model: nn.Module, feats: torch.Tensor, labels: np.ndarray, cfg: Config,
                  device: torch.device):
    """Returns (mean loss, P(PD) array). No augmentation."""
    model.eval()
    dl = DataLoader(CochleagramDataset(feats, labels, False, cfg), batch_size=64, shuffle=False)
    probs, loss_sum = [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += ce(logits, y).item()
        probs.append(torch.softmax(logits, 1)[:, 1].cpu())
    return loss_sum / len(labels), torch.cat(probs).numpy()


def inner_split(y: np.ndarray, groups: list, cfg: Config, seed: int):
    """Stratified (and speaker-disjoint if speaker_regex is set) train / inner-val split."""
    n_val = max(4, int(round(cfg.val_fraction * len(y))))
    if cfg.speaker_regex:
        k = max(2, int(round(1 / cfg.val_fraction)))
        sgkf = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
        tr, va = next(sgkf.split(np.zeros(len(y)), y, groups))
    else:
        sss = StratifiedShuffleSplit(1, test_size=n_val, random_state=seed)
        tr, va = next(sss.split(np.zeros(len(y)), y))
    return tr, va


def train_fold(cfg: Config, x_tr, y_tr, x_va, y_va, device, ckpt_path: Path) -> dict:
    model = AuditoryNet(cfg.dropout).to(device)
    counts = np.bincount(y_tr, minlength=2).astype(np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(1.0 / counts[y_tr], dtype=torch.double),
        num_samples=len(y_tr), replacement=True,
    )
    loader = DataLoader(CochleagramDataset(x_tr, y_tr, True, cfg), batch_size=cfg.batch_size,
                        sampler=sampler, num_workers=cfg.num_workers, drop_last=len(y_tr) > cfg.batch_size)
    # Eq. 3.12: w_PD = 1, w_HC = N_PD / N_HC  (+ label smoothing against over-confidence)
    class_w = torch.tensor([counts[1] / counts[0], 1.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_key, best_state, best_epoch, bad = (-1.0, 0.0), None, -1, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss, tr_correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(y)
            tr_correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        va_loss, va_p = predict_probs(model, x_va, y_va, cfg, device)
        va_acc = float(((va_p >= 0.5).astype(int) == y_va).mean())
        va_auc = safe_auc(y_va, va_p)
        for k, v in zip(hist, (tr_loss / n, tr_correct / n, va_loss, va_acc, va_auc)):
            hist[k].append(float(v))
        # select on AUC, break ties with lower validation loss
        key = ((va_auc if not np.isnan(va_auc) else va_acc), -va_loss)
        if key > best_key:
            best_key, best_epoch, bad = key, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        log.info("  ep %02d/%d  loss %.4f acc %.3f | val loss %.4f acc %.3f auc %.4f",
                 epoch, cfg.epochs, tr_loss / n, tr_correct / n, va_loss, va_acc, va_auc)
        if bad >= cfg.patience:
            log.info("  early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "config": cfg.to_dict(), "best_epoch": best_epoch,
                "val_auc": float(best_key[0])}, ckpt_path)
    return {"model": model, "best_epoch": best_epoch, "history": hist}


def run_task(cfg: Config, task: str, device: torch.device) -> Optional[dict]:
    files = discover_files(cfg.data_dir, task, cfg.speaker_regex)
    if files is None:
        return None
    log.info("Task %s: extracting cochleagrams for %d files", task, len(files.paths))
    feats, kept = extract_features(files.paths, cfg, device)
    y = files.labels[kept]
    groups = [files.groups[i] for i in kept]
    counts = np.bincount(y, minlength=2)
    if counts.min() < max(cfg.n_folds, 10):
        log.warning("Task %s: class counts %s too small - skipping.", task, counts)
        return None
    log.info("Task %s: HC=%d PD=%d", task, counts[0], counts[1])

    if cfg.speaker_regex:
        splits = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(
            np.zeros(len(y)), y, groups)
    else:
        splits = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed).split(np.zeros(len(y)), y)

    oof = np.full(len(y), np.nan)
    fold_metrics, histories = [], []
    out = Path(cfg.output_dir)
    for k, (tr_all, te) in enumerate(splits, 1):
        seed_everything(cfg.seed + k)
        itr, iva = inner_split(y[tr_all], [groups[i] for i in tr_all], cfg, cfg.seed + k)
        tr, va = tr_all[itr], tr_all[iva]
        log.info("Task %s | fold %d/%d (train %d, inner-val %d, test %d)",
                 task, k, cfg.n_folds, len(tr), len(va), len(te))
        res = train_fold(cfg, feats[tr], y[tr], feats[va], y[va], device,
                         out / "checkpoints" / f"{task}_fold{k}.pt")
        model = res["model"]
        _, p_te = predict_probs(model, feats[te], y[te], cfg, device)
        _, p_tr = predict_probs(model, feats[tr], y[tr], cfg, device)
        oof[te] = p_te
        m = summarise(y[te], p_te)
        tr_acc = float(((p_tr >= 0.5).astype(int) == y[tr]).mean())
        fold_metrics.append({"fold": k, "accuracy": m["accuracy"], "auc": m["auc"],
                             "train_acc": tr_acc, "best_epoch": res["best_epoch"]})
        histories.append(res["history"])
        log.info("Task %s | fold %d -> TEST acc %.4f auc %.4f | train acc %.4f (gap %.1f pts)",
                 task, k, m["accuracy"], m["auc"], tr_acc, (tr_acc - m["accuracy"]) * 100)

    agg = summarise(y, oof)
    accs = np.array([f["accuracy"] for f in fold_metrics])
    aucs = np.array([f["auc"] for f in fold_metrics])
    tr_accs = np.array([f["train_acc"] for f in fold_metrics])
    gap = float((tr_accs - accs).mean())
    if gap > 0.10:
        log.warning("Task %s: train-test accuracy gap %.1f pts -> overfitting. See %s_history.png.",
                    task, gap * 100, task)
    result = {
        "task": task, "n_hc": int(counts[0]), "n_pd": int(counts[1]),
        "folds": fold_metrics,
        "mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()),
        "mean_auc": float(np.nanmean(aucs)), "std_auc": float(np.nanstd(aucs)),
        "mean_train_acc": float(tr_accs.mean()), "train_test_gap": gap,
        "aggregate": agg, "histories": histories,
    }
    (out / f"{task}_results.json").write_text(json.dumps(result, indent=2))
    plot_task_dashboard(result, out / f"{task}_5fold_results.png")
    plot_history(result, out / f"{task}_history.png")
    return result


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Train AuditoryNet with 5-fold stratified CV.")
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    cfg = Config.from_yaml(a.config)
    for k in ("data_dir", "output_dir", "tasks", "epochs", "device"):
        v = getattr(a, k)
        if v is not None:
            setattr(cfg, k, v)

    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output_dir) / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    log.info("Device: %s | trainable params: %s", device, f"{count_parameters(AuditoryNet(cfg.dropout)):,}")

    results, t0 = [], time.time()
    for task in cfg.tasks:
        r = run_task(cfg, task, device)
        if r:
            results.append(r)
    if not results:
        raise SystemExit(f"No tasks were trained. Check data_dir='{cfg.data_dir}' layout (see README).")

    rows = ["task,n_hc,n_pd,mean_acc,std_acc,mean_auc,std_auc,sensitivity,specificity,train_acc,train_test_gap"]
    for r in results:
        g = r["aggregate"]
        rows.append(f"{r['task']},{r['n_hc']},{r['n_pd']},{r['mean_accuracy']:.4f},{r['std_accuracy']:.4f},"
                    f"{r['mean_auc']:.4f},{r['std_auc']:.4f},{g['sensitivity']:.4f},{g['specificity']:.4f},"
                    f"{r['mean_train_acc']:.4f},{r['train_test_gap']:.4f}")
    (Path(cfg.output_dir) / "summary.csv").write_text("\n".join(rows) + "\n")
    plot_cross_task(results, Path(cfg.output_dir) / "all_datasets_comparison.png")
    log.info("Done in %.1f min. Results in %s", (time.time() - t0) / 60, cfg.output_dir)


if __name__ == "__main__":
    main()
