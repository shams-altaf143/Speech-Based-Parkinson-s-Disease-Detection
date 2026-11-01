"""Result visualisation: per-task dashboard and cross-task comparison."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_task_dashboard(r: dict, path: Path) -> None:
    folds = r["folds"]
    x = np.arange(1, len(folds) + 1)
    acc = np.array([f["accuracy"] for f in folds]) * 100
    auc = np.array([f["auc"] for f in folds]) * 100
    g = r["aggregate"]
    cm = np.array(g["confusion_matrix"])
    rep = g["report"]

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle(f"AuditoryNet - Task {r['task']} | PD={r['n_pd']}, HC={r['n_hc']}", fontweight="bold")

    w = 0.38
    ax[0, 0].bar(x - w / 2, acc, w, label="Accuracy (%)")
    ax[0, 0].bar(x + w / 2, auc, w, label="AUC-ROC (%)")
    ax[0, 0].set(title="Per-fold accuracy & AUC-ROC", xlabel="Fold", ylim=(max(0, min(acc.min(), np.nanmin(auc)) - 5), 101))
    ax[0, 0].legend()

    im = ax[0, 1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0, 1].text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm.sum():.1%})", ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax[0, 1].set(title="Aggregated confusion matrix", xticks=[0, 1], yticks=[0, 1],
                 xticklabels=["Pred HC", "Pred PD"], yticklabels=["True HC", "True PD"])
    fig.colorbar(im, ax=ax[0, 1], fraction=0.046)

    ax[0, 2].axis("off")
    rows = [
        ["Mean accuracy", f"{r['mean_accuracy']*100:.2f} ± {r['std_accuracy']*100:.2f} %"],
        ["Mean AUC-ROC", f"{r['mean_auc']:.4f} ± {r['std_auc']:.4f}"],
        ["Sensitivity (PD)", f"{g['sensitivity']:.4f}"],
        ["Specificity (HC)", f"{g['specificity']:.4f}"],
        ["PD confidence", f"{g['pd_confidence']*100:.1f} %"],
        ["HC conf. error", f"{g['hc_conf_error']*100:.1f} %"],
        ["F1 - PD", f"{rep['PD']['f1-score']:.4f}"],
        ["F1 - HC", f"{rep['Healthy']['f1-score']:.4f}"],
    ]
    t = ax[0, 2].table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    t.scale(1, 1.8)
    ax[0, 2].set_title("Summary statistics")

    for vals, lab in ((acc, "Accuracy"), (auc, "AUC-ROC")):
        ax[1, 0].plot(x, vals, marker="o", label=lab)
        ax[1, 0].fill_between(x, vals.mean() - np.nanstd(vals), vals.mean() + np.nanstd(vals), alpha=0.15)
    ax[1, 0].set(title="Fold-wise trend (mean ± std)", xlabel="Fold", ylabel="Score (%)")
    ax[1, 0].legend()

    metrics = ["precision", "recall", "f1-score"]
    xm = np.arange(3)
    ax[1, 1].bar(xm - w / 2, [rep["Healthy"][m] for m in metrics], w, label="Healthy (HC)")
    ax[1, 1].bar(xm + w / 2, [rep["PD"][m] for m in metrics], w, label="Parkinson's (PD)")
    ax[1, 1].set(title="Class-wise precision / recall / F1", xticks=xm, xticklabels=metrics, ylim=(0, 1.05))
    ax[1, 1].legend()

    bd = {"PD acc\n(sensitivity)": g["sensitivity"] * 100, "HC acc\n(specificity)": g["specificity"] * 100,
          "PD conf.": g["pd_confidence"] * 100, "HC conf.\nerror": g["hc_conf_error"] * 100}
    ax[1, 2].bar(list(bd), list(bd.values()), color=["#c0392b", "#2c6e91", "#2a9d8f", "#e76f51"])
    for i, v in enumerate(bd.values()):
        ax[1, 2].text(i, v + 1, f"{v:.1f}%", ha="center")
    ax[1, 2].set(title="PD vs HC performance breakdown", ylim=(0, 110))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cross_task(results: list, path: Path) -> None:
    tasks = [r["task"] for r in results]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, key, sk, title, scale in (
        (ax[0], "mean_accuracy", "std_accuracy", "Mean accuracy (%) ± std", 100),
        (ax[1], "mean_auc", "std_auc", "Mean AUC-ROC ± std", 1),
    ):
        m = np.array([r[key] for r in results]) * scale
        s = np.array([r[sk] for r in results]) * scale
        a.bar(tasks, m, yerr=s, capsize=4, color=plt.cm.tab10(np.arange(len(tasks))))
        for i, v in enumerate(m):
            a.text(i, v + s[i] + 0.002 * scale, f"{v:.1f}" if scale == 100 else f"{v:.4f}", ha="center", fontsize=8)
        lo = max(0, m.min() - 10 * scale / 100 * 2)
        a.set(title=title, xlabel="Speech task", ylim=(lo, m.max() + 6 * scale / 100))
    fig.suptitle("AuditoryNet - cross-task performance (5-fold stratified CV)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(result: dict, path: Path) -> None:
    """Per-fold learning curves: train vs inner-val loss, and inner-val AUC.
    A widening train/val loss gap = overfitting."""
    hist = result["histories"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, h in enumerate(hist, 1):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax[0].plot(ep, h["train_loss"], label=f"fold {k}")
        ax[1].plot(ep, h["val_loss"], label=f"fold {k}")
        ax[2].plot(ep, h["val_auc"], label=f"fold {k}")
    ax[0].set(title="Training loss", xlabel="Epoch")
    ax[1].set(title="Inner-validation loss", xlabel="Epoch")
    ax[2].set(title="Inner-validation AUC-ROC", xlabel="Epoch")
    ax[0].legend(fontsize=8)
    gaps = [f["train_acc"] - f["accuracy"] for f in result["folds"]]
    fig.suptitle(f"Task {result['task']} - learning curves (mean train-test accuracy gap {np.mean(gaps)*100:.1f} pts)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
