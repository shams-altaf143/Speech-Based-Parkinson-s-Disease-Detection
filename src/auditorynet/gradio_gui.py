"""Gradio GUI for AuditoryNet (designed for Google Colab, works anywhere).

    from auditorynet.gradio_gui import App
    App("data", "outputs").build().queue().launch(share=True)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import pandas as pd

from auditorynet import gui_utils as U

ALL_TASKS = ["A", "E", "I", "O", "U", "petaka", "apto"]
DISCLAIMER = ("**Research prototype - NOT a medical device.** The score only shows how similar a recording is "
              "to the Parkinson's samples the model was trained on. It is not a diagnosis; consult a clinician "
              "for any health concern.")


class App:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir, self.output_dir = data_dir, output_dir
        self.proc: Optional[subprocess.Popen] = None
        self.stop_flag = False
        self._cache: Dict = {}

    # ------------------------------------------------------------------ dataset
    def scan(self):
        rows = U.scan_dataset(self.data_dir, ALL_TASKS)
        ok = [r["Task"] for r in rows if r["Status"] == "OK"]
        msg = (f"✅ Ready to train: **{', '.join(ok)}**" if ok
               else f"⚠️ No usable task found in `{self.data_dir}`.")
        return pd.DataFrame(rows), msg

    # ------------------------------------------------------------------ training
    def _running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def merge_results(self) -> int:
        """Rebuild summary.csv and the cross-task plot from all *_results.json files."""
        from auditorynet.plots import plot_cross_task

        out = Path(self.output_dir)
        results = [json.loads(p.read_text()) for p in out.glob("*_results.json")]
        order = {t: i for i, t in enumerate(ALL_TASKS)}
        results.sort(key=lambda r: order.get(r["task"], 99))
        if not results:
            return 0
        rows = ["task,n_hc,n_pd,mean_acc,std_acc,mean_auc,std_auc,sensitivity,specificity,train_acc,train_test_gap"]
        for r in results:
            g = r["aggregate"]
            rows.append(f"{r['task']},{r['n_hc']},{r['n_pd']},{r['mean_accuracy']:.4f},{r['std_accuracy']:.4f},"
                        f"{r['mean_auc']:.4f},{r['std_auc']:.4f},{g['sensitivity']:.4f},{g['specificity']:.4f},"
                        f"{r['mean_train_acc']:.4f},{r['train_test_gap']:.4f}")
        (out / "summary.csv").write_text("\n".join(rows) + "\n")
        plot_cross_task(results, out / "all_datasets_comparison.png")
        return len(results)

    def train(self, tasks, epochs, batch, lr, dropout, smoothing, patience, speaker_regex, force):
        """Generator: streams (log, status) while training runs in a subprocess."""
        if self._running():
            raise gr.Error("Training is already running.")
        if not tasks:
            raise gr.Error("Select at least one task.")
        err = U.validate_regex(speaker_regex or "")
        if err:
            raise gr.Error(err)
        scan = {r["Task"]: r for r in U.scan_dataset(self.data_dir, list(tasks))}
        bad = [t for t, r in scan.items() if r["Status"] != "OK"]
        if bad:
            raise gr.Error(f"Not trainable (see Dataset tab): {', '.join(bad)}")

        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg_path = out / "gui_config.yaml"
        U.write_config({"data_dir": self.data_dir, "output_dir": self.output_dir, "tasks": list(tasks),
                        "epochs": int(epochs), "batch_size": int(batch), "lr": float(lr),
                        "dropout": float(dropout), "label_smoothing": float(smoothing),
                        "patience": int(patience), "device": "auto",
                        "speaker_regex": (speaker_regex or "").strip() or None}, cfg_path)
        self.stop_flag = False
        log, failed = "", []
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        for i, t in enumerate(tasks):
            if self.stop_flag:
                break
            if (out / f"{t}_results.json").exists() and not force:
                log += f"[skip] {t}: results already exist (tick 'Retrain' to redo)\n"
                yield log[-7000:], f"Task {t}: skipped"
                continue
            log += f"\n===== TASK {t} ({i + 1}/{len(tasks)}) =====\n"
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "auditorynet.train", "--config", str(cfg_path), "--tasks", t],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            task_log = ""
            for line in self.proc.stdout:
                log += line
                task_log += line
                _, label = U.parse_progress(task_log, [t], 5, int(epochs))
                yield log[-7000:], f"Task {t} ({i + 1}/{len(tasks)}): {label}"
            rc = self.proc.wait()
            if rc != 0 and not self.stop_flag:
                failed.append(t)
                log += f"!! Task {t} failed (exit code {rc}) - see the messages above\n"
        n = self.merge_results()
        if self.stop_flag:
            status = "Stopped by user."
        elif failed:
            status = f"Finished with errors in: {', '.join(failed)}"
        else:
            status = f"All done ({n} task result(s)). Open the Results tab."
        yield log[-7000:], status

    def stop(self):
        self.stop_flag = True
        if self._running():
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        return "Stop requested."

    # ------------------------------------------------------------------ results
    def results_view(self):
        out = Path(self.output_dir)
        p = out / "summary.csv"
        if not p.exists():
            return pd.DataFrame(), [], "No results yet - train first."
        df = pd.read_csv(p)
        warns = [f"⚠️ Task {r['task']}: train-test accuracy gap {r['train_test_gap'] * 100:.1f} points "
                 "→ overfitting (see learning curves)." for _, r in df.iterrows() if r["train_test_gap"] > 0.10]
        imgs = []
        cmp_img = out / "all_datasets_comparison.png"
        if cmp_img.exists():
            imgs.append((str(cmp_img), "Cross-task comparison"))
        for t in df["task"].astype(str):
            for name, cap in ((f"{t}_5fold_results.png", "dashboard"), (f"{t}_history.png", "learning curves")):
                if (out / name).exists():
                    imgs.append((str(out / name), f"Task {t} - {cap}"))
        return df, imgs, "\n\n".join(warns) or "✅ No overfitting warning (train-test gap ≤ 10 points)."

    # ------------------------------------------------------------------ predict
    def model_tasks(self) -> List[str]:
        return U.available_checkpoint_tasks(Path(self.output_dir) / "checkpoints")

    def refresh_models(self):
        tasks = self.model_tasks()
        return gr.update(choices=tasks, value=tasks[0] if tasks else None)

    def _load(self, task: str):
        ck = sorted((Path(self.output_dir) / "checkpoints").glob(f"{task}_fold*.pt"))
        if not ck:
            raise gr.Error(f"No trained models for task '{task}'. Train first.")
        key = (task, tuple(f.stat().st_mtime for f in ck))
        if key not in self._cache:
            from auditorynet.predict import load_ensemble
            from auditorynet.utils import get_device

            dev = get_device("auto")
            models, cfg = load_ensemble(ck, dev)
            self._cache = {key: (models, cfg, dev)}
        return self._cache[key]

    @staticmethod
    def _coch_fig(wav_path: str, cfg):
        import numpy as np
        import torch
        from matplotlib.figure import Figure

        from auditorynet.audio import fix_length, load_raw, normalise
        from auditorynet.gammatone import Cochleagram

        w = normalise(fix_length(load_raw(wav_path, cfg.sample_rate), cfg.n_samples)).astype(np.float32)
        c = Cochleagram.from_config(cfg)(torch.from_numpy(w)[None])[0, 0].numpy()
        fig = Figure(figsize=(7, 3))
        ax = fig.subplots()
        ax.imshow(c, origin="lower", aspect="auto", cmap="magma", extent=[0, cfg.duration, 0, cfg.n_channels])
        ax.set(xlabel="Time (s)", ylabel="Gammatone channel (low → high Hz)", title="Cochleagram (first 3 s)")
        fig.tight_layout()
        return fig

    def predict(self, audio_path, task, threshold):
        if not audio_path:
            raise gr.Error("Upload or record a voice sample first.")
        if not task:
            raise gr.Error("No trained model selected. Train first, then press 'Refresh models'.")
        try:
            from matplotlib.figure import Figure

            from auditorynet.audio import load_raw
            from auditorynet.predict import predict_file

            models, cfg, dev = self._load(task)
            secs = len(load_raw(audio_path, cfg.sample_rate)) / cfg.sample_rate
            res = predict_file(audio_path, models, cfg, dev, float(threshold))
            p = res["p_pd"]
            note = f"\n\n⚠️ The recording is only {secs:.1f}s long; results may be unreliable." if secs < 1 else ""
            verdict = "closer to **Parkinson's (PD)** samples" if p >= threshold else "closer to **healthy** samples"
            text = (f"### PD likelihood score: {p:.1%}\nThe voice is {verdict} (threshold {threshold:.2f}, "
                    f"{res['n_windows']} window(s), {len(models)} models).{note}\n\n{DISCLAIMER}")
            wf = Figure(figsize=(6, 2.6))
            ax = wf.subplots()
            ax.bar(range(1, res["n_windows"] + 1), res["window_p_pd"], color="#2c6e91")
            ax.axhline(threshold, color="red", ls="--", lw=1)
            ax.set(ylim=(0, 1), xlabel="3-second window", ylabel="PD score", title="Score per window")
            wf.tight_layout()
            return ({"Closer to PD samples": p, "Closer to healthy samples": 1 - p}, text, wf,
                    self._coch_fig(audio_path, cfg))
        except gr.Error:
            raise
        except Exception as e:  # unreadable audio, etc.
            raise gr.Error(f"Could not analyse this file: {e}")

    # ------------------------------------------------------------------ layout
    def build(self) -> gr.Blocks:
        usable = [r["Task"] for r in U.scan_dataset(self.data_dir, ALL_TASKS) if r["Status"] == "OK"]
        mtasks = self.model_tasks()
        with gr.Blocks(title="AuditoryNet") as demo:
            gr.Markdown("# 🎙️ AuditoryNet - Parkinson's Disease Speech Analysis\n"
                        "Gammatone cochleagram + auditory-inspired CNN")
            gr.Markdown(DISCLAIMER)

            with gr.Tab("1 · Dataset"):
                msg = gr.Markdown()
                tbl = gr.Dataframe(interactive=False, label="Tasks found")
                gr.Button("Scan dataset").click(self.scan, None, [tbl, msg])
                demo.load(self.scan, None, [tbl, msg])

            with gr.Tab("2 · Train"):
                with gr.Row():
                    with gr.Column():
                        tasks = gr.CheckboxGroup(ALL_TASKS, value=usable or ["E"], label="Tasks")
                        epochs = gr.Slider(1, 100, 25, step=1, label="Max epochs")
                        batch = gr.Number(16, label="Batch size", precision=0)
                    with gr.Column():
                        lr = gr.Number(1e-3, label="Learning rate")
                        dropout = gr.Slider(0, 0.8, 0.4, step=0.05, label="Dropout")
                        smooth = gr.Slider(0, 0.3, 0.05, step=0.01, label="Label smoothing")
                    with gr.Column():
                        patience = gr.Number(8, label="Early-stopping patience", precision=0)
                        rx = gr.Textbox("", label="Speaker regex (optional)",
                                        info=r"One capture group, e.g. ^(spk\d+)_ - makes folds speaker-disjoint")
                        force = gr.Checkbox(False, label="Retrain tasks that already have results")
                with gr.Row():
                    start = gr.Button("▶ Start training", variant="primary")
                    stop = gr.Button("■ Stop")
                status = gr.Textbox(label="Status", interactive=False)
                log = gr.Textbox(label="Training log", lines=18, max_lines=18, interactive=False)
                start.click(self.train, [tasks, epochs, batch, lr, dropout, smooth, patience, rx, force],
                            [log, status])
                stop.click(self.stop, None, status)

            with gr.Tab("3 · Results"):
                refresh = gr.Button("Refresh results")
                warn = gr.Markdown()
                summary = gr.Dataframe(interactive=False, label="Summary (5-fold CV, test folds)")
                gallery = gr.Gallery(label="Plots", columns=2, height="auto")
                refresh.click(self.results_view, None, [summary, gallery, warn])
                demo.load(self.results_view, None, [summary, gallery, warn])

            with gr.Tab("4 · Predict"):
                with gr.Row():
                    with gr.Column():
                        task = gr.Dropdown(mtasks, value=mtasks[0] if mtasks else None,
                                           label="Speech task of the recording",
                                           info="Must match what the person said: vowel A/E/I/O/U, petaka or apto")
                        gr.Button("Refresh models").click(self.refresh_models, None, task)
                        thr = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="Decision threshold")
                        audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                         label="Upload or record voice (WAV)")
                        go = gr.Button("Analyse", variant="primary")
                    with gr.Column():
                        label = gr.Label(label="Model output")
                        text = gr.Markdown()
                        wplot = gr.Plot(label="Per-window score")
                        cplot = gr.Plot(label="Cochleagram")
                go.click(self.predict, [audio, task, thr], [label, text, wplot, cplot])
        return demo
