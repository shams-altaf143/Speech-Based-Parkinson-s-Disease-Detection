"""AuditoryNet GUI (Streamlit). Launch with:  auditorynet-gui"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

from auditorynet import gui_utils as U

ALL_TASKS = ["A", "E", "I", "O", "U", "petaka", "apto"]
DISCLAIMER = ("Research prototype - NOT a medical device. The score only indicates how similar a "
              "recording is to the Parkinson's samples the model was trained on. It is not a diagnosis; "
              "consult a clinician for any health concern.")

st.set_page_config(page_title="AuditoryNet", page_icon="🎙️", layout="wide")

for k, v in {"proc": None, "run_tasks": [], "run_epochs": 25, "run_folds": 5,
             "log_path": None, "finished_seen": True}.items():
    st.session_state.setdefault(k, v)


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("🎙️ AuditoryNet")
st.sidebar.caption("Speech-based Parkinson's detection\nGammatone cochleagram + auditory-inspired CNN")
data_dir = st.sidebar.text_input("Dataset folder", "data", key="data_dir")
output_dir = st.sidebar.text_input("Output folder", "outputs", key="output_dir")
st.sidebar.caption(f"Working folder: `{os.getcwd()}`")
st.sidebar.warning(DISCLAIMER)

st.title("AuditoryNet - Parkinson's Disease Speech Analysis")
tab_data, tab_train, tab_res, tab_pred, tab_about = st.tabs(
    ["1 · Dataset", "2 · Train", "3 · Results", "4 · Predict", "About"])


# ----------------------------------------------------------------------------- dataset
with tab_data:
    st.subheader("Dataset check")
    st.markdown("Expected layout (folder names are not case-sensitive):")
    st.code("data/\n  E/  PD/*.wav  Healthy/*.wav\n  I/  O/  U/  A/  petaka/  apto/   (same structure)", language="text")
    rows = U.scan_dataset(data_dir, ALL_TASKS)
    st.dataframe(rows, hide_index=True)
    ready = [r["Task"] for r in rows if r["Status"] == "OK"]
    if ready:
        st.success(f"Ready to train: {', '.join(ready)}")
    else:
        st.info(f"No usable task found in `{data_dir}`. Create the folders above and add WAV files.")


# ----------------------------------------------------------------------------- train
def _running() -> bool:
    p = st.session_state.proc
    return p is not None and p.poll() is None


def _start_training(settings: dict) -> None:
    out = Path(settings["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "gui_config.yaml"
    U.write_config(settings, cfg_path)
    log_path = out / "train.log"
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    with open(log_path, "w", encoding="utf-8") as fh:
        st.session_state.proc = subprocess.Popen(
            [sys.executable, "-m", "auditorynet.train", "--config", str(cfg_path)],
            stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=os.getcwd(), **kwargs)
    st.session_state.log_path = log_path
    st.session_state.run_tasks = list(settings["tasks"])
    st.session_state.run_epochs = settings["epochs"]
    st.session_state.run_folds = settings["n_folds"]
    st.session_state.finished_seen = False


def _stop_training() -> None:
    p = st.session_state.proc
    if p is not None and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    st.session_state.finished_seen = False


def _monitor() -> None:
    p = st.session_state.proc
    if p is None:
        st.caption("No training run started in this session.")
        return
    text = U.tail_file(st.session_state.log_path)
    frac, label = U.parse_progress(text, st.session_state.run_tasks,
                                   st.session_state.run_folds, st.session_state.run_epochs)
    rc = p.poll()
    if rc is None:
        st.progress(frac, text=f"{label}  (approximate)")
    elif rc == 0:
        st.progress(1.0, text="Finished")
        st.success("Training finished. Open the Results tab.")
    else:
        st.error(f"Training stopped with exit code {rc}. See the log below.")
    st.code(text[-3500:] or "(waiting for output...)", language="text")
    if rc is not None and not st.session_state.finished_seen:
        st.session_state.finished_seen = True
        st.rerun()


with tab_train:
    st.subheader("Train AuditoryNet (5-fold stratified cross-validation)")
    running = _running()
    c1, c2, c3 = st.columns(3)
    with c1:
        tasks = st.multiselect("Tasks", ALL_TASKS, default=ready or ["E"], disabled=running)
        epochs = st.number_input("Max epochs", 1, 300, 25, disabled=running)
        batch = st.number_input("Batch size", 2, 256, 16, disabled=running)
    with c2:
        lr = st.number_input("Learning rate", 1e-5, 1e-1, 1e-3, format="%.5f", disabled=running)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.4, 0.05, disabled=running)
        smoothing = st.slider("Label smoothing", 0.0, 0.3, 0.05, 0.01, disabled=running)
    with c3:
        patience = st.number_input("Early-stopping patience", 1, 50, 8, disabled=running)
        device = st.selectbox("Device", ["auto", "cpu", "cuda"], disabled=running)
        speaker_regex = st.text_input(
            "Speaker regex (optional)", "", disabled=running,
            help="One capture group extracting the speaker id from the file name, e.g. ^(spk\\d+)_ . "
                 "Makes folds speaker-disjoint (recommended when speakers have several recordings).")
    rx_err = U.validate_regex(speaker_regex)
    if rx_err:
        st.error(rx_err)
    not_ready = [t for t in tasks if t not in ready]
    if not_ready:
        st.warning(f"Not trainable yet (see Dataset tab): {', '.join(not_ready)}")

    b1, b2, _ = st.columns([1, 1, 4])
    can_start = bool(tasks) and not rx_err and not running and not not_ready
    if b1.button("▶ Start training", type="primary", disabled=not can_start):
        _start_training({
            "data_dir": data_dir, "output_dir": output_dir, "tasks": tasks, "epochs": int(epochs),
            "batch_size": int(batch), "lr": float(lr), "dropout": float(dropout),
            "label_smoothing": float(smoothing), "patience": int(patience), "device": device,
            "speaker_regex": speaker_regex.strip() or None, "n_folds": 5,
        })
        st.rerun()
    if b2.button("■ Stop", disabled=not running):
        _stop_training()
        st.rerun()

    st.fragment(run_every=2 if running else None)(_monitor)()


# ----------------------------------------------------------------------------- results
with tab_res:
    st.subheader("Results")
    out = Path(output_dir)
    summary = U.read_summary(out)
    if summary is None:
        st.info(f"No `summary.csv` in `{out}` yet. Train first (tab 2).")
    else:
        st.dataframe(summary, hide_index=True)
        st.download_button("Download summary.csv", (out / "summary.csv").read_bytes(), "summary.csv", "text/csv")
        if "train_test_gap" in summary:
            for _, r in summary.iterrows():
                if r["train_test_gap"] > 0.10:
                    st.warning(f"Task {r['task']}: train-test accuracy gap {r['train_test_gap']*100:.1f} points "
                               "→ overfitting. Check the learning curves below.")
        cmp_img = out / "all_datasets_comparison.png"
        if cmp_img.exists():
            st.image(str(cmp_img), caption="Cross-task comparison")
        task = st.selectbox("Task details", list(summary["task"].astype(str)))
        for name, cap in ((f"{task}_5fold_results.png", "Dashboard"), (f"{task}_history.png", "Learning curves")):
            if (out / name).exists():
                st.image(str(out / name), caption=cap)
        rj = out / f"{task}_results.json"
        if rj.exists():
            folds = json.loads(rj.read_text())["folds"]
            st.caption("Per-fold test results")
            st.dataframe(folds, hide_index=True)


# ----------------------------------------------------------------------------- predict
@st.cache_resource(show_spinner="Loading trained models...")
def _load_models(ckpt_dir: str, task: str, stamp: tuple):
    from auditorynet.predict import load_ensemble
    from auditorynet.utils import get_device

    paths = sorted(Path(ckpt_dir).glob(f"{task}_fold*.pt"))
    device = get_device("auto")
    models, cfg = load_ensemble(paths, device)
    return models, cfg, device


def _cochleagram_figure(wav_path: str, cfg):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from auditorynet.audio import fix_length, load_raw, normalise
    from auditorynet.gammatone import Cochleagram

    w = normalise(fix_length(load_raw(wav_path, cfg.sample_rate), cfg.n_samples)).astype(np.float32)
    c = Cochleagram.from_config(cfg)(torch.from_numpy(w)[None])[0, 0].numpy()
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.imshow(c, origin="lower", aspect="auto", cmap="magma", extent=[0, cfg.duration, 0, cfg.n_channels])
    ax.set(xlabel="Time (s)", ylabel="Gammatone channel (low → high Hz)", title="Cochleagram (first 3 s)")
    fig.tight_layout()
    return fig


with tab_pred:
    st.subheader("Analyse a voice recording")
    ckpt_dir = Path(output_dir) / "checkpoints"
    ptasks = U.available_checkpoint_tasks(ckpt_dir)
    if not ptasks:
        st.info(f"No trained models in `{ckpt_dir}`. Train first (tab 2).")
    else:
        task = st.selectbox("Speech task of the recording", ptasks,
                            help="Must match what the person said: sustained vowel (A/E/I/O/U), petaka, or apto.")
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
        upload = st.file_uploader("Upload a WAV file", type=["wav"])
        recorded = st.audio_input("...or record with your microphone") if hasattr(st, "audio_input") else None
        src = recorded or upload
        if src is not None:
            data = src.getvalue()
            st.audio(data, format="audio/wav")
            if st.button("Analyse", type="primary"):
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                try:
                    tmp.write(data)
                    tmp.close()
                    from auditorynet.audio import load_raw
                    from auditorynet.predict import predict_file

                    stamp = tuple(f.stat().st_mtime for f in sorted(ckpt_dir.glob(f"{task}_fold*.pt")))
                    models, cfg, device = _load_models(str(ckpt_dir), task, stamp)
                    secs = len(load_raw(tmp.name, cfg.sample_rate)) / cfg.sample_rate
                    if secs < 1.0:
                        st.warning(f"The recording is only {secs:.1f}s long; results may be unreliable.")
                    res = predict_file(tmp.name, models, cfg, device, threshold)
                    p = res["p_pd"]
                    m1, m2 = st.columns(2)
                    m1.metric("PD likelihood score", f"{p:.1%}")
                    m2.metric("Model output", "Closer to PD samples" if p >= threshold
                              else "Closer to healthy samples")
                    st.progress(float(p))
                    if res["n_windows"] > 1:
                        st.caption("Score per 3-second window")
                        st.bar_chart(res["window_p_pd"])
                    fig = _cochleagram_figure(tmp.name, cfg)
                    st.pyplot(fig)
                    st.warning(DISCLAIMER)
                except Exception as e:  # unreadable audio, missing libs, ...
                    st.error(f"Could not analyse this file: {e}")
                finally:
                    try:
                        os.unlink(tmp.name)
                    except OSError:
                        pass


# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**AuditoryNet** converts speech into a 64-channel Gammatone *cochleagram* and classifies it with a CNN whose
blocks mimic the primary auditory cortex (A1Block) and auditory belt (BeltBlock).

**Workflow:** 1) check your dataset → 2) train → 3) inspect results and overfitting → 4) analyse new recordings.

**Tips:** keep one speaker's recordings in the same fold with *Speaker regex*; use the same task and a similar
microphone/environment as the training data; scores from a small dataset do not generalise to the clinic.
""")
    st.warning(DISCLAIMER)
