from pathlib import Path

from auditorynet.config import Config
from auditorynet.dataset import discover_files
from auditorynet.predict import load_ensemble, predict_file
from auditorynet.train import main as train_main
from auditorynet.utils import get_device


def test_discover_files(synthetic_data):
    tf = discover_files(str(synthetic_data), "e")  # case-insensitive
    assert len(tf.paths) == 24 and tf.labels.sum() == 12
    assert discover_files(str(synthetic_data), "missing") is None


def test_train_and_predict_smoke(synthetic_data, tmp_path):
    out = tmp_path / "out"
    train_main(["--data-dir", str(synthetic_data), "--output-dir", str(out),
                "--tasks", "E", "--epochs", "2", "--device", "cpu"])
    assert (out / "summary.csv").exists()
    assert (out / "E_5fold_results.png").exists()
    ckpts = sorted((out / "checkpoints").glob("E_fold*.pt"))
    assert len(ckpts) == 5
    dev = get_device("cpu")
    models, cfg = load_ensemble(ckpts, dev)
    wav = next(Path(synthetic_data, "E", "PD").glob("*.wav"))
    r = predict_file(str(wav), models, cfg, dev)
    assert 0.0 <= r["p_pd"] <= 1.0 and r["prediction"] in ("Healthy", "PD")
