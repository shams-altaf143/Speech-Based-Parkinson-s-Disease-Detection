import pytest

from auditorynet import gui_utils as U


def test_scan_and_regex(synthetic_data):
    rows = {r["Task"]: r for r in U.scan_dataset(str(synthetic_data), ["E", "petaka", "U"])}
    assert rows["E"]["Status"] == "OK" and rows["E"]["PD files"] == 12
    assert rows["U"]["Status"].startswith("Missing")
    assert U.validate_regex("") is None
    assert U.validate_regex("(") .startswith("Invalid")
    assert "capture group" in U.validate_regex("abc")
    assert U.validate_regex(r"^(spk\d+)_") is None


def test_write_config_roundtrip(tmp_path):
    from auditorynet.config import Config
    p = tmp_path / "c.yaml"
    U.write_config({"epochs": 3, "tasks": ["E"], "speaker_regex": ""}, p)
    cfg = Config.from_yaml(p)
    assert cfg.epochs == 3 and cfg.tasks == ["E"] and cfg.speaker_regex is None
    with pytest.raises(KeyError):
        U.write_config({"nope": 1}, p)


def test_parse_progress():
    log = ("Task E | fold 2/5 (train 1, inner-val 1, test 1)\n  ep 05/25  loss\n")
    frac, label = U.parse_progress(log, ["E", "I"], 5, 25)
    assert 0 < frac < 1 and "fold 2/5" in label and "epoch 5/25" in label
    assert U.parse_progress("Done in 1.0 min", ["E"], 5, 25)[0] == 1.0


def test_app_renders(synthetic_data, tmp_path):
    pytest.importorskip("streamlit")
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = Path(__file__).parents[1] / "src" / "auditorynet" / "gui_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    assert not at.exception
    at.sidebar.text_input(key="data_dir").set_value(str(synthetic_data))
    at.sidebar.text_input(key="output_dir").set_value(str(tmp_path / "out"))
    at.run()
    assert not at.exception
    assert any("Ready to train" in s.value for s in at.success)


def test_results_and_predict_tabs_render(tmp_path):
    pytest.importorskip("streamlit")
    import json
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    out = tmp_path / "out"
    (out / "checkpoints").mkdir(parents=True)
    (out / "checkpoints" / "E_fold1.pt").write_bytes(b"x")
    (out / "summary.csv").write_text(
        "task,n_hc,n_pd,mean_acc,std_acc,mean_auc,std_auc,sensitivity,specificity,train_acc,train_test_gap\n"
        "E,300,300,0.93,0.01,0.99,0.00,0.98,0.89,1.0,0.17\n")
    (out / "E_results.json").write_text(json.dumps({"folds": [{"fold": 1, "accuracy": 0.9}]}))
    app = Path(__file__).parents[1] / "src" / "auditorynet" / "gui_app.py"
    at = AppTest.from_file(str(app), default_timeout=60).run()
    at.sidebar.text_input(key="output_dir").set_value(str(out))
    at.run()
    assert not at.exception
    assert any("overfitting" in w.value for w in at.warning)
