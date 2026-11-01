import json, pytest
gr = pytest.importorskip("gradio")
from auditorynet.gradio_gui import App

def test_app(synthetic_data, tmp_path):
    out = tmp_path / "out"
    app = App(str(synthetic_data), str(out))
    df, msg = app.scan()
    assert "Ready to train" in msg and len(df) == 7
    demo = app.build()
    assert demo is not None
    # results with fake summary
    out.mkdir()
    (out / "summary.csv").write_text("task,n_hc,n_pd,mean_acc,std_acc,mean_auc,std_auc,sensitivity,specificity,train_acc,train_test_gap\nE,300,300,0.9,0.01,0.99,0.0,0.9,0.9,1.0,0.2\n")
    d, imgs, w = app.results_view()
    assert len(d) == 1 and "overfitting" in w
    # predict without models -> clean gradio error
    with pytest.raises(gr.Error):
        app.predict(None, "E", 0.5)
    with pytest.raises(gr.Error):
        app.predict("x.wav", "E", 0.5)
    # validation errors in train generator
    with pytest.raises(gr.Error):
        list(app.train([], 5, 16, 1e-3, .4, .05, 8, "", False))
    with pytest.raises(gr.Error):
        list(app.train(["E"], 5, 16, 1e-3, .4, .05, 8, "(", False))
    with pytest.raises(gr.Error):
        list(app.train(["U"], 5, 16, 1e-3, .4, .05, 8, "", False))  # missing task
    # real subprocess path (torch absent here -> should fail gracefully, not crash)
    outs = list(app.train(["E"], 2, 16, 1e-3, .4, .05, 8, "", False))
    assert outs and "TASK E" in outs[-1][0]
