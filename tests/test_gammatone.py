import numpy as np
import torch

from auditorynet.gammatone import Cochleagram, build_filterbank, centre_frequencies


def test_centre_frequencies_span_and_monotonic():
    fc = centre_frequencies(64, 80.0, 8000.0)
    assert fc.shape == (64,)
    assert abs(fc[0] - 80.0) < 1e-6 and abs(fc[-1] - 8000.0) < 1e-3
    assert np.all(np.diff(fc) > 0)


def test_filterbank_shape_and_peak_norm():
    fb = build_filterbank(16000, 64, 80.0, 8000.0, 400)
    assert fb.shape == (64, 400)
    np.testing.assert_allclose(np.abs(fb).max(axis=1), 1.0, atol=1e-5)


def test_cochleagram_shape_range_and_selectivity():
    sr, n = 16000, 48000
    t = torch.arange(n) / sr
    wave = torch.sin(2 * torch.pi * 200 * t)[None]
    coch = Cochleagram()
    c = coch(wave)
    assert c.shape == (1, 1, 64, 128)
    assert float(c.min()) >= 0 and float(c.max()) <= 1 + 1e-6
    fc = centre_frequencies(64, 80.0, 8000.0)
    peak_ch = int(c[0, 0].mean(-1).argmax())
    assert fc[peak_ch] < 800  # low-frequency tone excites low-frequency channels
