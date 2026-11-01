import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def synthetic_data(tmp_path):
    """Tiny fake dataset: PD = noisy/tremolo vowel, HC = clean vowel."""
    rng = np.random.default_rng(0)
    sr = 16000
    for task in ("E", "petaka"):
        for cls, label in (("PD", 1), ("Healthy", 0)):
            d = tmp_path / task / cls
            d.mkdir(parents=True)
            for i in range(12):
                t = np.arange(int(sr * rng.uniform(1.5, 4.0))) / sr
                f0 = rng.uniform(100, 220)
                w = np.sin(2 * np.pi * f0 * t)
                if label:
                    w = w * (1 + 0.4 * np.sin(2 * np.pi * 5 * t)) + 0.15 * rng.standard_normal(len(t))
                sf.write(d / f"spk{label}{i}_{task}.wav", (0.5 * w).astype(np.float32), sr)
    return tmp_path
