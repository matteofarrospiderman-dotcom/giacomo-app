"""Known effects applied to synthetic guitar for detector calibration tests."""

import numpy as np
from scipy.signal import fftconvolve


def distort(x: np.ndarray, drive: float = 20.0) -> np.ndarray:
    y = np.tanh(drive * x)
    # cabinet-like low-pass (2nd-order ~5 kHz) so it resembles a mic'd amp
    from scipy.signal import butter, lfilter
    b, a = butter(2, 5000 / 22050)
    return (lfilter(b, a, y, axis=-1) * 0.5).astype(np.float32)


def reverb(x: np.ndarray, rt60: float = 1.5, mix: float = 0.35, sr: int = 44100, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    out = []
    for ch in range(x.shape[0]):
        ir = rng.normal(0, 1, n) * 10 ** (-3 * t / rt60)
        ir /= np.sqrt(np.sum(ir**2))
        wet = fftconvolve(x[ch], ir)[: x.shape[1]]
        out.append((1 - mix) * x[ch] + mix * wet * 3)
    return np.stack(out).astype(np.float32)


def delay(x: np.ndarray, ms: float = 375, feedback: float = 0.45, mix: float = 0.4, sr: int = 44100) -> np.ndarray:
    d = int(ms * sr / 1000)
    y = x.copy()
    wet = np.zeros_like(x)
    g = mix
    for k in range(1, 8):
        if k * d >= x.shape[1]:
            break
        wet[:, k * d:] += g * x[:, : x.shape[1] - k * d]
        g *= feedback
    return (y + wet).astype(np.float32)


def tremolo(x: np.ndarray, rate: float = 5.0, depth: float = 0.6, sr: int = 44100) -> np.ndarray:
    t = np.arange(x.shape[1]) / sr
    return (x * (1 - depth / 2 + depth / 2 * np.sin(2 * np.pi * rate * t))).astype(np.float32)
