"""Multi-guitar separation by stereo placement.

No open neural model separates *two guitars from each other*. In practice most
commercial productions place multiple guitar parts at different stereo positions
(double-tracked rhythm hard-left/hard-right, lead centre, acoustic panned...). We exploit
that with azimuth discrimination (in the spirit of ADRess, Barry et al. 2004):

1. For each time-frequency bin of the guitar stem compute a pan index
   ``p = (|R| - |L|) / (|R| + |L|)`` in [-1, 1].
2. Build an energy-weighted pan histogram and find its modes.
3. If there are >= 2 well separated modes that each hold a meaningful share of the
   energy, create one soft mask per mode (Gaussian in pan space) and resynthesise one
   stem per guitar position.
4. Check whether the resulting parts actually play *different* music (chroma
   correlation). Highly correlated L/R parts are reported as a double-tracked part so we
   transcribe it once instead of producing two identical tabs.

When guitars share the same pan position this cannot separate them; the note-level
lead/rhythm splitter in :mod:`app.transcription.parts` is the second line of defence.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import gaussian_filter1d

N_FFT = 4096
HOP = 1024


def pan_profile(stereo: np.ndarray, sr: int) -> dict:
    L = np.abs(librosa.stft(stereo[0], n_fft=N_FFT, hop_length=HOP))
    R = np.abs(librosa.stft(stereo[1], n_fft=N_FFT, hop_length=HOP))
    E = L**2 + R**2
    p = (R - L) / (R + L + 1e-10)
    thr = np.percentile(E, 90) * 0.05
    sel = E > thr
    hist, edges = np.histogram(p[sel], bins=81, range=(-1, 1), weights=E[sel])
    hist = gaussian_filter1d(hist.astype(float), 1.5)
    hist = hist / (hist.sum() + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    side_ratio = float(np.sum(((stereo[0] - stereo[1]) / 2) ** 2) / (np.sum(((stereo[0] + stereo[1]) / 2) ** 2) + 1e-12))
    return {"hist": hist, "centers": centers, "L": L, "R": R, "p": p, "E": E, "side_ratio": side_ratio}


def find_pan_modes(hist: np.ndarray, centers: np.ndarray, min_share: float = 0.12, min_sep: float = 0.45) -> list[dict]:
    peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]]
    # include edges (hard-panned content piles up at +-1)
    if hist[0] > hist[1]:
        peaks.insert(0, 0)
    if hist[-1] > hist[-2]:
        peaks.append(len(hist) - 1)
    peaks.sort(key=lambda i: -hist[i])
    modes: list[dict] = []
    for i in peaks:
        c = centers[i]
        if all(abs(c - m["pan"]) >= min_sep for m in modes):
            share = float(hist[np.abs(centers - c) < min_sep / 2].sum())
            if share >= min_share:
                modes.append({"pan": float(c), "share": share})
    modes.sort(key=lambda m: m["pan"])
    return modes


def split_guitars(stereo: np.ndarray, sr: int) -> dict:
    """Return {'parts': [{id, label, pan, share, audio}], 'double_tracked': bool, 'analysis': {...}}."""
    if stereo.shape[0] < 2 or np.allclose(stereo[0], stereo[1], atol=1e-4):
        return {"parts": [], "double_tracked": False, "analysis": {"reason": "mono source"}}
    prof = pan_profile(stereo, sr)
    modes = find_pan_modes(prof["hist"], prof["centers"])
    analysis = {
        "pan_histogram": [round(float(v), 4) for v in prof["hist"]],
        "modes": modes,
        "side_ratio": round(prof["side_ratio"], 3),
    }
    if len(modes) < 2:
        analysis["reason"] = "single stereo position"
        return {"parts": [], "double_tracked": False, "analysis": analysis}

    X = [librosa.stft(stereo[c], n_fft=N_FFT, hop_length=HOP) for c in range(2)]
    p = prof["p"]
    sigma = 0.22
    weights = np.stack([np.exp(-0.5 * ((p - m["pan"]) / sigma) ** 2) for m in modes])
    weights /= weights.sum(axis=0, keepdims=True) + 1e-10
    parts = []
    chromas = []
    for k, m in enumerate(modes):
        chans = [librosa.istft(X[c] * weights[k], hop_length=HOP, length=stereo.shape[-1]) for c in range(2)]
        audio = np.stack(chans).astype(np.float32)
        if len(modes) == 2 and modes[1]["pan"] - modes[0]["pan"] >= 0.4:
            side = "left" if k == 0 else "right"  # relative placement of the two parts
        else:
            side = "left" if m["pan"] < -0.25 else "right" if m["pan"] > 0.25 else "center"
        parts.append({"id": f"guitar_{side}", "label": f"Guitar ({side})", "pan": m["pan"], "share": m["share"],
                      "audio": audio})
        mono = audio.mean(axis=0)
        chromas.append(librosa.feature.chroma_stft(y=mono, sr=sr, hop_length=2048))
    # de-duplicate ids (two modes on the same side)
    seen: dict[str, int] = {}
    for part in parts:
        n = seen.get(part["id"], 0)
        seen[part["id"]] = n + 1
        if n:
            part["id"] += f"_{n + 1}"
    # similarity between parts: are they the same riff double-tracked?
    sims = []
    for i in range(len(chromas)):
        for j in range(i + 1, len(chromas)):
            a, b = chromas[i], chromas[j]
            n = min(a.shape[1], b.shape[1])
            a, b = a[:, :n], b[:, :n]
            act = (a.sum(0) > 0.5) & (b.sum(0) > 0.5)
            if act.sum() < 10:
                sims.append(0.0)
                continue
            an = a[:, act] / (np.linalg.norm(a[:, act], axis=0) + 1e-9)
            bn = b[:, act] / (np.linalg.norm(b[:, act], axis=0) + 1e-9)
            sims.append(float(np.mean(np.sum(an * bn, axis=0))))
    analysis["part_similarity"] = [round(s, 3) for s in sims]
    double = bool(len(parts) == 2 and sims and sims[0] > 0.9)
    return {"parts": parts, "double_tracked": double, "analysis": analysis}
