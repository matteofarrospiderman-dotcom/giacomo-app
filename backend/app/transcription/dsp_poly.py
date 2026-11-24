"""Model-free polyphonic pitch salience and note tracking.

An independent "second opinion" for the ensemble (and a full fallback when the Basic
Pitch model is missing):

1. Constant-Q transform, 3 bins/semitone, hop 256 @ 22.05 kHz (frame-aligned with
   Basic Pitch).
2. Log compression + spectral whitening (subtract a running median across frequency) so
   timbre/EQ matter less.
3. Harmonic summation salience (Klapuri 2006 style):
   ``S(p, t) = sum_h w_h * max_{+-1 bin} X(bin(p) + 36*log2(h), t)`` with w_h = 0.84^(h-1).
4. Sub-harmonic suppression: salience at p is reduced when p-12 explains its energy.
5. Notes: hysteresis thresholding per pitch + splitting at per-band onsets.

It is weaker than a trained model on dense mixes but has different failure modes, which
is exactly what makes it useful for cross-validation.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.ndimage import median_filter, uniform_filter1d

from .base import NoteEvent

SR = 22050
HOP = 256
BPO = 36
FMIN_MIDI = 28  # E1
N_PITCH = 72  # E1..D#7
N_HARM = 8


def salience(y22: np.ndarray) -> dict:
    fmin = librosa.midi_to_hz(FMIN_MIDI) * 2 ** (-1 / 36)  # centre bin on the semitone
    n_bins = (N_PITCH + 12 * 3 + 6) * 3  # room for harmonics up to h=8 (36 semitones)
    max_bins = int(np.floor(BPO * np.log2((SR / 2 * 0.95) / fmin)))
    n_bins = min(n_bins, max_bins)
    C = np.abs(librosa.cqt(y22, sr=SR, hop_length=HOP, fmin=fmin, n_bins=n_bins, bins_per_octave=BPO))
    L = np.log1p(100 * C / (C.max() + 1e-9))
    W = np.maximum(L - median_filter(L, size=(BPO // 2 + 1, 1)), 0)  # whitening
    T = W.shape[1]
    S = np.zeros((N_PITCH, T), np.float32)
    Wmax = np.maximum(np.maximum(W, np.roll(W, 1, axis=0)), np.roll(W, -1, axis=0))
    for p in range(N_PITCH):
        b0 = p * 3 + 1
        acc = np.zeros(T)
        for h in range(1, N_HARM + 1):
            b = b0 + int(round(BPO * np.log2(h)))
            if b >= Wmax.shape[0]:
                break
            acc += (0.84 ** (h - 1)) * Wmax[b]
        S[p] = acc
    # sub-harmonic suppression: energy at p that is just the 2nd harmonic of p-12
    S2 = S.copy()
    S2[12:] -= 0.55 * S[:-12]
    S2 = np.maximum(S2, 0)
    frame_max = S2.max(axis=0, keepdims=True) + 1e-9
    rel = S2 / frame_max
    glob = S2 / (np.percentile(S2, 99.5) + 1e-9)
    return {"salience": S2, "relative": rel, "global": np.clip(glob, 0, 1.5), "fmin_midi": FMIN_MIDI}


def support(sal: dict, pitch: int, t0: float, t1: float) -> float:
    """Mean relative salience of ``pitch`` within [t0, t1] (0..1)."""
    p = pitch - sal["fmin_midi"]
    if p < 0 or p >= sal["relative"].shape[0]:
        return 0.0
    f0 = int(t0 * SR / HOP)
    f1 = max(f0 + 1, int(t1 * SR / HOP))
    seg = sal["relative"][p, f0:f1]
    g = sal["global"][p, f0:f1]
    if seg.size == 0:
        return 0.0
    return float(np.clip(0.6 * np.mean(seg) + 0.4 * np.mean(np.minimum(g, 1)), 0, 1))


def transcribe(sal: dict, on_thr: float = 0.55, off_thr: float = 0.35, min_dur: float = 0.07) -> list[NoteEvent]:
    rel = sal["relative"] * np.minimum(1.0, sal["global"] * 2.5)
    rel = uniform_filter1d(rel, 3, axis=1)
    P, T = rel.shape
    # local maxima across pitch only (one note per spectral peak)
    peak = (rel >= np.roll(rel, 1, axis=0)) & (rel >= np.roll(rel, -1, axis=0))
    act = rel * peak
    flux = np.zeros_like(rel)
    flux[:, 1:] = np.maximum(rel[:, 1:] - rel[:, :-1], 0)
    notes = []
    min_len = int(min_dur * SR / HOP)
    for p in range(P):
        row = act[p]
        t = 0
        while t < T:
            if row[t] >= on_thr:
                s = t
                while t < T and rel[p, t] >= off_thr:
                    t += 1
                e = t
                # split at strong re-attacks inside the run
                cuts = [s]
                for k in range(s + min_len, e - min_len):
                    if flux[p, k] > 0.25 and flux[p, k] == flux[p, max(s, k - 3): k + 4].max():
                        cuts.append(k)
                cuts.append(e)
                for a, b in zip(cuts[:-1], cuts[1:]):
                    if b - a >= min_len:
                        amp = float(np.mean(rel[p, a:b]))
                        notes.append(NoteEvent(start=a * HOP / SR, end=b * HOP / SR, pitch=p + FMIN_MIDI,
                                               amplitude=amp, sources={"dsp": amp}))
            else:
                t += 1
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
