"""Synthetic test material with exact ground truth.

Karplus-Strong plucked strings sound close enough to a guitar for transcription models,
and because we generate them we know every pitch, onset, string and fret.
"""

from __future__ import annotations

import numpy as np


def karplus_strong(freq: float, dur: float, sr: int = 44100, decay: float = 0.996, bright: float = 0.5,
                   seed: int = 0, bend_to: float | None = None, vibrato: tuple[float, float] | None = None) -> np.ndarray:
    """Plucked string. ``bend_to`` = target freq reached at 40% of the note (bend);
    ``vibrato`` = (rate_hz, depth_semitones). Pitch modulation uses a fractional delay line."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    period0 = sr / freq
    buf_len = int(period0 * 1.3) + 4
    buf = (rng.uniform(-1, 1, buf_len)).astype(np.float64)
    # lowpass the excitation for a softer pick
    for i in range(1, buf_len):
        buf[i] = bright * buf[i] + (1 - bright) * buf[i - 1]
    out = np.zeros(n)
    line = np.zeros(int(sr / 30) + 8)
    L = len(line)
    p0 = int(period0)
    line[L - p0:] = buf[:p0]  # excitation sits in the d samples 'behind' the write head
    w = 0
    t = np.arange(n) / sr
    f_inst = np.full(n, freq)
    if bend_to:
        ramp = np.clip(t / (0.4 * dur), 0, 1)
        f_inst = freq * (bend_to / freq) ** ramp
    if vibrato:
        rate, depth = vibrato
        f_inst = f_inst * 2 ** (depth / 12 * np.sin(2 * np.pi * rate * t) * np.clip(t / 0.15, 0, 1))
    prev = 0.0
    for i in range(n):
        d = sr / f_inst[i]
        r = w - d
        ri = int(np.floor(r))
        frac = r - ri
        a = line[ri % L]
        b = line[(ri + 1) % L]
        s = a * (1 - frac) + b * frac
        y = decay * 0.5 * (s + prev)
        prev = s
        line[w % L] = y
        out[i] = s
        w += 1
    env = np.minimum(1.0, t / 0.002)
    return (out * env).astype(np.float32)


STANDARD = [40, 45, 50, 55, 59, 64]


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def render(notes: list[dict], dur: float, sr: int = 44100, pan: float = 0.0) -> np.ndarray:
    """notes: [{'pitch', 'start', 'end', 'vel', optional 'bend_to', 'vibrato'}] -> stereo array."""
    y = np.zeros(int(dur * sr) + sr, np.float32)
    for k, n in enumerate(notes):
        length = n["end"] - n["start"] + 0.25
        bend = midi_hz(n["bend_to"]) if n.get("bend_to") else None
        s = karplus_strong(midi_hz(n["pitch"]), length, sr, seed=k, bend_to=bend, vibrato=n.get("vibrato"))
        # damp after note end (player releases)
        rel = int((n["end"] - n["start"]) * sr)
        if rel < len(s):
            s[rel:] *= np.exp(-np.arange(len(s) - rel) / (0.02 * sr))
        i0 = int(n["start"] * sr)
        if i0 >= len(y):
            continue
        m = min(len(s), len(y) - i0)
        y[i0: i0 + m] += n.get("vel", 0.8) * s[:m]
    y = y[: int(dur * sr)]
    g_l = np.sqrt(0.5 * (1 - pan))
    g_r = np.sqrt(0.5 * (1 + pan))
    return np.stack([y * g_l, y * g_r]).astype(np.float32)


def click_track(bpm: float, dur: float, sr: int = 44100, beats_per_bar: int = 4) -> np.ndarray:
    y = np.zeros(int(dur * sr), np.float32)
    period = 60.0 / bpm
    rng = np.random.default_rng(1)
    t = 0.0
    k = 0
    while t < dur:
        i0 = int(t * sr)
        L = int(0.05 * sr)
        burst = rng.normal(0, 1, L) * np.exp(-np.arange(L) / (0.008 * sr))
        if k % beats_per_bar == 0:  # kick-like low thump on downbeats
            tt = np.arange(L) / sr
            burst = burst * 0.3 + 1.5 * np.sin(2 * np.pi * 60 * tt) * np.exp(-tt / 0.03)
        y[i0: i0 + L] += 0.35 * burst[: len(y) - i0]
        t += period
        k += 1
    return np.stack([y, y])


def riff_ground_truth(bpm: float = 100.0, bars: int = 4) -> list[dict]:
    """A playable E-minor pentatonic riff in standard tuning, 8th notes, with known fingering."""
    pattern = [  # (string, fret)  string 1 = high e
        (6, 0), (6, 3), (5, 0), (5, 2), (4, 0), (4, 2), (5, 2), (5, 0),
        (6, 0), (6, 3), (5, 0), (5, 2), (4, 2), (3, 0), (3, 2), (4, 2),
    ]
    eighth = 60.0 / bpm / 2
    notes = []
    t = 0.5
    for b in range(bars):
        for k in range(8):
            s, f = pattern[(b * 8 + k) % len(pattern)]
            pitch = STANDARD[6 - s] + f
            notes.append({"pitch": pitch, "start": t, "end": t + eighth * 0.95, "string": s, "fret": f, "vel": 0.8})
            t += eighth
    return notes
