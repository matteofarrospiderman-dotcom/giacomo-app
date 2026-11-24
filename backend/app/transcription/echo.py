"""Effects-aware transcription: remove delay/echo repeats from the note list.

A delay pedal repeats every note at a fixed lag, quieter each time. Transcription
models faithfully report those repeats as extra notes, which makes delay-heavy leads
(The Edge, Gilmour...) unreadable. We detect the delay *in the note domain*:

1. For every pair of same-pitch notes where the later one is clearly quieter (>= 1.5 dB
   by harmonic salience at that pitch), histogram their onset lag (120 ms - 1.2 s).
2. A sharp histogram peak supported by many pairs is the delay time ``T``. Ordinary
   repeated notes (riffs) are equally loud and do not contribute; we also require the
   peak not to be explained by the most common inter-onset interval of the part.
3. Notes that sit at ``T`` (or ``2T``) after a louder same-pitch note are rejected as
   echoes (kept in the rejected list, restorable in the editor).

The detected delay time is also handed to the tone analysis as independent evidence.
"""

from __future__ import annotations

import numpy as np
import librosa

from .base import NoteEvent

SR = 22050
HOP = 256


def _level_db(y22: np.ndarray) -> np.ndarray:
    rms = librosa.feature.rms(y=y22, frame_length=1024, hop_length=HOP)[0]
    return 20 * np.log10(rms + 1e-7)


def _onset_level(env: np.ndarray, t: float) -> float:
    f = int(t * SR / HOP)
    seg = env[f: f + 4]
    return float(seg.max()) if seg.size else -120.0


def _pitch_level(sal: dict | None, env: np.ndarray, n: NoteEvent) -> float:
    """Pitch-specific onset level in dB: harmonic salience at the note's pitch (robust to
    other instruments bleeding into the stem); broadband RMS if no salience is available."""
    if sal is not None:
        p = n.pitch - sal["fmin_midi"]
        if 0 <= p < sal["salience"].shape[0]:
            f = int(n.start * SR / HOP)
            seg = sal["salience"][p, f: f + 5]
            if seg.size:
                # the salience is already log-compressed, so it is used as a linear level and
                # compared by ratio (0.85 ~ a repeat 3 dB quieter); see _quieter()
                return float(seg.max())
    return float(10 ** (_onset_level(env, n.start) / 20))


def _quieter(later: float, earlier: float) -> bool:
    return later <= 0.85 * earlier


def find_delay(notes: list[NoteEvent], y22: np.ndarray, min_pairs: int = 5, sal: dict | None = None) -> dict | None:
    if len(notes) < 8:
        return None
    env = _level_db(y22)
    lv = {id(n): _pitch_level(sal, env, n) for n in notes}
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    lags = []
    for lst in by_pitch.values():
        lst.sort(key=lambda n: n.start)
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                d = b.start - a.start
                if d > 1.2:
                    break
                if d >= 0.12 and _quieter(lv[id(b)], lv[id(a)]):
                    lags.append(d)
    if len(lags) < min_pairs:
        return None
    bins = np.arange(0.12, 1.21, 0.01)
    hist, _ = np.histogram(lags, bins=bins)
    smooth = np.convolve(hist, [1, 2, 1], mode="same")
    k = int(np.argmax(smooth))
    support = int(hist[max(0, k - 1): k + 2].sum())
    lag = float(np.median([x for x in lags if abs(x - (bins[k] + 0.005)) <= 0.02]) if support else bins[k])
    # the most common inter-onset interval of the part (riff rhythm) must not explain it
    starts = np.sort([n.start for n in notes])
    ioi = np.diff(starts)
    ioi = ioi[(ioi > 0.08) & (ioi < 1.2)]
    common = float(np.median(ioi)) if ioi.size else 0.0
    share = support / max(1, len(notes))
    concentration = support / max(1, len(lags))  # echoes pile up at one lag; riff repeats spread out
    if support < min_pairs or share < 0.08 or concentration < 0.3:
        return None
    if common and any(abs(lag - k * common) < 0.015 for k in (1, 2, 3, 4)) and share < 0.4:
        return None  # lag explained by the riff's own rhythm
    return {"time_s": round(lag, 3), "pairs": support, "share": round(share, 3), "concentration": round(concentration, 3),
            "confidence": round(float(np.clip(0.3 + share + 0.4 * concentration, 0, 0.9)), 2)}


def suppress_echoes(notes: list[NoteEvent], y22: np.ndarray, sal: dict | None = None
                    ) -> tuple[list[NoteEvent], list[NoteEvent], dict | None]:
    info = find_delay(notes, y22, sal=sal)
    if not info:
        return notes, [], None
    env = _level_db(y22)
    lv = {id(n): _pitch_level(sal, env, n) for n in notes}
    T = info["time_s"]
    order = sorted(notes, key=lambda n: n.start)
    echoes: set[int] = set()
    for i, b in enumerate(order):
        for a in order[max(0, i - 60): i]:
            if a.pitch != b.pitch or id(a) in echoes:
                continue
            d = b.start - a.start
            for mult, tol in ((1, 0.025), (2, 0.035)):
                if abs(d - mult * T) <= tol and _quieter(lv[id(b)], lv[id(a)]):
                    echoes.add(id(b))
                    break
            if id(b) in echoes:
                break
    kept = [n for n in notes if id(n) not in echoes]
    removed = [n for n in notes if id(n) in echoes]
    for n in removed:
        n.flags.add("rejected:delay_echo")
    info["removed"] = len(removed)
    return kept, removed, info
