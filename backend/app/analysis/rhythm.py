"""Tempo, beat, downbeat and time-signature estimation.

Algorithm
---------
1. Onset strength envelope (spectral flux, median-aggregated across mel bands). When a
   drum stem is available it is used, because kick/snare transients give the cleanest
   beat evidence; otherwise the full mix is used.
2. Tempo candidates come from the autocorrelation tempogram. Octave errors (the classic
   60 vs 120 BPM confusion) are resolved with a log-normal prior centred on ~115 BPM and
   with the strength of the half/double tempo peaks.
3. Beats: dynamic-programming beat tracker (Ellis 2007, ``librosa.beat.beat_track``)
   constrained to the chosen tempo.
4. Meter + downbeat: every (beats-per-bar, phase) hypothesis is scored with three
   beat-synchronous cues: low-frequency onset accent (kick drums land on downbeats),
   harmonic change (chords usually change on bar lines), and overall accent.
5. Compound meters (6/8, 12/8) are detected from the ratio of onset energy on
   triplet vs. duple subdivisions of the beat.

Limitations: rubato/free-time performances, songs with tempo changes larger than the
beat tracker's tightness allows, and odd meters (5/4, 7/8) are only partially handled
(odd meters are evaluated but have a low prior).
"""

from __future__ import annotations

import numpy as np
import librosa

HOP = 512


def _onset_env(y: np.ndarray, sr: int, fmax: float | None = None) -> np.ndarray:
    if fmax is not None and fmax < 1000:
        # low-band flux straight from STFT bins (a mel bank this narrow would have empty filters)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        L = np.log1p(100 * S[(freqs >= 30) & (freqs <= fmax)])
        flux = np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0).sum(axis=0)
        return flux.astype(np.float32)
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=96, fmax=fmax or sr / 2)
    return librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=sr, hop_length=HOP, aggregate=np.median)


def _tempo_candidates(oenv: np.ndarray, sr: int) -> list[tuple[float, float]]:
    tg = librosa.autocorrelate(oenv - oenv.mean(), max_size=int(4.0 * sr / HOP))
    tg = np.maximum(tg, 0)
    if tg.max() <= 0:
        return [(120.0, 0.0)]
    tg = tg / tg.max()
    lags = np.arange(len(tg))
    bpms = np.zeros_like(tg, dtype=float)
    bpms[1:] = 60.0 * sr / (HOP * lags[1:])
    valid = (bpms >= 40) & (bpms <= 240)
    cands = []
    for i in range(2, len(tg) - 1):
        if valid[i] and tg[i] > tg[i - 1] and tg[i] >= tg[i + 1] and tg[i] > 0.05:
            # parabolic interpolation of the lag for sub-frame precision
            a, b, c = tg[i - 1], tg[i], tg[i + 1]
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
            lag = i + shift
            cands.append((60.0 * sr / (HOP * lag), float(b)))
    cands.sort(key=lambda x: -x[1])
    return cands[:8] or [(120.0, 0.0)]


def _choose_tempo(cands: list[tuple[float, float]]) -> tuple[float, float, list[dict]]:
    def prior(bpm: float) -> float:
        return float(np.exp(-0.5 * (np.log2(bpm / 115.0) / 0.55) ** 2))

    def support(bpm: float) -> float:
        # strength of harmonically related peaks reinforces a candidate
        s = 0.0
        for b, w in cands:
            for ratio in (0.5, 2.0, 1.5, 1 / 1.5, 3.0, 1 / 3.0):
                if abs(np.log2(b / (bpm * ratio))) < 0.03:
                    s += 0.25 * w
        return s

    scored = []
    for bpm, w in cands:
        score = (w + support(bpm)) * prior(bpm)
        scored.append({"bpm": round(bpm, 2), "strength": round(w, 3), "score": round(score, 4)})
    scored.sort(key=lambda d: -d["score"])
    best = scored[0]
    second = scored[1]["score"] if len(scored) > 1 else 0.0
    conf = float(np.clip((best["score"] - second) / (best["score"] + 1e-9) * 1.5 + 0.3 * best["strength"], 0, 1))
    return float(best["bpm"]), conf, scored


def _beat_sync(feature: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    if len(beat_frames) < 2:
        return np.zeros(len(beat_frames))
    vals = []
    for i, f in enumerate(beat_frames):
        lo = max(0, f - 3)
        hi = min(len(feature), f + 3)
        vals.append(feature[lo:hi].max() if hi > lo else 0.0)
    v = np.asarray(vals, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


def _harmonic_change(y: np.ndarray, sr: int, beat_frames: np.ndarray) -> tuple[np.ndarray, float]:
    """Chord-change strength at each beat, and how harmonic the signal is (0..1).

    For purely percussive input chroma is flat noise, so the cue is down-weighted by the
    median chroma peakiness (max/mean per frame): ~1.3 for noise, ~1.9+ for music.
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    if chroma.shape[1] < 3 or len(beat_frames) < 3:
        return np.zeros(len(beat_frames)), 0.0
    peaky = float(np.median(chroma.max(axis=0) / (chroma.mean(axis=0) + 1e-9)))
    harmonicity = float(np.clip((peaky - 1.35) / 0.5, 0, 1))
    sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
    if len(beat_frames) and beat_frames[0] == 0:
        # sync() omits the empty span before a beat at frame 0; keep columns aligned (i, i+1)
        sync = np.hstack([sync[:, :1], sync])
    sync = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)
    # sync() returns len(beats)+1 segments: column i is the span *ending* at beat i, column
    # i+1 the span *starting* at beat i, so the change at beat i compares columns i and i+1
    change = np.zeros(len(beat_frames))
    for i in range(min(len(beat_frames), sync.shape[1] - 1)):
        change[i] = 1.0 - float(np.dot(sync[:, i], sync[:, i + 1]))
    return (change - change.mean()) / (change.std() + 1e-9), harmonicity


def _subdivision_ratio(oenv: np.ndarray, beat_frames: np.ndarray) -> float:
    """>1 means triplet subdivisions dominate (compound/swing feel)."""
    trip, dup = [], []
    for a, b in zip(beat_frames[:-1], beat_frames[1:]):
        L = b - a
        if L < 6:
            continue
        def at(frac: float) -> float:
            idx = int(round(a + frac * L))
            return float(oenv[max(0, idx - 1): idx + 2].max()) if idx < len(oenv) else 0.0
        trip.append(at(1 / 3) + at(2 / 3))
        dup.append(2 * at(1 / 2))
    if not trip:
        return 1.0
    return float((np.mean(trip) + 1e-9) / (np.mean(dup) + 1e-9))


def analyze_rhythm(y_mix: np.ndarray, sr: int, y_drums: np.ndarray | None = None) -> dict:
    y_beat = y_drums if y_drums is not None and np.sqrt(np.mean(y_drums**2)) > 1e-3 else y_mix
    oenv = _onset_env(y_beat, sr)
    cands = _tempo_candidates(oenv, sr)
    bpm, tempo_conf, scored = _choose_tempo(cands)

    _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP, bpm=bpm, tightness=120,
                                             trim=False, units="frames")
    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    duration = len(y_mix) / sr

    # Refine the global BPM from the beat grid. Individual beat times are quantised to the
    # onset-envelope hop (~23 ms), so a median of intervals is biased; a robust linear fit of
    # beat time against beat index gives sub-frame precision.
    if len(beat_times) > 4:
        ibi = np.diff(beat_times)
        idx = np.arange(len(beat_times))
        slope = float(np.polyfit(idx, beat_times, 1)[0])
        resid = beat_times - np.polyval(np.polyfit(idx, beat_times, 1), idx)
        good = np.abs(resid) < 3 * (np.std(resid) + 1e-9)
        if good.sum() > 4:
            slope = float(np.polyfit(idx[good], beat_times[good], 1)[0])
        bpm = float(60.0 / slope) if slope > 0 else float(60.0 / np.median(ibi))
        tempo_stability = float(1.0 - np.clip(np.std(ibi) / (np.mean(ibi) + 1e-9) * 4, 0, 1))
    else:
        tempo_stability = 0.0

    # ---- meter & downbeats
    low_env = _onset_env(y_beat, sr, fmax=200.0)
    accent = _beat_sync(oenv, beat_frames)
    low_accent = _beat_sync(low_env, beat_frames)
    harm, harmonicity = _harmonic_change(y_mix, sr, beat_frames)
    meter_priors = {4: 1.0, 3: 0.55, 2: 0.35, 6: 0.15, 5: 0.06, 7: 0.05}
    hyps = []
    n = len(beat_frames)
    for m, pri in meter_priors.items():
        if n < 2 * m:
            continue
        for p in range(m):
            idx = np.arange(p, n, m)
            mask = np.zeros(n, bool)
            mask[idx] = True
            s = 0.45 * (low_accent[mask].mean() - low_accent[~mask].mean())
            s += 0.35 * harmonicity * (harm[mask].mean() - harm[~mask].mean())
            s += 0.20 * (accent[mask].mean() - accent[~mask].mean())
            hyps.append({"beats_per_bar": m, "phase": p, "score": float(s), "posterior": float(s + 0.35 * np.log(pri))})
    if hyps:
        hyps.sort(key=lambda h: -h["posterior"])
        best = hyps[0]
        others = [h for h in hyps if h["beats_per_bar"] != best["beats_per_bar"]]
        margin = best["posterior"] - (others[0]["posterior"] if others else best["posterior"] - 0.5)
        meter_conf = float(np.clip(0.35 + margin, 0.05, 0.95))
        m, phase = best["beats_per_bar"], best["phase"]
    else:
        m, phase, meter_conf = 4, 0, 0.1

    sub_ratio = _subdivision_ratio(oenv, beat_frames)
    compound = sub_ratio > 1.35
    if compound and m in (2, 4):
        numerator, denominator = (6, 8) if m == 2 else (12, 8)
        feel = "compound"
    else:
        numerator, denominator = m, 4
        feel = "swing/triplet" if sub_ratio > 1.15 else "straight"

    downbeats = beat_times[phase::m] if len(beat_times) else np.array([])

    # local tempo curve (BPM per beat) for display and tempo-map MIDI export
    tempo_curve = []
    if len(beat_times) > 2:
        ibi = np.diff(beat_times)
        smooth = np.convolve(np.pad(ibi, 2, mode="edge"), np.ones(5) / 5, mode="valid") if len(ibi) >= 5 else ibi
        tempo_curve = [{"t": float(t), "bpm": float(60 / d)} for t, d in zip(beat_times[:-1], smooth)]

    return {
        "bpm": round(bpm, 2),
        "tempo_confidence": round(tempo_conf * (0.5 + 0.5 * tempo_stability), 3),
        "tempo_stability": round(tempo_stability, 3),
        "tempo_candidates": scored[:5],
        "beats": [round(float(t), 4) for t in beat_times],
        "downbeats": [round(float(t), 4) for t in downbeats],
        "beats_per_bar": m,
        "time_signature": [numerator, denominator],
        "meter_confidence": round(meter_conf, 3),
        "meter_hypotheses": hyps[:6],
        "feel": feel,
        "subdivision_ratio": round(sub_ratio, 3),
        "tempo_curve": tempo_curve[:: max(1, len(tempo_curve) // 400)],
        "duration": duration,
        "source": "drums stem" if y_beat is y_drums else "full mix",
    }


def build_beat_grid(beats: list[float], duration: float, bpm: float) -> np.ndarray:
    """Extend tracked beats to cover [0, duration] by extrapolating at the edge tempo."""
    b = np.asarray(beats, dtype=float)
    period = 60.0 / max(bpm, 1.0)
    if len(b) < 2:
        return np.arange(0.0, duration + period, period)
    head = []
    t = b[0] - (b[1] - b[0])
    while t > -period:
        head.append(t)
        t -= (b[1] - b[0])
    tail = []
    t = b[-1] + (b[-1] - b[-2])
    while t < duration + 2 * period:
        tail.append(t)
        t += (b[-1] - b[-2])
    return np.concatenate([np.array(head[::-1]), b, np.array(tail)])
