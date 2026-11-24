"""Guitar tone & effects analysis from a (separated) guitar stem.

Each detector measures interpretable evidence and maps it to a score. Scores become one
of four confidence tiers, and we never claim a specific pedal/amp model: we describe
the *character* of the processing and estimate parameter ranges.

========================  ============================================================
Effect                    Evidence used
========================  ============================================================
Gain / distortion         crest factor, upper-harmonic energy ratio, mid-band spectral
                          flatness, note sustain (decay rate in dB/s)
Fuzz (vs distortion)      extreme gain + flat broadband mids + weak cabinet-like top
Compression               loudness range of note attacks, attack-to-sustain ratio
Reverb                    Schroeder decay of tails after phrase ends (RT60), L/R tail
                          decorrelation, tail-to-direct energy (wet/dry)
Delay / echo              regularly spaced, geometrically decaying energy bumps after
                          phrase ends + envelope autocorrelation; feedback from decay
Tremolo                   periodic 2-12 Hz amplitude modulation on sustained notes
Chorus / vibrato unit     slow periodic modulation of partial amplitudes / pitch common
                          to *all* sustained notes, stereo decorrelation
Flanger / phaser          periodically sweeping comb (cepstral peak 0.5-12 ms) /
                          sweeping spectral notches
Wah                       a strongly moving resonant peak 400-2500 Hz
EQ / cabinet              long-term spectrum vs a neutral guitar reference, roll-off
Noise gate                abrupt drops to a very low floor in gaps of a high-gain part
Octave / pitch shift      sub-octave energy under monophonic notes; constant parallel
                          intervals in the transcription (harmoniser)
Stereo width              mid/side ratio, L/R correlation, pan position
========================  ============================================================

Known limitations: separation artefacts add their own "reverb" and "flanging";
effects baked in before mastering compression are partly masked; a stereo-doubled
part can look like chorus. Hence the conservative tiers.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import find_peaks

TIERS = [(0.75, "high"), (0.5, "probable"), (0.3, "possible")]


def tier(score: float) -> str:
    for thr, name in TIERS:
        if score >= thr:
            return name
    return "uncertain"


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))


def _band_levels(ltas_db: np.ndarray, freqs: np.ndarray) -> dict:
    bands = {"low": (80, 250), "low_mid": (250, 800), "mid": (800, 2000), "upper_mid": (2000, 4000),
             "presence": (4000, 6500), "air": (6500, 12000)}
    out = {}
    for k, (a, b) in bands.items():
        m = (freqs >= a) & (freqs < b)
        out[k] = float(np.mean(ltas_db[m])) if m.any() else -120.0
    return out


# typical relative band levels of a moderately driven guitar through a cabinet (dB, mid = 0)
REFERENCE = {"low": -2.0, "low_mid": 2.0, "mid": 0.0, "upper_mid": -4.0, "presence": -12.0, "air": -26.0}


def _gaps(notes: list[dict], duration: float, min_gap: float = 0.45) -> list[tuple[float, float]]:
    """Silences after phrase ends (no note sounding/starting), used for tails."""
    if not notes:
        return []
    ev = sorted((n["start"], n["end"]) for n in notes)
    gaps = []
    cur_end = ev[0][1]
    for s, e in ev[1:]:
        if s - cur_end >= min_gap:
            gaps.append((cur_end, s))
        cur_end = max(cur_end, e)
    if duration - cur_end >= min_gap:
        gaps.append((cur_end, duration))
    return gaps


def analyze_distortion(mono: np.ndarray, sr: int, notes: list[dict], S: np.ndarray, freqs: np.ndarray, hop: int) -> dict:
    peak = np.max(np.abs(mono)) + 1e-9
    rms = np.sqrt(np.mean(mono**2)) + 1e-9
    crest = float(_db(peak / rms))
    # harmonic ratio on confident sustained notes: energy above the 4th harmonic vs below
    ratios = []
    for n in notes:
        if n["end"] - n["start"] < 0.2 or n.get("confidence", 0) < 0.5:
            continue
        f0 = 440 * 2 ** ((n["pitch"] - 69) / 12)
        a = int((n["start"] + 0.03) * sr / hop)
        b = int(min(n["end"], n["start"] + 0.4) * sr / hop)
        if b <= a or b > S.shape[1]:
            continue
        spec = S[:, a:b].mean(axis=1)
        low = spec[(freqs > 0.8 * f0) & (freqs < 3.5 * f0)].sum()
        high = spec[(freqs >= 3.5 * f0) & (freqs < min(16 * f0, 8000))].sum()
        if low > 0:
            ratios.append(high / low)
        if len(ratios) > 200:
            break
    hr = float(np.median(ratios)) if ratios else 0.3
    mid = (freqs > 1000) & (freqs < 5000)
    flat = float(np.median(librosa.feature.spectral_flatness(S=S[mid] + 1e-9)))
    # sustain: decay rate of isolated long notes
    decays = []
    env = _db(np.sqrt(np.mean(S**2, axis=0)))
    for n in notes:
        if n["end"] - n["start"] < 0.6:
            continue
        a = int((n["start"] + 0.05) * sr / hop)
        b = int((n["end"] - 0.02) * sr / hop)
        if b - a < 6 or b > len(env):
            continue
        seg = env[a:b]
        t = np.arange(len(seg)) * hop / sr
        slope = np.polyfit(t, seg, 1)[0]
        decays.append(-slope)
    decay = float(np.median(decays)) if decays else 12.0
    # map evidence to a 0..10 gain index; each indicator also votes independently
    votes = np.array([
        np.clip((18 - crest) / 10, 0, 1),
        np.clip((hr - 0.25) / 1.2, 0, 1),
        np.clip((flat - 0.01) / 0.08, 0, 1),
        np.clip((14 - decay) / 12, 0, 1),
    ])
    gain = float(np.clip(votes @ np.array([3.0, 3.5, 1.5, 2.0]), 0, 10))
    if gain < 2:
        label = "Clean"
    elif gain < 3.8:
        label = "Edge of breakup / light crunch"
    elif gain < 5.5:
        label = "Overdrive / crunch"
    elif gain < 7.5:
        label = "Distortion"
    else:
        label = "High-gain distortion"
    evidence_n = len(ratios) + len(decays)
    agreement = 1.0 - float(np.std(votes)) * 1.6  # indicators that disagree -> lower confidence
    conf = float(np.clip((0.3 + 0.04 * min(evidence_n, 10)) * np.clip(agreement, 0.3, 1.0), 0.2, 0.85))
    fuzz_score = float(np.clip((gain - 7) / 3, 0, 1) * np.clip((flat - 0.05) / 0.08, 0, 1))
    return {
        "gain_index": round(gain, 2), "label": label, "confidence": round(conf, 2), "tier": tier(conf),
        "crest_factor_db": round(crest, 2), "upper_harmonic_ratio": round(hr, 3), "mid_flatness": round(flat, 4),
        "sustain_decay_db_per_s": round(decay, 2), "fuzz_likelihood": round(fuzz_score, 2),
        "evidence": [
            f"Crest factor {crest:.1f} dB ({'very compressed/saturated' if crest < 10 else 'moderate' if crest < 15 else 'dynamic/clean'})",
            f"Upper-harmonic energy ratio {hr:.2f} ({'rich' if hr > 0.8 else 'moderate' if hr > 0.4 else 'pure'})",
            f"Long notes decay at {decay:.1f} dB/s ({'long sustain' if decay < 6 else 'natural decay'})",
        ],
    }


def analyze_compression(mono: np.ndarray, sr: int, notes: list[dict], gain_index: float) -> dict:
    hop = 256
    rms = librosa.feature.rms(y=mono, frame_length=1024, hop_length=hop)[0]
    env = _db(rms)
    peaks = []
    sustains = []
    for n in notes:
        a = int(n["start"] * sr / hop)
        if a + 12 >= len(env):
            continue
        p = env[a: a + 4].max()
        peaks.append(p)
        if n["end"] - n["start"] > 0.25:
            sustains.append(p - env[a + 10: a + 14].mean())
    if len(peaks) < 6:
        return {"detected": False, "confidence": 0.2, "tier": "uncertain", "note": "Not enough notes to judge dynamics"}
    attack_range = float(np.percentile(peaks, 95) - np.percentile(peaks, 10))
    act = env[env > env.max() - 40]
    lra = float(np.percentile(act, 95) - np.percentile(act, 10)) if act.size else 0.0
    ats = float(np.median(sustains)) if sustains else 6.0
    score = 0.0
    score += np.clip((14 - attack_range) / 10, 0, 1) * 0.45
    score += np.clip((8 - ats) / 7, 0, 1) * 0.35
    score += np.clip((20 - lra) / 14, 0, 1) * 0.2
    # distortion compresses on its own: discount accordingly
    score_eff = float(score * (1 - 0.4 * np.clip(gain_index / 10, 0, 1)))
    ratio = "2:1 (light)" if score_eff < 0.45 else "4:1 (moderate)" if score_eff < 0.7 else "8:1+ (heavy)"
    return {"detected": score_eff > 0.35, "amount": round(score_eff, 2), "estimated_ratio": ratio,
            "confidence": round(score_eff, 2), "tier": tier(score_eff),
            "attack_level_range_db": round(attack_range, 1), "loudness_range_db": round(lra, 1),
            "attack_to_sustain_db": round(ats, 1),
            "evidence": [f"Note attacks vary by only {attack_range:.1f} dB", f"Attack-to-sustain drop {ats:.1f} dB"]}


def analyze_reverb(stereo: np.ndarray, sr: int, gaps: list[tuple[float, float]], max_window: float = 3.0) -> dict:
    """RT60 from the slope of the level envelope in phrase-ending tails.

    The first 40 ms after the note ends (direct-sound damping) are skipped; the slope is
    fitted on the part of the tail between -3 dB and -40 dB relative to its start, but
    only while above the noise/silence floor. A very bumpy tail is echo, not reverb, and
    lowers confidence.
    """
    mono = stereo.mean(axis=0)
    hop = 256
    rate = sr / hop
    env = _db(librosa.feature.rms(y=mono, frame_length=1024, hop_length=hop)[0])
    level_ref = np.percentile(env, 95)
    rt60s, tails, corrs, bumpy = [], [], [], []
    for g0, g1 in gaps:
        a = int((g0 + 0.04) * rate)
        b = int(min(g1, g0 + max_window) * rate)
        if b - a < 12:
            continue
        seg = env[a:b]
        start = float(np.median(seg[:3]))
        if start < level_ref - 35:
            continue
        floor = max(level_ref - 75, -150)
        valid = (seg <= start - 3) & (seg >= start - 40) & (seg > floor + 6)
        idx = np.where(valid)[0]
        if len(idx) < 6:
            # fell through the window almost immediately -> dry
            rt60s.append(0.1)
            continue
        idx = idx[: np.argmax(np.diff(np.concatenate([idx, [idx[-1] + 99]])) > 3) + 1]  # first contiguous run
        if len(idx) < 6:
            rt60s.append(0.1)
            continue
        t = idx / rate
        y = seg[idx]
        slope, icpt = np.polyfit(t, y, 1)
        resid = float(np.std(y - (slope * t + icpt)))
        if slope < 0:
            rt60s.append(float(np.clip(-60 / slope, 0.05, 8)))
            bumpy.append(resid)
        tails.append(start - level_ref)
        s0, s1 = int((g0 + 0.08) * sr), int(min(g1, g0 + 0.8) * sr)
        if s1 - s0 > 2048:
            L, R = stereo[0, s0:s1], stereo[1, s0:s1]
            if np.std(L) > 1e-6 and np.std(R) > 1e-6:
                corrs.append(float(np.corrcoef(L, R)[0, 1]))
    if not rt60s:
        return {"detected": None, "confidence": 0.15, "tier": "uncertain",
                "note": "No clear phrase endings to measure reverb tails"}
    rt60 = float(np.median(rt60s))
    wet = float(np.median(tails)) if tails else -40.0
    corr = float(np.median(corrs)) if corrs else 1.0
    bump = float(np.median(bumpy)) if bumpy else 0.0
    if rt60 < 0.3:
        kind = "None / very dry"
    elif rt60 < 0.8:
        kind = "Room / short plate"
    elif rt60 < 1.8:
        kind = "Plate / medium hall"
    elif rt60 < 3.2:
        kind = "Hall"
    else:
        kind = "Large hall / ambient"
    detected = rt60 >= 0.3
    conf = float(np.clip(0.3 + 0.08 * len(rt60s), 0.3, 0.8))
    if detected and bump > 2.0:
        conf *= 0.6  # tail is dominated by discrete echoes
    mix = float(np.clip((wet + 40) / 30, 0, 1)) * 100 if detected else 0.0
    return {"detected": detected, "type": kind, "rt60_s": round(rt60, 2), "decay_s": round(rt60, 2),
            "tail_level_db": round(wet, 1), "estimated_mix_percent": round(mix), "stereo_tail_correlation": round(corr, 2),
            "stereo": corr < 0.6, "confidence": round(conf, 2), "tier": tier(conf), "tail_roughness_db": round(bump, 2),
            "measurements": len(rt60s),
            "evidence": [f"Median RT60 ≈ {rt60:.2f} s over {len(rt60s)} phrase endings",
                         f"Tail L/R correlation {corr:.2f} ({'wide stereo reverb' if corr < 0.6 else 'mono-ish'})"]}


def analyze_delay(mono: np.ndarray, sr: int, gaps: list[tuple[float, float]], bpm: float) -> dict:
    hop = 220  # 5 ms
    rms = librosa.feature.rms(y=mono, frame_length=882, hop_length=hop)[0]
    env = _db(rms)
    candidates = []
    ref = float(np.percentile(env, 95))
    for g0, g1 in gaps:
        a = int(g0 * sr / hop)
        b = int(min(g1, g0 + 2.5) * sr / hop)
        if b - a < 60:
            continue
        seg = env[a:b]
        alive = np.where(seg > ref - 60)[0]
        if len(alive) < 40:
            continue
        seg = seg[: alive[-1] + 1]
        resid = _detrend(seg, 3)
        pk, _ = find_peaks(resid, prominence=3.0, distance=int(0.06 * sr / hop))
        pk = pk[seg[pk] > ref - 55]
        if len(pk) >= 3:
            d = np.diff(pk) * hop / sr
            med = float(np.median(d))
            regular = float(np.mean(np.abs(d - med) < 0.1 * med)) if med > 0 else 0
            if regular < 0.66:
                continue
            lv = seg[pk]
            fb = float(np.median(np.diff(lv)))
            first_echo = float(lv[0] - seg[0])
            candidates.append((med, regular, fb, first_echo))
    # global autocorrelation of the envelope as secondary evidence
    onset = np.maximum(np.diff(env, prepend=env[0]), 0)
    ac = librosa.autocorrelate(onset - onset.mean(), max_size=int(1.2 * sr / hop))
    ac = ac / (ac[0] + 1e-9)
    if candidates:
        times = np.array([c[0] for c in candidates])
        regs = np.array([c[1] for c in candidates])
        t = float(np.median(times))
        consistency = float(np.mean(np.abs(times - t) < 0.1 * t))
        lag = int(t * sr / hop)
        ac_val = float(ac[lag]) if 0 < lag < len(ac) else 0.0
        score = float(np.clip(0.15 + 0.35 * consistency * np.mean(regs) + 0.08 * min(len(candidates), 4)
                              + 0.15 * max(ac_val, 0), 0, 0.9))
        if len(candidates) < 2 or consistency < 0.5:
            score = min(score, 0.4)
        fb_db = float(np.median([c[2] for c in candidates]))
        feedback = float(np.clip(10 ** (fb_db / 20), 0.05, 0.9))
        mix = float(np.clip(10 ** (np.median([c[3] for c in candidates]) / 20), 0.05, 0.8))
        beat = 60.0 / max(bpm, 1)
        rel = t / beat
        names = {0.25: "1/16", 0.5: "1/8", 0.75: "dotted 1/8", 1.0: "1/4", 1 / 3: "1/8 triplet", 1.5: "dotted 1/4",
                 2 / 3: "1/4 triplet", 2.0: "1/2"}
        best = min(names, key=lambda r: abs(rel - r))
        synced = abs(rel - best) < 0.06 * best
        return {"detected": score >= 0.45, "time_ms": round(t * 1000), "feedback_percent": round(feedback * 100),
                "estimated_mix_percent": round(mix * 100), "tempo_synced": synced,
                "note_value": names[best] if synced else None, "confidence": round(score, 2), "tier": tier(score),
                "evidence": [f"Regular echoes every ≈ {t * 1000:.0f} ms after {len(candidates)} phrase endings",
                             f"Echo level drops {abs(fb_db):.1f} dB per repeat"]}
    return {"detected": False, "confidence": 0.2, "tier": "uncertain", "evidence": ["No regular echoes found in tails"]}


def _detrend(series: np.ndarray, deg: int = 2) -> np.ndarray:
    if len(series) < deg + 2:
        return series - np.mean(series)
    t = np.linspace(-1, 1, len(series))
    return series - np.polyval(np.polyfit(t, series, deg), t)


def _modulation(series: np.ndarray, rate_hz: float, fmin: float, fmax: float) -> tuple[float, float]:
    """Return (strength, frequency) of periodic modulation in ``series`` sampled at rate_hz."""
    x = _detrend(np.asarray(series, dtype=float))
    if len(x) < 16 or np.std(x) < 1e-6:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-12
    lo = max(1, int(rate_hz / fmax))
    hi = min(len(ac) - 1, int(rate_hz / fmin))
    if hi <= lo + 1:
        return 0.0, 0.0
    seg = ac[lo:hi]
    # require a genuine local maximum (not the edge of a decaying autocorrelation)
    k_rel = int(np.argmax(seg))
    if k_rel == 0 or k_rel == len(seg) - 1:
        return 0.0, 0.0
    k = lo + k_rel
    return float(ac[k]), float(rate_hz / k)


def _clean_windows(notes: list[dict], min_len: float = 0.5) -> list[tuple[float, float]]:
    """Sustained-note windows that contain no other note attack (so note rhythm can't fake modulation)."""
    starts = np.array(sorted(n["start"] for n in notes))
    out = []
    for n in notes:
        if n["end"] - n["start"] < min_len + 0.1:
            continue
        a = n["start"] + 0.1
        later = starts[(starts > n["start"] + 0.03) & (starts < n["end"])]
        b = float(later[0]) - 0.02 if later.size else n["end"]
        if b - a >= min_len:
            out.append((a, b))
    return out


def analyze_modulation(stereo: np.ndarray, sr: int, notes: list[dict], f0_contours: list[np.ndarray]) -> dict:
    mono = stereo.mean(axis=0)
    hop = 220
    rate = sr / hop
    rms = librosa.feature.rms(y=mono, frame_length=882, hop_length=hop)[0]
    env = _db(rms)
    windows = _clean_windows(notes)
    trem, chorus_amp = [], []
    for a_t, b_t in windows:
        seg = env[int(a_t * rate): int(b_t * rate)]
        if len(seg) < 40:
            continue
        s, f = _modulation(seg, rate, 2.0, 12.0)
        resid = _detrend(seg)
        depth = float(np.percentile(resid, 95) - np.percentile(resid, 5))
        if s > 0.3 and depth > 2.0:
            trem.append((s, f, depth))
        s2, f2 = _modulation(seg, rate, 0.3, 2.5)
        if s2 > 0.3:
            chorus_amp.append((s2, f2))
    out = {}
    n_win = max(1, len(windows))
    if trem and len(windows) >= 2:
        s = float(np.median([t[0] for t in trem]))
        share = len(trem) / n_win
        rates = np.array([t[1] for t in trem])
        consistency = float(np.mean(np.abs(rates - np.median(rates)) < 0.15 * np.median(rates)))
        score = float(np.clip(s * share * consistency * 1.3, 0, 0.9))
        out["tremolo"] = {"detected": score >= 0.45, "rate_hz": round(float(np.median(rates)), 2),
                          "depth_db": round(float(np.median([t[2] for t in trem])), 1), "confidence": round(score, 2),
                          "tier": tier(score), "windows": len(windows)}
    else:
        out["tremolo"] = {"detected": False, "confidence": 0.15, "tier": "uncertain", "windows": len(windows)}
    # chorus / vibrato unit: slow pitch modulation common to all sustained notes
    slow = [(_modulation(c, 1 / (256 / 22050), 0.3, 3.0)) for c in f0_contours if c is not None and len(c) > 60]
    slow = [x for x in slow if x[0] > 0.35]
    L, R = stereo[0], stereo[1]
    corr = float(np.corrcoef(L[:: 4], R[:: 4])[0, 1]) if np.std(L) > 1e-6 and np.std(R) > 1e-6 else 1.0
    sus = sum(1 for c in f0_contours if c is not None and len(c) > 60)
    share = len(slow) / max(1, sus)
    amp_share = len(chorus_amp) / n_win
    score = float(np.clip(0.45 * share + 0.3 * amp_share + 0.25 * np.clip((0.85 - corr) / 0.5, 0, 1), 0, 0.85))
    out["chorus"] = {"detected": score >= 0.45, "rate_hz": round(float(np.median([x[1] for x in slow])), 2) if slow else None,
                     "depth": "wide" if corr < 0.4 else "moderate" if corr < 0.75 else "subtle",
                     "stereo_correlation": round(corr, 2), "confidence": round(score, 2), "tier": tier(score),
                     "note": "Stereo doubling of a part can mimic chorus" if corr < 0.5 else None}
    # flanger: sweeping comb -> cepstral peak at short quefrency that moves periodically
    S = np.abs(librosa.stft(mono[: sr * 60], n_fft=2048, hop_length=512))
    ceps = np.fft.irfft(np.log(S + 1e-6), axis=0)
    q = np.arange(ceps.shape[0]) / sr * 1000  # ms
    band = (q > 0.5) & (q < 12)
    if band.sum() > 4 and ceps.shape[1] > 40:
        peak_q = q[band][np.argmax(ceps[band], axis=0)]
        strength = np.max(ceps[band], axis=0)
        s, f = _modulation(peak_q, sr / 512, 0.05, 2.0)
        fl = float(np.clip(s * np.clip(np.median(strength) * 20, 0, 1), 0, 0.8))
        out["flanger"] = {"detected": fl >= 0.5, "rate_hz": round(f, 2) if fl > 0.3 else None,
                          "confidence": round(fl, 2), "tier": tier(fl)}
        # phaser: notch positions (spectral valleys) sweeping
        sm = librosa.decompose.nn_filter(np.log(S + 1e-6), aggregate=np.median, width=3) if S.shape[1] > 10 else np.log(S + 1e-6)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mb = (freqs > 200) & (freqs < 4000)
        valley = freqs[mb][np.argmin(sm[mb], axis=0)]
        s2, f2 = _modulation(np.log2(valley + 1), sr / 512, 0.05, 2.0)
        ph = float(np.clip(s2 * 0.8, 0, 0.7))
        out["phaser"] = {"detected": ph >= 0.5, "rate_hz": round(f2, 2) if ph > 0.3 else None,
                         "confidence": round(ph, 2), "tier": tier(ph)}
        # wah: strongly moving resonant peak 400-2500 Hz
        wb = (freqs > 400) & (freqs < 2500)
        res = freqs[wb][np.argmax(sm[wb], axis=0)]
        active = np.mean(S, axis=0) > np.percentile(np.mean(S, axis=0), 40)
        if active.sum() > 20:
            spread = float(np.percentile(np.log2(res[active]), 90) - np.percentile(np.log2(res[active]), 10))
            motion = float(np.median(np.abs(np.diff(np.log2(res[active])))))
            w = float(np.clip((spread - 0.8) / 1.2, 0, 1) * np.clip(motion * 8, 0, 1) * 0.8)
        else:
            spread, w = 0.0, 0.0
        out["wah"] = {"detected": w >= 0.5, "sweep_octaves": round(spread, 2), "confidence": round(w, 2), "tier": tier(w)}
    return out


def analyze_eq(mono: np.ndarray, sr: int) -> dict:
    S = np.abs(librosa.stft(mono, n_fft=4096, hop_length=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    active = S.sum(axis=0) > np.percentile(S.sum(axis=0), 30)
    ltas = S[:, active].mean(axis=1) if active.any() else S.mean(axis=1)
    ltas_db = _db(ltas)
    bands = _band_levels(ltas_db, freqs)
    ref_mid = bands["mid"]
    rel = {k: round(v - ref_mid, 1) for k, v in bands.items()}
    dev = {k: round(rel[k] - REFERENCE[k], 1) for k in rel}
    peak = ltas_db[(freqs > 80) & (freqs < 5000)].max()
    above = np.where((ltas_db > peak - 30) & (freqs > 1000))[0]
    rolloff = float(freqs[above.max()]) if above.size else 1000.0
    character = []
    scoop = (dev["low"] + dev["presence"]) / 2 - dev["mid"]
    if scoop > 3:
        character.append("Scooped mids")
    elif scoop < -3:
        character.append("Mid-forward")
    if dev["presence"] + dev["upper_mid"] > 5:
        character.append("Bright")
    elif dev["presence"] + dev["upper_mid"] < -6:
        character.append("Dark / warm")
    if dev["low"] > 4:
        character.append("Heavy low end")
    elif dev["low"] < -5:
        character.append("Thin / tight low end")
    if rolloff < 6500:
        character.append("Speaker-cabinet roll-off")
    elif rolloff > 10000:
        character.append("Full-range (DI, acoustic or no cab sim)")
    # suggested amp EQ knobs (0-10, 5 = flat)
    knobs = {
        "bass": round(float(np.clip(5 + dev["low"] / 2.5, 1, 9)), 1),
        "mid": round(float(np.clip(5 - scoop / 2.0, 1, 9)), 1),
        "treble": round(float(np.clip(5 + dev["upper_mid"] / 2.5, 1, 9)), 1),
        "presence": round(float(np.clip(5 + dev["presence"] / 3.0, 1, 9)), 1),
    }
    return {"band_levels_db": rel, "deviation_from_reference_db": dev, "rolloff_hz": round(rolloff),
            "character": character or ["Balanced"], "suggested_amp_eq": knobs,
            "cabinet": {"detected": rolloff < 7500, "confidence": 0.6 if rolloff < 6500 else 0.4,
                        "tier": tier(0.6 if rolloff < 6500 else 0.35)},
            "spectrum": {"freqs": [round(float(f)) for f in freqs[2:-1:8]], "db": [round(float(v), 1) for v in ltas_db[2:-1:8]]}}


def analyze_gate(mono: np.ndarray, sr: int, gaps: list[tuple[float, float]], gain_index: float) -> dict:
    hop = 256
    env = _db(librosa.feature.rms(y=mono, frame_length=1024, hop_length=hop)[0])
    floors, drops = [], []
    for g0, g1 in gaps:
        a = int((g0 + 0.25) * sr / hop)
        b = int(g1 * sr / hop)
        if b - a < 8:
            continue
        floors.append(float(np.percentile(env[a:b], 20)))
        seg = env[int(g0 * sr / hop): a]
        if len(seg) > 3:
            drops.append(float(np.max(-np.diff(seg))))
    if not floors:
        return {"detected": None, "confidence": 0.15, "tier": "uncertain"}
    floor = float(np.median(floors))
    top = float(np.percentile(env, 95))
    depth = top - floor
    abrupt = float(np.median(drops)) if drops else 0.0
    score = float(np.clip((depth - 45) / 25, 0, 1) * np.clip(gain_index / 6, 0, 1) * 0.6 + np.clip((abrupt - 6) / 10, 0, 0.3))
    return {"detected": score >= 0.45, "gap_floor_db": round(floor - top, 1), "confidence": round(score, 2),
            "tier": tier(score), "evidence": [f"Silences sit {depth:.0f} dB below the playing level"]}


def analyze_octave(mono: np.ndarray, sr: int, notes: list[dict]) -> dict:
    S = np.abs(librosa.stft(mono, n_fft=8192, hop_length=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=8192)
    ratios = []
    for n in notes:
        if n["end"] - n["start"] < 0.25 or n.get("confidence", 0) < 0.6:
            continue
        f0 = 440 * 2 ** ((n["pitch"] - 69) / 12)
        if f0 / 2 < 40:
            continue
        a, b = int(n["start"] * sr / 2048), int(n["end"] * sr / 2048)
        if b <= a or b > S.shape[1]:
            continue
        spec = S[:, a:b].mean(axis=1)
        def e(f):
            m = (freqs > f * 0.97) & (freqs < f * 1.03)
            return spec[m].max() if m.any() else 0.0
        ratios.append(e(f0 / 2) / (e(f0) + 1e-9))
    if len(ratios) < 5:
        return {"octave_down": {"detected": None, "confidence": 0.15, "tier": "uncertain"}}
    r = float(np.median(ratios))
    score = float(np.clip((r - 0.2) / 0.5, 0, 0.85))
    return {"octave_down": {"detected": score >= 0.5, "sub_octave_ratio": round(r, 3), "confidence": round(score, 2),
                            "tier": tier(score)}}


def analyze_harmonizer(notes: list[dict]) -> dict:
    """Constant parallel intervals on a melodic line suggest a harmoniser/pitch shifter."""
    groups: list[list[dict]] = []
    for n in sorted(notes, key=lambda n: n["start"]):
        if groups and abs(n["start"] - groups[-1][0]["start"]) < 0.02:
            groups[-1].append(n)
        else:
            groups.append([n])
    pairs = [g for g in groups if len(g) == 2]
    if len(pairs) < 10:
        return {"detected": False, "confidence": 0.1, "tier": "uncertain"}
    ivs = [abs(g[0]["pitch"] - g[1]["pitch"]) for g in pairs]
    durs = [abs((g[0]["end"] - g[0]["start"]) - (g[1]["end"] - g[1]["start"])) < 0.03 for g in pairs]
    vals, counts = np.unique(ivs, return_counts=True)
    top = int(vals[np.argmax(counts)])
    share = counts.max() / len(ivs)
    score = float(np.clip((share - 0.5) * 1.6 * np.mean(durs), 0, 0.8)) if top not in (7, 5) else float(
        np.clip((share - 0.7) * 1.2 * np.mean(durs), 0, 0.5))  # 5ths/4ths are usually just power chords
    names = {12: "octave", 7: "fifth", 5: "fourth", 3: "minor third", 4: "major third", 19: "octave + fifth"}
    return {"detected": score >= 0.5, "interval": names.get(top, f"{top} semitones"), "share": round(float(share), 2),
            "confidence": round(score, 2), "tier": tier(score),
            "note": "Could also be two double-tracked guitars playing a harmony"}


def analyze_stereo(stereo: np.ndarray) -> dict:
    L, R = stereo[0].astype(np.float64), stereo[1].astype(np.float64)
    M, Sd = (L + R) / 2, (L - R) / 2
    em, es = np.sum(M**2) + 1e-12, np.sum(Sd**2)
    width = float(np.sqrt(es / em))
    corr = float(np.corrcoef(L[::4], R[::4])[0, 1]) if np.std(L) > 1e-9 and np.std(R) > 1e-9 else 1.0
    el, er = np.sum(L**2), np.sum(R**2)
    pan = float((er - el) / (er + el + 1e-12))
    return {"width": round(width, 3), "correlation": round(corr, 3), "pan": round(pan, 2),
            "description": ("mono" if width < 0.05 else "narrow" if width < 0.3 else "wide" if width < 0.8 else "very wide"),
            "pan_label": "center" if abs(pan) < 0.15 else ("right" if pan > 0 else "left")}


def analyze_tone(stereo: np.ndarray, sr: int, notes: list[dict], bpm: float, duration: float,
                 f0_contours: list[np.ndarray] | None = None) -> dict:
    if stereo.ndim == 1:
        stereo = np.vstack([stereo, stereo])
    mono = stereo.mean(axis=0)
    if np.max(np.abs(mono)) < 1e-4:
        return {"available": False, "reason": "Stem is silent"}
    hop = 1024
    S = np.abs(librosa.stft(mono, n_fft=4096, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    gaps = _gaps(notes, duration)
    dist = analyze_distortion(mono, sr, notes, S, freqs, hop)
    comp = analyze_compression(mono, sr, notes, dist["gain_index"])
    dly = analyze_delay(mono, sr, gaps, bpm)
    if dly.get("detected"):
        # measure the reverb only before the first echo arrives, otherwise repeats look like a long tail
        rev = analyze_reverb(stereo, sr, gaps, max_window=max(0.12, 0.9 * dly["time_ms"] / 1000))
        if rev.get("detected"):
            # repeats of the sustained notes fill the tail, so the decay estimate is unreliable
            rev["confidence"] = round(min(rev["confidence"], 0.45), 2)
            rev["tier"] = tier(rev["confidence"])
            rev.setdefault("evidence", []).append("Tail overlaps delay repeats: reverb amount is a rough estimate")
    else:
        rev = analyze_reverb(stereo, sr, gaps)
    mod = analyze_modulation(stereo, sr, notes, f0_contours or [])
    if dly.get("detected") and mod.get("tremolo", {}).get("rate_hz"):
        echo_rate = 1000.0 / dly["time_ms"]
        tr = mod["tremolo"]
        if any(abs(tr["rate_hz"] - k * echo_rate) < 0.12 * k * echo_rate for k in (1, 2)):
            tr.update({"detected": False, "confidence": round(tr["confidence"] * 0.3, 2), "tier": "uncertain",
                       "note": "Amplitude modulation explained by delay repeats"})
    eq = analyze_eq(mono, sr)
    gate = analyze_gate(mono, sr, gaps, dist["gain_index"])
    octv = analyze_octave(mono, sr, notes)
    harm = analyze_harmonizer(notes)
    stereo_info = analyze_stereo(stereo)

    effects = []

    def add(name, key, d, summary):
        conf = float(d.get("confidence", 0))
        effects.append({"effect": name, "key": key, "detected": bool(d.get("detected")), "confidence": round(conf, 2),
                        "tier": d.get("tier", tier(conf)), "summary": summary, "details": d})

    gi = dist["gain_index"]
    effects.append({"effect": "Gain / saturation", "key": "gain", "detected": True, "confidence": dist["confidence"],
                    "tier": dist["tier"], "summary": f"{dist['label']} (gain ≈ {gi:.1f}/10)", "details": dist})
    if dist["fuzz_likelihood"] > 0.3:
        add("Fuzz", "fuzz", {"detected": dist["fuzz_likelihood"] > 0.5, "confidence": dist["fuzz_likelihood"]},
            "Very saturated, flat-mid spectrum typical of fuzz pedals")
    add("Compression", "compression", comp, f"{comp.get('estimated_ratio', '')} – attacks within "
        f"{comp.get('attack_level_range_db', '?')} dB" if comp.get("detected") else "No strong compression detected")
    add("Reverb", "reverb", rev, f"{rev.get('type', 'Unknown')} – decay ≈ {rev.get('rt60_s', '?')} s" if rev.get("detected")
        else rev.get("note", "Dry or not measurable"))
    add("Delay / echo", "delay", dly, (f"{dly['time_ms']} ms" + (f" ({dly['note_value']})" if dly.get("note_value") else "")
                                       + f", feedback ≈ {dly['feedback_percent']}%") if dly.get("detected") else "No regular echoes detected")
    for k, label in (("tremolo", "Tremolo"), ("chorus", "Chorus / vibrato"), ("flanger", "Flanger"), ("phaser", "Phaser"),
                     ("wah", "Wah")):
        if k in mod:
            d = mod[k]
            rate = d.get("rate_hz")
            add(label, k, d, (f"rate ≈ {rate} Hz" if rate else "detected") if d.get("detected") else "Not detected")
    add("Noise gate", "gate", gate, "Abrupt silence between phrases" if gate.get("detected") else "No obvious gating")
    add("Octave down", "octave", octv["octave_down"], "Sub-octave energy under notes" if octv["octave_down"].get("detected")
        else "Not detected")
    add("Harmonizer / pitch shift", "harmonizer", harm, f"Parallel {harm.get('interval')}" if harm.get("detected") else "Not detected")
    effects.append({"effect": "EQ / cabinet", "key": "eq", "detected": True, "confidence": 0.6, "tier": "probable",
                    "summary": ", ".join(eq["character"]), "details": eq})
    effects.append({"effect": "Stereo image", "key": "stereo", "detected": True, "confidence": 0.9, "tier": "high",
                    "summary": f"{stereo_info['description']} ({stereo_info['pan_label']})", "details": stereo_info})

    # amplifier character (described, not identified)
    amp = describe_amp(gi, eq)
    return {"available": True, "effects": effects, "amp_character": amp, "gain": dist, "eq": eq, "stereo": stereo_info,
            "disclaimer": ("Estimates describe the sound's character from the separated stem. The exact pedals, amp, "
                           "settings and studio processing cannot be identified from a mixed recording.")}


def describe_amp(gain: float, eq: dict) -> dict:
    ch = eq["character"]
    scooped = "Scooped mids" in ch
    midfwd = "Mid-forward" in ch
    bright = "Bright" in ch
    if gain < 2:
        style = "American-style clean (bright, glassy, lots of headroom)" if bright or scooped else "Warm clean (dark, round)"
    elif gain < 4:
        style = "Chimey British-style crunch (upper-mid focus)" if bright else "Tweed-style edge-of-breakup"
    elif gain < 6.5:
        style = "Classic British crunch (mid-forward)" if midfwd or not scooped else "Hot-rodded crunch"
    elif gain < 8:
        style = "Modern British/hot-rodded distortion"
    else:
        style = "Modern high-gain (tight lows, scooped mids)" if scooped else "Saturated high-gain lead"
    return {"style": style, "gain": round(gain, 1), "eq": eq["suggested_amp_eq"],
            "cabinet": "4x12 closed back" if gain >= 5 else "1x12 / 2x12 open back"}
