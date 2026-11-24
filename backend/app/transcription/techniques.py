"""Articulation / playing-technique detection.

Every detector is a transparent DSP rule over measurable evidence, and each technique
carries its own confidence so the UI can show *why* it was detected:

=================  ==========================================================  ============
Technique          Evidence                                                    Reliability
=================  ==========================================================  ============
bend / release     fine pitch contour rises >= 0.4 st and holds (or returns)   good (mono)
pre-bend release   contour falls >= 0.8 st from the attack pitch               fair
vibrato            3.5-9 Hz periodic pitch modulation, depth >= 0.12 st         good (mono)
slide              glide in f0 between consecutive notes, weak 2nd attack      fair
hammer-on/pull-off legato (no gap), weak broadband attack, interval 1-7 st     fair
palm mute          fast decay + darker attack spectrum on low notes            fair
dead strum (x)     strong broadband onset with no pitched energy               fair
strum direction    onset order across pitches inside a chord (+ metric pos.)   low-fair
tremolo picking    >= 4 repeated attacks of one pitch with IOI < 95 ms          good
=================  ==========================================================  ============

Final legato labels are confirmed after fingering (hammer-ons/pull-offs need the same
string), see :func:`app.guitar.fingering.finalize_techniques`.
"""

from __future__ import annotations

import numpy as np
import librosa

from .base import NoteEvent

SR = 22050
HOP = 256


def _features(y22: np.ndarray) -> dict:
    S = np.abs(librosa.stft(y22, n_fft=1024, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=1024)
    rms = librosa.feature.rms(S=S, frame_length=1024)[0]
    rms_db = 20 * np.log10(rms + 1e-6)
    logS = np.log1p(S * 10)
    flux = np.maximum(logS[:, 1:] - logS[:, :-1], 0)
    hf = flux[freqs >= 2000].sum(axis=0)
    hf = np.concatenate([[0], hf])
    full = np.concatenate([[0], flux.sum(axis=0)])
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
    return {
        "rms_db": rms_db,
        "hf_flux": hf / (np.percentile(hf, 99) + 1e-9),
        "flux": full / (np.percentile(full, 99) + 1e-9),
        "centroid": centroid,
    }


def _frame(t: float) -> int:
    return int(round(t * SR / HOP))


def _attack_strength(feat: dict, t: float) -> float:
    f = _frame(t)
    seg = feat["hf_flux"][max(0, f - 2): f + 3]
    return float(seg.max()) if seg.size else 0.0


def _quantize_bend(st: float) -> float:
    return float(np.clip(round(st * 2) / 2, 0.5, 3.0))


def detect_bend(n: NoteEvent) -> dict | None:
    c = n.contour
    if c is None or len(c) < 6:
        return None
    hop = n.contour_hop
    skip = min(len(c) - 1, int(0.025 / hop))
    body = c[skip:]
    peak = float(np.max(body))
    k_peak = int(np.argmax(body)) + skip
    above = body > 0.35
    hold = above.sum() * hop
    if peak >= 0.4 and hold >= 0.04:
        start_level = float(np.median(c[: max(2, skip + 1)]))
        if start_level > 0.35:
            return None  # starts high: either pre-bend or the pitch class is wrong
        tail = c[k_peak:]
        released = len(tail) > 3 and float(np.min(tail[len(tail) // 2:])) < 0.25 and peak - float(tail[-1]) > 0.35
        rise_t = (k_peak - skip) * hop
        amount = _quantize_bend(peak)
        conf = float(np.clip(0.45 + 0.4 * min(1, hold / 0.12) + (0.15 if n.sources.get("contour") == "f0" else 0), 0, 0.95))
        return {"type": "bend_release" if released else "bend", "semitones": amount, "measured": round(peak, 2),
                "rise_time": round(rise_t, 3), "peak_at": round(k_peak * hop, 3), "confidence": round(conf, 2)}
    trough = float(np.min(body))
    if trough <= -0.8 and float(np.median(c[: max(2, skip + 1)])) > -0.25:
        amount = _quantize_bend(-trough)
        return {"type": "prebend_release", "semitones": amount, "measured": round(-trough, 2),
                "confidence": 0.45}
    return None


def detect_vibrato(n: NoteEvent) -> dict | None:
    c = n.contour
    if c is None:
        return None
    hop = n.contour_hop
    skip = int(0.08 / hop)
    seg = c[skip:]
    if len(seg) * hop < 0.25:
        return None
    win = max(3, int(0.16 / hop) | 1)
    trend = np.convolve(seg, np.ones(win) / win, mode="same")
    d = seg - trend
    d = d[win // 2: len(d) - win // 2] if len(d) > win else d
    if len(d) < 12:
        return None
    d = d - d.mean()
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    if ac[0] <= 1e-9:
        return None
    ac = ac / ac[0]
    lag_min = int(1 / 9.0 / hop)
    lag_max = min(len(ac) - 1, int(1 / 3.5 / hop))
    if lag_max <= lag_min + 1:
        return None
    k = lag_min + int(np.argmax(ac[lag_min:lag_max]))
    strength = float(ac[k])
    depth = float(np.percentile(d, 95) - np.percentile(d, 5))
    if strength < 0.35 or depth < 0.12:
        return None
    rate = 1.0 / (k * hop)
    return {"type": "vibrato", "rate_hz": round(rate, 2), "depth_semitones": round(depth, 2),
            "wide": depth > 0.6, "confidence": round(float(np.clip(0.3 + strength * 0.6, 0, 0.95)), 2)}


def _glide_between(a: NoteEvent, b: NoteEvent, f0: dict | None) -> float:
    """Fraction of the interval a->b covered by a continuous glide near the boundary."""
    if f0 is None:
        return 0.0
    hop = f0["hop"]
    i0 = int(max(a.start, b.start - 0.12) / hop)
    i1 = int((b.start + 0.04) / hop)
    seg = f0["midi"][i0:i1]
    v = f0["voicing"][i0:i1]
    seg = seg[(v > 0.4) & np.isfinite(seg)]
    interval = b.pitch - a.pitch
    if len(seg) < 4 or interval == 0:
        return 0.0
    rel = (seg - a.pitch) / interval
    inter = np.mean((rel > 0.15) & (rel < 0.85))  # time spent between the two pitches
    mono = np.mean(np.diff(seg) * np.sign(interval) > -0.15)
    return float(np.clip(inter * 3, 0, 1) * mono)


def merge_bend_chains(notes: list[NoteEvent], attack_ref: float) -> tuple[list[NoteEvent], int]:
    """Re-join bends that the note decoder split into a semitone 'staircase'.

    A bend from E to F# often comes out as E -> F -> F#, each later step without a pick
    attack. A chain of contiguous notes moving by 1-2 semitones, without new attacks,
    rising at most 3 semitones over >= 120 ms (slides are faster), is one bent note;
    an immediate return towards the first pitch is its release.
    """
    order = sorted(notes, key=lambda n: n.start)
    used: set[int] = set()
    out: list[NoteEvent] = []
    merged = 0
    for i, a in enumerate(order):
        if id(a) in used:
            continue
        chain = [a]
        j = i + 1
        peak = a.pitch
        rising = True
        while j < len(order):
            prev, b = chain[-1], order[j]
            if b.start - prev.end > 0.04:
                break
            if b.start < prev.end - 0.04 or id(b) in used:
                j += 1
                continue
            step = b.pitch - prev.pitch
            weak = getattr(b, "attack", 1.0) < 0.3 * attack_ref or "no_onset" in b.flags
            if not weak or step == 0 or abs(step) > 2:
                break
            if rising and step > 0 and b.pitch - a.pitch <= 3:
                peak = b.pitch
            elif step < 0 and b.pitch >= a.pitch:
                rising = False  # release back towards the fretted pitch
            else:
                break
            chain.append(b)
            j += 1
        rise_time = next((c.start for c in chain if c.pitch == peak), a.start) - a.start
        if len(chain) >= 2 and peak > a.pitch and rise_time >= 0.12:
            hop = a.contour_hop
            parts = []
            for c in chain:
                cc = c.contour if c.contour is not None else np.zeros(max(1, int(c.duration / hop)), np.float32)
                parts.append(np.asarray(cc, np.float32) + (c.pitch - a.pitch))
            a.contour = np.concatenate(parts)
            a.end = chain[-1].end
            a.confidence = float(np.mean([c.confidence for c in chain]))
            a.flags.add("merged_bend")
            for c in chain[1:]:
                used.add(id(c))
            merged += len(chain) - 1
        out.append(a)
    return out, merged


def detect(notes: list[NoteEvent], y22: np.ndarray, f0: dict | None, polyphony: np.ndarray,
           beat_times: np.ndarray | None = None) -> dict:
    feat = _features(y22)
    for n in notes:
        n.techniques = []
        n.attack = _attack_strength(feat, n.start)  # type: ignore[attr-defined]
    if notes:
        ref = float(np.percentile([n.attack for n in notes], 75))  # type: ignore[attr-defined]
        merged_list, n_merged = merge_bend_chains(notes, ref)
        notes[:] = merged_list  # in place: callers keep their reference
    else:
        n_merged = 0

    # ---- per-note contour techniques
    for n in notes:
        b = detect_bend(n)
        if b:
            if b["type"] == "prebend_release":
                # the attack pitch is the bent pitch: the fretted note is lower
                n.techniques.append({**b, "fretted_pitch": n.pitch - int(round(b["semitones"]))})
            else:
                n.techniques.append(b)
        v = detect_vibrato(n)
        if v:
            n.techniques.append(v)
        # slide into the note from below/above (contour starts >= 1 st away and converges fast)
        if n.contour is not None and len(n.contour) > 6:
            st = float(np.median(n.contour[:3]))
            if st <= -1.0 and float(np.median(n.contour[4:8])) > -0.4:
                n.techniques.append({"type": "slide_in", "direction": "from_below", "confidence": 0.45})
            elif st >= 1.0 and float(np.median(n.contour[4:8])) < 0.4:
                n.techniques.append({"type": "slide_in", "direction": "from_above", "confidence": 0.4})

    # ---- transitions between consecutive notes in monophonic lines
    order = sorted(notes, key=lambda n: n.start)
    attacks = np.array([getattr(n, "attack", 0.5) for n in order]) if order else np.array([])
    attack_ref = float(np.percentile(attacks, 75)) if len(attacks) else 1.0
    for a, b in zip(order[:-1], order[1:]):
        fa = _frame(b.start)
        if polyphony[max(0, fa - 2): fa + 2].max(initial=0) > 2:
            continue
        gap = b.start - a.end
        if not (-0.05 <= gap <= 0.045) or a.pitch == b.pitch:
            continue
        interval = b.pitch - a.pitch
        weak = getattr(b, "attack", 1.0) < 0.45 * attack_ref or "no_onset" in b.flags or b.onset_strength < 0.45
        glide = _glide_between(a, b, f0)
        if glide > 0.35 and abs(interval) <= 12:
            if 0 < interval <= 3 and not weak:
                continue  # glide but re-picked: not a slide
            a.techniques.append({"type": "slide", "direction": "up" if interval > 0 else "down", "to_pitch": b.pitch,
                                 "legato": weak, "confidence": round(0.35 + 0.5 * glide, 2)})
        elif weak and 1 <= abs(interval) <= 7:
            kind = "hammer_on" if interval > 0 else "pull_off"
            conf = 0.4 + 0.3 * (1 - min(1.0, getattr(b, "attack", 0.0) / (0.45 * attack_ref + 1e-9)))
            b.techniques.append({"type": kind, "from_pitch": a.pitch, "confidence": round(float(conf), 2),
                                 "provisional": True})

    # ---- palm mute
    if notes:
        cents = []
        for n in notes:
            f = _frame(n.start)
            cents.append(float(np.median(feat["centroid"][f: f + 6])) if f + 1 < len(feat["centroid"]) else 0.0)
        med_c = float(np.median(cents)) if cents else 1.0
        for n, c in zip(notes, cents):
            if n.pitch > 60:
                continue
            f = _frame(n.start)
            seg = feat["rms_db"][f: f + int(0.15 * SR / HOP)]
            if len(seg) < 5:
                continue
            peak = float(seg[:4].max())
            drop = peak - float(seg[-1])
            short = n.duration < 0.2
            dark = c < 0.85 * med_c
            score = 0.4 * min(1.0, drop / 18.0) + 0.3 * short + 0.3 * dark
            if drop > 10 and score >= 0.62:
                n.techniques.append({"type": "palm_mute", "decay_db": round(drop, 1),
                                     "confidence": round(min(0.85, score), 2)})

    # ---- tremolo picking: runs of >= 4 fast repeated same-pitch attacks
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in order:
        by_pitch.setdefault(n.pitch, []).append(n)
    for lst in by_pitch.values():
        run = [lst[0]] if lst else []
        for prev, cur in zip(lst[:-1], lst[1:]):
            if cur.start - prev.start < 0.095:
                run.append(cur)
            else:
                if len(run) >= 4:
                    for m in run:
                        m.techniques.append({"type": "tremolo_picking", "confidence": 0.7})
                run = [cur]
        if len(run) >= 4:
            for m in run:
                m.techniques.append({"type": "tremolo_picking", "confidence": 0.7})

    # ---- dead / muted strums: strong broadband attacks without pitched notes
    dead = []
    on = librosa.onset.onset_detect(onset_envelope=feat["flux"], sr=SR, hop_length=HOP, units="time",
                                    delta=0.15, wait=4)
    starts = np.array([n.start for n in notes]) if notes else np.array([])
    for t in on:
        if starts.size and np.min(np.abs(starts - t)) < 0.05:
            continue
        f = _frame(t)
        if feat["flux"][max(0, f - 1): f + 2].max(initial=0) < 0.45:
            continue
        if polyphony[f: f + 3].max(initial=0) > 0 and starts.size and np.min(np.abs(starts - t)) < 0.12:
            continue
        dead.append({"time": round(float(t), 3), "strength": round(float(feat["flux"][f]), 2), "confidence": 0.45})

    # ---- strum direction for chord groups
    strums = []
    groups = group_onsets(notes, tol=0.05)
    for g in groups:
        if len(g) < 3:
            continue
        ts = np.array([n.start for n in g])
        ps = np.array([n.pitch for n in g])
        spread = float(ts.max() - ts.min())
        direction, source, conf = None, None, 0.3
        if spread >= 0.012 and np.std(ps) > 0:
            r = float(np.corrcoef(ps, ts)[0, 1])
            if abs(r) > 0.5:
                direction = "down" if r > 0 else "up"
                source, conf = "onset order", round(0.35 + 0.4 * abs(r), 2)
        if direction is None and beat_times is not None and len(beat_times) > 1:
            t0 = float(ts.min())
            k = np.searchsorted(beat_times, t0) - 1
            if 0 <= k < len(beat_times) - 1:
                frac = (t0 - beat_times[k]) / (beat_times[k + 1] - beat_times[k])
                direction = "down" if frac < 0.25 or frac > 0.85 else "up"
                source, conf = "metric position", 0.3
        if direction:
            for n in g:
                n.techniques.append({"type": "strum", "direction": direction, "source": source, "confidence": conf})
            strums.append({"time": round(float(ts.min()), 3), "direction": direction, "spread_ms": round(spread * 1000, 1),
                           "source": source})

    # dedupe technique types per note (keep highest confidence)
    for n in notes:
        best: dict[str, dict] = {}
        for t in n.techniques:
            if t["type"] not in best or t.get("confidence", 0) > best[t["type"]].get("confidence", 0):
                best[t["type"]] = t
        n.techniques = list(best.values())
    return {"dead_strums": dead, "strums": strums, "merged_bend_notes": n_merged}


def group_onsets(notes: list[NoteEvent], tol: float = 0.04) -> list[list[NoteEvent]]:
    """Group notes whose onsets fall within ``tol`` seconds (chords / strums)."""
    groups: list[list[NoteEvent]] = []
    for n in sorted(notes, key=lambda n: n.start):
        if groups and n.start - groups[-1][0].start <= tol:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups
