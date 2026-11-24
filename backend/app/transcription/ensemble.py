"""Ensemble transcription: fuse Basic Pitch, DSP salience and an f0 tracker.

No single model is trusted blindly:

* **Basic Pitch** proposes polyphonic note candidates (primary, best recall).
* **DSP harmonic salience** independently measures whether each candidate's pitch is
  actually present in the spectrum (support score) and proposes notes the network
  missed.
* **CREPE/pYIN** gives a precise f0 in monophonic passages: it confirms pitch, exposes
  octave errors, and provides the fine contour for bends/vibrato.

Each note gets ``confidence`` in [0, 1] from a weighted vote, plus review ``flags``.
Candidates that were rejected (harmonic ghosts, very weak, out of range) are kept in a
separate list so the UI can show them as "rejected suggestions" the user can restore,
instead of silently throwing information away.
"""

from __future__ import annotations

import numpy as np

from . import basic_pitch_onnx as bp
from . import dsp_poly, f0_tracker
from .base import NoteEvent, confidence_level

SR = 22050
HOP = 256

INSTRUMENT_RANGE = {"guitar": (28, 96), "bass": (23, 72)}
# floors sit just below the lowest supported string (B1 on 7-string / drop B guitars, B0 on
# 5-string bass), otherwise the tracker locks onto the second harmonic of those notes
F0_RANGE = {"guitar": (38.0, 1400.0), "bass": (29.0, 450.0)}
GHOST_INTERVALS = (12, 19, 24, 28, 31)  # harmonics 2,3,4,5,6 above a fundamental


def _merge_fragments(notes: list[NoteEvent]) -> list[NoteEvent]:
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    out = []
    for p, lst in by_pitch.items():
        lst.sort(key=lambda n: n.start)
        merged: list[NoteEvent] = []
        for n in lst:
            if merged:
                prev = merged[-1]
                gap = n.start - prev.end
                weak = n.onset_strength < 0.55 or "no_onset" in n.flags
                if gap < 0.04 and weak:
                    # continuation of the previous note (re-detected sustain) -> extend
                    if n.contour is not None and prev.contour is not None:
                        pad = max(0, int(round((n.contour_t0 - prev.contour_t0) / prev.contour_hop)) - len(prev.contour))
                        prev.contour = np.concatenate([prev.contour, np.full(pad, prev.contour[-1]), n.contour])
                    prev.end = max(prev.end, n.end)
                    prev.amplitude = max(prev.amplitude, n.amplitude)
                    continue
                if "no_onset" in n.flags and n.duration < 0.09 and 0 <= (lst_next_start(lst, n) - n.end) < 0.04:
                    continue  # pre-echo of the next attack
            merged.append(n)
        out.extend(merged)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def lst_next_start(lst: list[NoteEvent], n: NoteEvent) -> float:
    later = [m.start for m in lst if m.start > n.start + 1e-6]
    return min(later) if later else 1e9


def _suppress_ghosts(notes: list[NoteEvent], ratio: float = 0.66) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Remove notes that are most likely harmonics of a louder lower note."""
    keep, rejected = [], []
    for b in notes:
        ghost_of = None
        for a in notes:
            if a is b or a.pitch >= b.pitch:
                continue
            if (b.pitch - a.pitch) not in GHOST_INTERVALS:
                continue
            if abs(a.start - b.start) > 0.06 or a.overlaps(b) < 0.6 * b.duration:
                continue
            if b.amplitude < ratio * a.amplitude:
                ghost_of = a
                break
            if b.amplitude < 0.85 * a.amplitude:
                b.flags.add("possible_harmonic")
        if ghost_of is not None:
            b.flags.add("rejected:harmonic_ghost")
            rejected.append(b)
        else:
            keep.append(b)
    return keep, rejected


def _polyphony(notes: list[NoteEvent], n_frames: int) -> np.ndarray:
    poly = np.zeros(n_frames, np.int16)
    for n in notes:
        a, b = int(n.start * SR / HOP), int(n.end * SR / HOP)
        poly[a: max(a + 1, b)] += 1
    return poly


def _f0_support(track: dict | None, n: NoteEvent, poly: np.ndarray) -> tuple[float | None, float]:
    if track is None:
        return None, 0.0
    a = int(n.start / track["hop"])
    b = max(a + 1, int(n.end / track["hop"]))
    midi = track["midi"][a:b]
    v = track["voicing"][a:b]
    mono = poly[a:b] <= 1
    ok = (v > 0.55) & np.isfinite(midi) & mono[: len(midi)]
    if ok.sum() < max(2, 0.4 * (b - a)):
        return None, 0.0
    d = midi[ok] - n.pitch
    match = np.mean((d > -0.6) & (d < 2.6))  # allow upward bends
    octave = np.mean(np.abs(np.abs(d) - 12) < 0.6)
    return float(match), float(octave)


def transcribe(y22: np.ndarray, instrument: str = "guitar", params: dict | None = None,
               progress=None) -> dict:
    params = params or {}
    prog = progress or (lambda p, m: None)
    engines = {}
    lo, hi = INSTRUMENT_RANGE.get(instrument, (28, 96))
    n_frames = int(np.ceil(len(y22) / HOP)) + 1

    prog(0.05, "Computing harmonic salience (DSP)")
    sal = dsp_poly.salience(y22)

    ok, why = bp.available()
    if ok:
        prog(0.15, "Running Basic Pitch")
        post = bp.run_model(y22, progress=lambda p, m: prog(0.15 + 0.45 * p, m))
        primary = bp.decode_notes(post, onset_threshold=params.get("onset_threshold", 0.5),
                                  frame_threshold=params.get("frame_threshold", 0.3),
                                  min_midi=lo - 2, max_midi=hi + 4)
        engines["basic_pitch"] = "onnx icassp_2022"
    else:
        engines["basic_pitch"] = f"unavailable: {why}"
        primary = dsp_poly.transcribe(sal)
        for n in primary:
            n.sources = {"dsp_primary": n.amplitude}
    dsp_notes = dsp_poly.transcribe(sal) if ok else []

    prog(0.65, "Tracking monophonic f0")
    fmin, fmax = F0_RANGE.get(instrument, (65.0, 1400.0))
    try:
        track = f0_tracker.track(y22, fmin, fmax)
    except Exception as e:  # noqa: BLE001
        track = None
        engines["f0"] = f"failed: {e}"
    if track is not None:
        engines["f0"] = track["engine"]

    prog(0.8, "Fusing estimates")
    notes = _merge_fragments(primary)
    notes, rejected = _suppress_ghosts(notes, ratio=params.get("ghost_ratio", 0.66))
    poly = _polyphony(notes, n_frames)

    for n in notes:
        c_model = float(np.clip((n.amplitude - 0.22) / 0.5, 0, 1)) if ok else float(np.clip(n.amplitude, 0, 1))
        s_dsp = dsp_poly.support(sal, n.pitch, n.start, n.end)
        s_f0, octave = _f0_support(track, n, poly)
        if s_f0 is None:
            conf = 0.65 * c_model + 0.35 * s_dsp
        else:
            conf = 0.5 * c_model + 0.22 * s_dsp + 0.28 * s_f0
            if octave > 0.5:
                n.flags.add("ambiguous_pitch")
                n.flags.add("octave_disagreement")
                conf *= 0.75
        if n.duration < 0.08:
            conf *= 0.88
        if "possible_harmonic" in n.flags:
            conf *= 0.8
        if s_dsp < 0.12 and c_model < 0.5:
            n.flags.add("weak_evidence")
        n.sources = {"basic_pitch" if ok else "dsp_primary": round(c_model, 3), "dsp": round(s_dsp, 3)}
        if s_f0 is not None:
            n.sources["f0"] = round(s_f0, 3)
        n.confidence = float(np.clip(conf, 0, 1))
        if n.confidence < 0.45 and "ambiguous_pitch" not in n.flags:
            n.flags.add("ambiguous_pitch")

    # notes the network missed but the DSP salience is very sure about
    added = 0
    for d in dsp_notes:
        if d.amplitude < 0.8 or d.duration < 0.09:
            continue
        clash = any(abs(n.pitch - d.pitch) in (0, 12, 19, 24) and n.overlaps(d) > 0.3 * d.duration for n in notes)
        if clash:
            continue
        d.confidence = 0.35 + 0.2 * min(1.0, d.amplitude - 0.8)
        d.flags.update({"single_source", "ambiguous_pitch"})
        d.sources = {"dsp": round(d.amplitude, 3)}
        notes.append(d)
        added += 1

    # f0-only notes in monophonic passages
    if track is not None:
        for seg in f0_tracker.segment_notes(track, voicing_thr=0.75):
            if not (lo <= seg["pitch"] <= hi) or seg["end"] - seg["start"] < 0.08:
                continue
            if any(n.overlaps(NoteEvent(seg["start"], seg["end"], seg["pitch"])) > 0.3 * (seg["end"] - seg["start"])
                   for n in notes if abs(n.pitch - seg["pitch"]) <= 1 or abs(n.pitch - seg["pitch"]) == 12):
                continue
            a = int(seg["start"] * SR / HOP)
            b = int(seg["end"] * SR / HOP)
            if poly[a:b].max(initial=0) > 0:
                continue
            nn = NoteEvent(seg["start"], seg["end"], seg["pitch"], amplitude=seg["voicing"],
                           confidence=0.4 * seg["voicing"], sources={"f0": round(seg["voicing"], 3)},
                           flags={"single_source"})
            notes.append(nn)
            added += 1

    # instrument range + weak-note rejection
    final = []
    for n in notes:
        if n.pitch < lo or n.pitch > hi:
            n.flags.add("rejected:out_of_range")
            rejected.append(n)
        elif n.confidence < params.get("min_confidence", 0.25):
            n.flags.add("rejected:too_weak")
            rejected.append(n)
        else:
            final.append(n)

    # attach the best available fine pitch contour (f0 is more precise when monophonic)
    poly = _polyphony(final, n_frames)
    for n in final:
        if track is None:
            continue
        a = int(n.start / track["hop"])
        b = max(a + 2, int(n.end / track["hop"]))
        if poly[a:b].max(initial=0) <= 1:
            seg = track["midi"][a:b]
            vv = track["voicing"][a:b]
            good = np.isfinite(seg) & (vv > 0.5)
            if good.mean() > 0.7:
                c = np.where(good, seg - n.pitch, np.nan)
                # fill gaps by interpolation, then discard octave jumps
                idx = np.arange(len(c))
                c = np.interp(idx, idx[good], c[good]).astype(np.float32)
                if np.all(np.abs(c) < 4):
                    n.contour = c
                    n.contour_t0 = a * track["hop"]
                    n.contour_hop = track["hop"]
                    n.sources["contour"] = "f0"

    # velocity from model amplitude & attack (0..127)
    for n in final:
        n.velocity = int(np.clip(40 + 87 * (0.6 * n.amplitude + 0.4 * n.onset_strength), 20, 127))

    final.sort(key=lambda n: (n.start, n.pitch))
    prog(1.0, f"{len(final)} notes ({added} recovered by secondary engines)")
    return {
        "notes": final,
        "rejected": rejected,
        "engines": engines,
        "f0": track,
        "salience": sal,
        "polyphony": _polyphony(final, n_frames),
        "stats": {
            "candidates": len(primary),
            "kept": len(final),
            "rejected": len(rejected),
            "recovered": added,
            "mean_confidence": float(np.mean([n.confidence for n in final])) if final else 0.0,
            "levels": {lvl: sum(1 for n in final if confidence_level(n.confidence) == lvl)
                       for lvl in ("high", "medium", "low")},
        },
    }
