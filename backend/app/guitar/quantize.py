"""Rhythm quantisation: seconds -> beats -> ticks -> measures -> notated beats.

* Notes are mapped to *fractional beat positions* by interpolating between the tracked
  beat times, so tempo drift in a live recording does not break the grid.
* Each beat chooses its own subdivision (straight 8ths/16ths/32nds or triplets/sextuplets)
  by minimising onset displacement plus a small complexity penalty: a triplet is only
  chosen when the onsets genuinely sit on triplet positions.
* Ticks use 960 per quarter note (Guitar Pro / MIDI convention).
* ``build_measures`` turns quantised notes into a single tab voice per measure:
  a sequence of beats (chords or rests) whose durations are decomposed into notatable
  values with ties across bar lines.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..analysis.rhythm import build_beat_grid

TPQ = 960  # ticks per quarter note

SIMPLE_SUBDIVS = [(1, 0.0), (2, 0.0), (4, 0.01), (3, 0.05), (8, 0.06), (6, 0.08)]
COMPOUND_SUBDIVS = [(1, 0.0), (3, 0.0), (6, 0.01), (2, 0.06), (12, 0.06), (4, 0.08)]


def make_grid(rhythm: dict, duration: float, time_signature: tuple[int, int] | None = None,
              downbeat_offset: int = 0) -> dict:
    ts = tuple(time_signature or rhythm.get("time_signature", (4, 4)))
    num, den = int(ts[0]), int(ts[1])
    compound = den == 8 and num % 3 == 0 and num > 3
    beats_per_bar = num // 3 if compound else num
    ticks_per_beat = int(TPQ * 4 / den * (3 if compound else 1))
    beats = build_beat_grid(rhythm.get("beats", []), duration, rhythm.get("bpm", 120.0))
    # locate the downbeat phase in the extended grid
    downbeats = rhythm.get("downbeats") or []
    if downbeats:
        first_db = downbeats[0]
        idx0 = int(np.argmin(np.abs(beats - first_db)))
    else:
        idx0 = 0
    idx0 = (idx0 + downbeat_offset) % beats_per_bar
    return {
        "beats": beats,
        "bar_phase": idx0,
        "beats_per_bar": beats_per_bar,
        "ticks_per_beat": ticks_per_beat,
        "time_signature": [num, den],
        "compound": compound,
        "bpm": float(rhythm.get("bpm", 120.0)),
    }


def time_to_beat(t: float | np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Fractional beat index; extrapolates linearly outside the grid (pickups, tails)."""
    idx = np.arange(len(beats), dtype=float)
    t = np.asarray(t, dtype=float)
    out = np.interp(t, beats, idx)
    if len(beats) >= 2:
        p0 = beats[1] - beats[0]
        p1 = beats[-1] - beats[-2]
        out = np.where(t < beats[0], (t - beats[0]) / p0, out)
        out = np.where(t > beats[-1], idx[-1] + (t - beats[-1]) / p1, out)
    return out


def beat_to_time(b: float | np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Time of a fractional beat index; extrapolates linearly outside the grid."""
    idx = np.arange(len(beats), dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.interp(b, idx, beats)
    if len(beats) >= 2:
        out = np.where(b < 0, beats[0] + b * (beats[1] - beats[0]), out)
        out = np.where(b > idx[-1], beats[-1] + (b - idx[-1]) * (beats[-1] - beats[-2]), out)
    return out


def quantize(notes: list[dict], grid: dict, force_first_bar: int | None = None) -> dict:
    """Adds ``tick``, ``dur_ticks``, ``measure`` and ``beat`` to each note (mutates)."""
    beats = grid["beats"]
    tpb = grid["ticks_per_beat"]
    bpb = grid["beats_per_bar"]
    subdivs = COMPOUND_SUBDIVS if grid["compound"] else SIMPLE_SUBDIVS
    if not notes:
        return {"first_bar_beat": grid["bar_phase"], "subdivisions": {}}
    on_b = time_to_beat(np.array([n["start"] for n in notes]), beats)
    off_b = time_to_beat(np.array([n["end"] for n in notes]), beats)

    # choose a subdivision per beat
    beat_ids = np.floor(on_b + 1e-6).astype(int)
    choice: dict[int, int] = {}
    for b in np.unique(beat_ids):
        fr = on_b[beat_ids == b] - b
        best = None
        for sub, pen in subdivs:
            err = np.abs(fr * sub - np.round(fr * sub)) / sub
            score = float(err.mean()) + pen
            if best is None or score < best[0] - 1e-9:
                best = (score, sub)
        choice[int(b)] = best[1]

    # bar 0 starts at the last bar line at or before the first note
    phase = grid["bar_phase"]
    first_beat = float(on_b.min())
    k = int(np.floor((first_beat - phase) / bpb))
    first_bar_beat = phase + k * bpb if force_first_bar is None else int(force_first_bar)

    def q_on(bpos: float) -> Fraction:
        b = int(np.floor(bpos + 1e-6))
        sub = choice.get(b, 4)
        return Fraction(b) + Fraction(int(round((bpos - b) * sub)), sub)

    for n, ob, fb in zip(notes, on_b, off_b):
        qb = q_on(float(ob))
        b_end = int(np.floor(fb + 1e-6))
        sub_end = choice.get(b_end, 4 if not grid["compound"] else 3)
        qe = Fraction(b_end) + Fraction(int(round((fb - b_end) * sub_end)), sub_end)
        min_len = Fraction(1, max(choice.get(int(qb), 4), 2))
        if qe - qb < min_len:
            qe = qb + min_len
        rel = qb - first_bar_beat
        n["beat"] = round(float(ob), 4)
        n["tick"] = int(round(float(rel) * tpb))
        n["dur_ticks"] = max(1, int(round(float(qe - qb) * tpb)))
        n["measure"] = int(np.floor(float(rel) / bpb + 1e-9))
    return {"first_bar_beat": int(first_bar_beat), "subdivisions": {str(k): v for k, v in choice.items()}}


def measures_info(grid: dict, first_bar_beat: int, n_measures: int) -> list[dict]:
    beats = grid["beats"]
    bpb = grid["beats_per_bar"]
    out = []
    for m in range(n_measures):
        b0 = first_bar_beat + m * bpb
        t0 = float(beat_to_time(b0, beats))
        t1 = float(beat_to_time(b0 + bpb, beats))
        out.append({"index": m, "start": round(t0, 4), "end": round(t1, 4), "start_tick": m * bpb * grid["ticks_per_beat"],
                    "ticks": bpb * grid["ticks_per_beat"], "time_signature": grid["time_signature"],
                    "bpm": round(60.0 * bpb / max(1e-3, t1 - t0), 2)})
    return out


# ----------------------------------------------------------------------------- notation

BASE_VALUES = [(1, 3840), (2, 1920), (4, 960), (8, 480), (16, 240), (32, 120), (64, 60)]


def _options() -> tuple[list[dict], list[dict]]:
    plain, tuplets = [], []
    for val, t in BASE_VALUES:
        plain.append({"value": val, "dotted": False, "tuplet": None, "ticks": t})
        if val <= 32:
            plain.append({"value": val, "dotted": True, "tuplet": None, "ticks": t * 3 // 2})
        tuplets.append({"value": val, "dotted": False, "tuplet": [3, 2], "ticks": t * 2 // 3})
    plain.sort(key=lambda o: -o["ticks"])
    tuplets.sort(key=lambda o: -o["ticks"])
    return plain, tuplets


_PLAIN, _TUPLETS = _options()


def decompose(ticks: int) -> list[dict]:
    """Split a tick duration into notatable values (dotted values, triplets), largest first."""
    out = []
    rem = int(ticks)
    for _ in range(24):
        if rem <= 0:
            break
        if rem % 60 == 0:
            pick = next((o for o in _PLAIN if o["ticks"] <= rem), None)
        else:
            # a triplet value whose remainder is plain-notatable, else the largest triplet
            pick = next((o for o in _TUPLETS if o["ticks"] <= rem and (rem - o["ticks"]) % 60 == 0), None)
            if pick is None:
                pick = next((o for o in _TUPLETS if o["ticks"] <= rem), None)
        if pick is None:  # sub-64th residue: absorb into the previous value
            if out:
                out[-1]["ticks"] += rem
            break
        out.append(dict(pick))
        rem -= pick["ticks"]
    return out


def build_measures(notes: list[dict], grid: dict, measures: list[dict]) -> list[dict]:
    """Single-voice notated beats per measure: [{'beats': [{tick, ticks, value, dotted,
    tuplet, notes:[{id, tie}], rest}]}]."""
    tpb = grid["ticks_per_beat"]
    bar_ticks = grid["beats_per_bar"] * tpb
    by_tick: dict[int, list[dict]] = {}
    for n in notes:
        if n.get("tick") is None:
            continue
        by_tick.setdefault(n["tick"], []).append(n)
    # a string can sound only one note per notated beat: when two land on the same tick and
    # string (e.g. a fast hammer-on quantised onto its source note), keep the later one, which is
    # what is heard at this rhythmic resolution. Writers such as GP5 cannot store both.
    for t, ns in by_tick.items():
        seen: dict[int, dict] = {}
        for n in sorted(ns, key=lambda n: n["start"]):
            s = n.get("string")
            if s is None:
                continue
            seen[s] = n
        by_tick[t] = [n for n in ns if n.get("string") is None or seen.get(n["string"]) is n]
    onsets = sorted(by_tick)
    total_ticks = len(measures) * bar_ticks
    events = []  # (tick, duration, note_ids)
    for i, t in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else min(total_ticks, t + max(n["dur_ticks"] for n in by_tick[t]))
        longest = max(n["dur_ticks"] for n in by_tick[t])
        dur = min(nxt - t, max(longest, 1)) if nxt > t else longest
        events.append((t, dur, by_tick[t]))
    result = []
    for m in measures:
        m0 = m["start_tick"]
        m1 = m0 + m["ticks"]
        beats = []
        cursor = m0
        for t, dur, ns in events:
            end = t + dur
            if end <= m0 or t >= m1:
                continue
            seg_start = max(t, m0)
            seg_end = min(end, m1)
            if seg_start > cursor:
                for part in decompose(seg_start - cursor):
                    beats.append({"tick": cursor - m0, **part, "rest": True, "notes": []})
                    cursor += part["ticks"]
                cursor = seg_start
            first = True
            for part in decompose(seg_end - seg_start):
                tie = (seg_start > t) or not first
                beats.append({"tick": cursor - m0, **part, "rest": False,
                              "notes": [{"id": n["id"], "tie": tie} for n in ns]})
                cursor += part["ticks"]
                first = False
            cursor = seg_end
        if cursor < m1:
            for part in decompose(m1 - cursor):
                beats.append({"tick": cursor - m0, **part, "rest": True, "notes": []})
                cursor += part["ticks"]
        result.append({"index": m["index"], "beats": beats, "time_signature": m["time_signature"]})
    return result
