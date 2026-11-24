"""Tuning identification.

Two separate questions:

1. **Reference pitch**: is the whole recording offset from A440 (e.g. tuned to 432 Hz,
   varispeed, or simply a guitar 15 cents flat)? Estimated from the deviation of all
   spectral peaks from the equal-tempered grid (``librosa.estimate_tuning``).
2. **String tuning** (standard, Eb, drop D, ...). We use *analysis-by-synthesis*: for
   each candidate tuning we actually fingerprint the transcription onto that tuning with
   the fingering optimiser and score how natural the result is:

   * notes below the lowest open string are impossible (strong penalty);
   * the average fingering cost per note (stretches, jumps) - a song written in drop D
     is markedly easier to play in drop D;
   * open-string evidence: long ringing notes and frequent chord roots that coincide
     with the candidate's open strings (players gravitate to open strings, and riffs in
     drop tunings hammer the lowest open string);
   * the key's tonic matching the lowest open string (common in rock and metal);
   * a prior reflecting how common each tuning is.

   Scores go through a softmax to give a ranked list with probabilities.

Limitation: when a part only uses notes that exist in several tunings (e.g. a lead
line above the 5th fret) the tunings are genuinely indistinguishable from audio; the
ranking then mostly reflects the prior and is reported with low confidence.
"""

from __future__ import annotations

import numpy as np

from ..guitar.fingering import FingeringOptimizer
from ..guitar.tunings import Tuning, candidate_tunings, midi_to_name


def _sample(notes: list[dict], max_n: int = 500) -> list[dict]:
    if len(notes) <= max_n:
        return notes
    # keep contiguous chunks so transition costs remain meaningful
    chunk = 50
    starts = np.linspace(0, len(notes) - chunk, max_n // chunk).astype(int)
    out = []
    for s in starts:
        out.extend(notes[s: s + chunk])
    return out


def detect_tuning(notes: list[dict], instrument: str = "guitar", strings: int | None = None,
                  key: dict | None = None, reference_cents: float = 0.0) -> dict:
    cands = candidate_tunings(instrument, strings)
    if instrument == "guitar" and strings is None:
        cands = [t for t in cands if t.strings <= 7] + [t for t in cands if t.strings == 8]
    usable = [dict(n) for n in notes if n.get("confidence", 0.5) >= 0.35 and not n.get("dead")]
    if len(usable) < 8:
        default = cands[0]
        return {"best": default.to_dict(), "confidence": 0.15, "ranking": [{"tuning": default.to_dict(), "probability": 1.0}],
                "reference": _reference(reference_cents), "evidence": ["Too few confident notes to infer tuning"]}
    sample = _sample(sorted(usable, key=lambda n: n["start"]))
    pitches = np.array([n["pitch"] for n in usable])
    durs = np.array([n["end"] - n["start"] for n in usable])
    weights = np.array([n.get("confidence", 0.5) for n in usable])
    lowest_common = int(np.percentile(pitches, 2))

    # chord roots: lowest pitch of each onset group
    roots = []
    for g in _groups(usable):
        if len(g) >= 2:
            roots.append(min(n["pitch"] for n in g))
    roots = np.array(roots) if roots else pitches[:0]

    results = []
    for t in cands:
        lowest = t.notes[0]
        below = weights[pitches < lowest].sum() / weights.sum()
        too_high = weights[pitches > t.notes[-1] + 24].sum() / weights.sum()
        opt = FingeringOptimizer(t, max_fret=24)
        trial = [dict(n, techniques=n.get("techniques", [])) for n in sample if n["pitch"] >= lowest]
        res = opt.optimize(trial) if trial else {"total_cost": 0.0}
        cost_per_note = res["total_cost"] / max(1, len(trial))
        open_set = set(t.notes)
        long_mask = durs > 0.45
        open_long = float(np.mean(np.isin(pitches[long_mask], list(open_set)))) if long_mask.any() else 0.0
        low_open_hits = float(np.mean(roots == lowest)) if len(roots) else float(np.mean(pitches == lowest))
        tonic_bonus = 0.0
        if key and key.get("tonic") is not None and key["tonic"] == lowest % 12:
            tonic_bonus = 0.35
        score = (-12.0 * below - 2.0 * too_high - 0.9 * cost_per_note + 1.4 * open_long + 2.2 * low_open_hits
                 + tonic_bonus + 0.6 * np.log(t.prior))
        results.append({"tuning": t, "score": float(score), "below_range": round(float(below), 3),
                        "cost_per_note": round(float(cost_per_note), 3), "open_long": round(open_long, 3),
                        "low_open_hits": round(low_open_hits, 3)})
    scores = np.array([r["score"] for r in results])
    p = np.exp((scores - scores.max()) * 1.5)
    p /= p.sum()
    order = np.argsort(-p)
    ranking = [{"tuning": results[i]["tuning"].to_dict(), "probability": round(float(p[i]), 3),
                "evidence": {k: v for k, v in results[i].items() if k not in ("tuning",)}} for i in order[:6]]
    best = results[order[0]]
    conf = float(p[order[0]])
    evidence = [
        f"Lowest frequently played note: {midi_to_name(lowest_common)}",
        f"{best['low_open_hits'] * 100:.0f}% of chord roots fall on the lowest open string of {best['tuning'].name}",
        f"Average fingering cost {best['cost_per_note']:.2f} per note",
    ]
    if best["below_range"] > 0:
        evidence.append(f"{best['below_range'] * 100:.1f}% of notes fall below this tuning's range (likely detection errors)")
    return {"best": best["tuning"].to_dict(), "confidence": round(conf, 3), "ranking": ranking,
            "reference": _reference(reference_cents), "evidence": evidence}


def _groups(notes: list[dict], tol: float = 0.035) -> list[list[dict]]:
    out: list[list[dict]] = []
    for n in sorted(notes, key=lambda n: n["start"]):
        if out and n["start"] - out[-1][0]["start"] <= tol:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def _reference(cents: float) -> dict:
    a4 = 440.0 * 2 ** (cents / 1200)
    note = None
    if abs(cents) > 40:
        note = ("Recording is close to a quarter-tone off A440: the tuning could be either the named tuning or one a "
                "semitone away. Check by ear against the stems.")
    elif abs(cents) > 8:
        note = f"Instruments are tuned about {cents:+.0f} cents from standard pitch (A4 ≈ {a4:.1f} Hz)."
    return {"cents": round(cents, 1), "a4_hz": round(a4, 2), "note": note}


def apply_tuning_offset(tuning: Tuning, semitones: int) -> list[int]:
    return [n + semitones for n in tuning.notes]
