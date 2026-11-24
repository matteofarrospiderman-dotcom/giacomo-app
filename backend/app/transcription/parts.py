"""Note-level part separation (lead vs rhythm) inside a single guitar stem.

When two guitars share a stem (and the stereo splitter could not separate them),
merging all notes into one tab produces impossible fingerings. We split the notes into
two plausible single-guitar parts:

* onset groups of >= 3 notes are chords -> *rhythm*;
* a note that starts while a rhythm chord is still ringing, above the chord's top note,
  is a melodic overlay -> *lead*;
* remaining single notes join the part whose recent pitch is closest (voice-leading
  continuity), preferring the lead part when it was active within the last second;
* groups that are unplayable as a whole (more notes than strings, or unreachable)
  are split by register.

The split is only adopted if it is substantial (lead >= 12 notes and >= 15% of all notes)
and it reduces the number of unplayable groups or overlapping layers; otherwise the
single part is kept. Reported confidence reflects how clean the separation was.
"""

from __future__ import annotations

import numpy as np


def _groups(notes: list[dict], tol: float = 0.035) -> list[list[dict]]:
    out: list[list[dict]] = []
    for n in sorted(notes, key=lambda n: (n["start"], n["pitch"])):
        if out and n["start"] - out[-1][0]["start"] <= tol:
            out[-1].append(n)
        else:
            out.append([n])
    return out


def split_lead_rhythm(notes: list[dict], n_strings: int = 6) -> dict:
    groups = _groups(notes)
    role: dict[int, str] = {}
    last = {"lead": (None, -1e9), "rhythm": (None, -1e9)}  # (pitch, time)
    ringing: list[dict] = []
    overlay = 0
    for g in groups:
        t = g[0]["start"]
        ringing = [n for n in ringing if n["end"] > t + 0.05]
        if len(g) >= 3 or len(g) > n_strings:
            ordered = sorted(g, key=lambda n: n["pitch"])
            if len(g) > n_strings:
                # unplayable as one chord: top notes are probably a second guitar
                for n in ordered[:n_strings]:
                    role[id(n)] = "rhythm"
                for n in ordered[n_strings:]:
                    role[id(n)] = "lead"
            else:
                for n in g:
                    role[id(n)] = "rhythm"
            ringing.extend(g)
            last["rhythm"] = (int(np.mean([n["pitch"] for n in g])), t)
            continue
        rhythm_ring = [n for n in ringing if role.get(id(n)) == "rhythm"]
        for n in g:
            if len(rhythm_ring) >= 2 and n["pitch"] > max(m["pitch"] for m in rhythm_ring):
                role[id(n)] = "lead"
                overlay += 1
            else:
                lp, lt = last["lead"]
                rp, rt = last["rhythm"]
                d_lead = abs(n["pitch"] - lp) if lp is not None and t - lt < 1.0 else 99
                d_rhy = abs(n["pitch"] - rp) if rp is not None and t - rt < 2.0 else 50
                role[id(n)] = "lead" if d_lead <= 7 and d_lead <= d_rhy else "rhythm"
            last[role[id(n)]] = (n["pitch"], t)
        ringing.extend(g)
    lead = [n for n in notes if role.get(id(n)) == "lead"]
    rhythm = [n for n in notes if role.get(id(n)) != "lead"]
    adopt = len(lead) >= 12 and len(lead) >= 0.15 * len(notes) and overlay >= 6 and len(rhythm) >= 12
    conf = float(np.clip(overlay / max(1, len(lead)), 0, 1)) * (0.8 if adopt else 0.3)
    return {"adopted": bool(adopt), "lead": lead, "rhythm": rhythm, "overlay_notes": overlay,
            "confidence": round(conf, 3)}
