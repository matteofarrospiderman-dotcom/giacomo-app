"""Helpers shared by all exporters."""

from __future__ import annotations

import copy

from ..guitar.quantize import build_measures, make_grid, measures_info, quantize
from ..guitar.tunings import tuning_from_dict

SHARP_STEPS = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1), ("A", 0),
               ("A", 1), ("B", 0)]
FLAT_STEPS = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0), ("G", -1), ("G", 0), ("A", -1), ("A", 0),
              ("B", -1), ("B", 0)]
# circle-of-fifths position of major keys by tonic pitch class
MAJOR_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}


def key_fifths(key: dict | None) -> tuple[int, str]:
    if not key or key.get("tonic") is None:
        return 0, "major"
    tonic = int(key["tonic"])
    if key.get("mode") == "minor":
        rel_major = (tonic + 3) % 12
        return MAJOR_FIFTHS[rel_major], "minor"
    return MAJOR_FIFTHS[tonic], "major"


def spell(pitch: int, flats: bool) -> tuple[str, int, int]:
    step, alter = (FLAT_STEPS if flats else SHARP_STEPS)[pitch % 12]
    return step, alter, pitch // 12 - 1


def aligned_tracks(tracks: list[dict], rhythm: dict, duration: float) -> tuple[list[dict], dict, list[dict]]:
    """Re-quantise all tracks on the first track's grid so multi-track files share bar lines."""
    if not tracks:
        return [], {}, []
    ref = tracks[0]
    ts = tuple(ref.get("time_signature") or rhythm.get("time_signature", [4, 4]))
    grid = make_grid(rhythm, duration, ts, int(ref.get("downbeat_offset", 0)))
    out = []
    first_bar = None
    for t in tracks:
        t2 = copy.deepcopy(t)
        info = quantize(t2["notes"], grid)
        fb = info.get("first_bar_beat", 0)
        first_bar = fb if first_bar is None else min(first_bar, fb)
        out.append(t2)
    # re-quantise against the common first bar
    for t2 in out:
        quantize(t2["notes"], grid, force_first_bar=first_bar)
    n_measures = max([n.get("measure", 0) for t in out for n in t["notes"]] + [0]) + 1
    measures = measures_info(grid, first_bar or 0, n_measures)
    for t2 in out:
        t2["notation"] = build_measures(t2["notes"], grid, measures)
        t2["_tuning"] = tuning_from_dict(t2["tuning"])
    return out, grid, measures


def note_index(track: dict) -> dict:
    return {n["id"]: n for n in track["notes"]}


def tech(n: dict, kind: str) -> dict | None:
    for t in n.get("techniques", []):
        if t["type"] == kind:
            return t
    return None


def has(n: dict, *kinds: str) -> bool:
    return any(t["type"] in kinds for t in n.get("techniques", []))


GM_PROGRAMS = {"electric_clean": 27, "electric_overdrive": 29, "electric_distortion": 30, "acoustic": 25, "bass": 33}


def gm_program(track: dict, tone: dict | None) -> int:
    if track.get("instrument") == "bass":
        return GM_PROGRAMS["bass"]
    gain = (tone or {}).get("gain", {}).get("gain_index", 3)
    if gain >= 6.5:
        return GM_PROGRAMS["electric_distortion"]
    if gain >= 3.5:
        return GM_PROGRAMS["electric_overdrive"]
    return GM_PROGRAMS["electric_clean"]
