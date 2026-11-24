"""Build and maintain the editable guitar track document.

A *track* is the JSON document the editor works on. It is produced from fused note
events and kept consistent after every edit by re-running the cheap stages (fingering,
technique confirmation, quantisation, review flags).
"""

from __future__ import annotations

import copy

import numpy as np

from ..transcription.base import NoteEvent, confidence_level
from .fingering import PROFILE_LABELS, FingeringOptimizer, finalize_techniques, playability_report
from .quantize import beat_to_time, build_measures, make_grid, measures_info, quantize
from .tunings import Tuning, midi_to_name, tuning_from_dict

FLAG_MESSAGES = {
    "ambiguous_pitch": "Pitch is uncertain (weak or conflicting evidence)",
    "octave_disagreement": "Engines disagree about the octave",
    "possible_harmonic": "May be a harmonic/overtone of a lower note rather than a played note",
    "uncertain_string": "Several string/fret positions are about equally plausible",
    "impossible_fingering": "Could not be fingered together with simultaneous notes (maybe a second part)",
    "unusual_timing": "Onset is far from the rhythmic grid (rubato, swing or a detection error)",
    "overlapping_instrument": "Another instrument plays this pitch at the same time (possible bleed)",
    "single_source": "Detected by only one engine",
    "weak_evidence": "Low energy in the spectrum at this pitch",
    "out_of_range": "Pitch is outside the range of the chosen tuning",
    "chord_disagreement": "Does not fit the chord detected from the full mix",
}


def note_to_dict(n: NoteEvent, idx: int) -> dict:
    d = {
        "id": f"n{idx}",
        "start": round(float(n.start), 4),
        "end": round(float(n.end), 4),
        "pitch": int(n.pitch),
        "name": midi_to_name(int(n.pitch)),
        "velocity": int(n.velocity),
        "confidence": round(float(n.confidence), 3),
        "level": confidence_level(n.confidence),
        "sources": n.sources,
        "flags": sorted(f for f in n.flags if not f.startswith("rejected:") and f not in ("no_onset", "merged_bend")),
        "techniques": n.techniques,
        "string": None,
        "fret": None,
        "locked": False,
        "edited": False,
    }
    if n.contour is not None and any(t["type"] in ("bend", "bend_release", "prebend_release", "vibrato", "slide")
                                     for t in n.techniques):
        step = max(1, int(round(0.02 / n.contour_hop)))
        d["contour"] = {"t0": round(n.contour_t0, 4), "hop": round(n.contour_hop * step, 5),
                        "values": [round(float(v), 2) for v in n.contour[::step]]}
    # pre-bend: the fretted pitch is lower than the sounding attack
    for t in n.techniques:
        if t["type"] == "prebend_release" and "fretted_pitch" in t:
            d["sounding_pitch"] = d["pitch"]
            d["pitch"] = int(t["fretted_pitch"])
            d["name"] = midi_to_name(d["pitch"])
    return d


def _flag(n: dict, f: str) -> None:
    if f not in n["flags"]:
        n["flags"].append(f)


def finalize(track: dict, rhythm: dict, duration: float, priors: dict | None = None,
             overlap_salience: dict | None = None, own_salience: dict | None = None, chords: list | None = None) -> dict:
    """(Re)compute fingering, alternatives, techniques, quantisation and review flags."""
    tuning = tuning_from_dict(track["tuning"])
    capo = int(track.get("capo", 0))
    eff = Tuning("effective", tuning.name, tuple(p + capo for p in tuning.notes), tuning.instrument)
    max_fret = int(track.get("max_fret", 24)) - capo
    notes = [n for n in track["notes"] if not n.get("dead")]
    playable = []
    for n in notes:
        n["flags"] = [f for f in n.get("flags", []) if f not in ("out_of_range", "impossible_fingering")]
        if n["pitch"] < eff.notes[0] or n["pitch"] > eff.notes[-1] + max_fret:
            _flag(n, "out_of_range")
            n["string"] = n["fret"] = None
        else:
            playable.append(n)
    profile = track.get("profile", "balanced")
    opt = FingeringOptimizer(eff, max_fret=max_fret, profile=profile, priors=priors)
    res = opt.optimize(playable)
    finalize_techniques(playable)

    # alternative whole-track fingerings for comparison
    alts = {}
    for prof in PROFILE_LABELS:
        if prof == profile:
            continue
        trial = copy.deepcopy(playable)
        FingeringOptimizer(eff, max_fret=max_fret, profile=prof, priors=priors).optimize(trial)
        changed = {t["id"]: [t["string"], t["fret"]] for t, o in zip(trial, playable)
                   if (t.get("string"), t.get("fret")) != (o.get("string"), o.get("fret"))}
        alts[prof] = {"label": PROFILE_LABELS[prof], "changes": changed, "n_changed": len(changed),
                      "report": playability_report(trial, eff)}
    track["alternative_fingerings"] = alts

    # rhythm grid
    ts = track.get("time_signature") or rhythm.get("time_signature", [4, 4])
    grid = make_grid(rhythm, duration, tuple(ts), int(track.get("downbeat_offset", 0)))
    qinfo = quantize(track["notes"], grid)
    n_measures = max([n.get("measure", 0) for n in track["notes"]] + [0]) + 1
    measures = measures_info(grid, qinfo.get("first_bar_beat", 0), n_measures)
    track["measures"] = measures
    track["notation"] = build_measures(track["notes"], grid, measures)
    track["grid"] = {"time_signature": grid["time_signature"], "beats_per_bar": grid["beats_per_bar"],
                     "ticks_per_beat": grid["ticks_per_beat"], "bpm": round(grid["bpm"], 2),
                     "first_bar_beat": qinfo.get("first_bar_beat", 0), "compound": grid["compound"]}
    track["time_signature"] = grid["time_signature"]

    # timing flags: onset far from its quantised position
    tpb = grid["ticks_per_beat"]
    beats = grid["beats"]
    fb = qinfo.get("first_bar_beat", 0)
    for n in track["notes"]:
        if n.get("tick") is None:
            continue
        qt = float(beat_to_time(fb + n["tick"] / tpb, beats))
        local_beat = float(beat_to_time(fb + n["tick"] / tpb + 1, beats)) - qt
        n["quantize_error"] = round(n["start"] - qt, 4)
        if abs(n["start"] - qt) > 0.18 * max(local_beat, 0.2):
            _flag(n, "unusual_timing")
        elif "unusual_timing" in n["flags"]:
            n["flags"].remove("unusual_timing")

    # instrument overlap (bleed) flags
    if overlap_salience is not None and own_salience is not None:
        from ..transcription.dsp_poly import support  # noqa: PLC0415

        for n in track["notes"]:
            other = support(overlap_salience, n["pitch"], n["start"], n["end"])
            own = support(own_salience, n["pitch"], n["start"], n["end"])
            if other > 0.55 and other > own:
                _flag(n, "overlapping_instrument")

    if chords:
        for n in track["notes"]:
            c = next((c for c in chords if c["start"] <= n["start"] < c["end"]), None)
            if c and c.get("root") is not None and c.get("confidence", 0) > 0.6 and n["confidence"] < 0.6:
                from ..analysis.harmony import QUALITIES  # noqa: PLC0415

                pcs = {(c["root"] + iv) % 12 for iv in QUALITIES[c["quality"]]}
                if n["pitch"] % 12 not in pcs and n["end"] - n["start"] > 0.25:
                    _flag(n, "chord_disagreement")

    for n in track["notes"]:
        n["level"] = confidence_level(n["confidence"]) if not n.get("edited") else "user"
        n["flags"].sort()

    track["review"] = build_review(track)
    track["playability"] = playability_report(playable, eff)
    track["fingering"] = {"total_cost": res["total_cost"], "groups": res["groups"],
                          "uncertain_groups": res.get("uncertain_groups", 0), "profile": profile,
                          "profile_label": PROFILE_LABELS.get(profile, profile)}
    track["stats"] = {
        "notes": len(track["notes"]),
        "levels": {lvl: sum(1 for n in track["notes"] if n["level"] == lvl) for lvl in ("high", "medium", "low", "user")},
        "flagged": sum(1 for n in track["notes"] if n["flags"]),
        "mean_confidence": round(float(np.mean([n["confidence"] for n in track["notes"]])), 3) if track["notes"] else 0,
        "techniques": _technique_counts(track["notes"]),
    }
    track["version"] = int(track.get("version", 0)) + 1
    return track


def _technique_counts(notes: list[dict]) -> dict:
    out: dict[str, int] = {}
    for n in notes:
        for t in n.get("techniques", []):
            out[t["type"]] = out.get(t["type"], 0) + 1
    return out


def build_review(track: dict) -> list[dict]:
    items = []
    for n in track["notes"]:
        for f in n.get("flags", []):
            if f in FLAG_MESSAGES:
                items.append({"note_id": n["id"], "time": n["start"], "flag": f, "message": FLAG_MESSAGES[f],
                              "severity": "high" if f in ("impossible_fingering", "out_of_range", "ambiguous_pitch")
                              else "medium" if f in ("uncertain_string", "possible_harmonic", "overlapping_instrument",
                                                     "octave_disagreement", "chord_disagreement") else "low"})
    items.sort(key=lambda x: x["time"])
    return items


def build_track(track_id: str, name: str, stem: str, instrument: str, role: str, notes: list[NoteEvent],
                rejected: list[NoteEvent], tuning: Tuning, rhythm: dict, duration: float, engines: dict,
                extra: dict | None = None, priors: dict | None = None, **finalize_kw) -> dict:
    nd = [note_to_dict(n, i) for i, n in enumerate(sorted(notes, key=lambda n: (n.start, n.pitch)))]
    rej = [{"start": round(n.start, 3), "end": round(n.end, 3), "pitch": n.pitch, "confidence": round(n.confidence, 3),
            "reason": next((f.split(":", 1)[1] for f in n.flags if f.startswith("rejected:")), "rejected")}
           for n in rejected]
    track = {
        "id": track_id,
        "name": name,
        "stem": stem,
        "instrument": instrument,
        "role": role,
        "tuning": tuning.to_dict(),
        "capo": 0,
        "max_fret": 24 if instrument == "guitar" else 20,
        "profile": "balanced",
        "notes": nd,
        "rejected": rej,
        "engines": engines,
        "version": 0,
        **(extra or {}),
    }
    return finalize(track, rhythm, duration, priors=priors, **finalize_kw)
