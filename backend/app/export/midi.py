"""Standard MIDI File export.

* A full **tempo map** is written (one tempo event per tracked beat) so the MIDI grid
  follows the recording's actual timing; notes keep their exact performed onsets.
  ``quantized=True`` instead writes a constant tempo and grid-snapped notes.
* Guitar tracks can use **one MIDI channel per string** (the convention of MIDI
  guitars and Guitar Pro) so bends on one string do not bend the others.
* Bends/slides/vibrato become pitch-bend curves (bend range set to +-4 semitones via
  RPN 0); palm mutes shorten notes; dead notes become very short, quiet notes.
"""

from __future__ import annotations

import io

import mido
import numpy as np

from ..guitar.quantize import make_grid, time_to_beat
from .common import gm_program

PPQ = 480
BEND_RANGE = 4


def _bend_curve(n: dict) -> list[tuple[float, float]]:
    """(time_s, semitones) pairs for pitch-bend messages."""
    c = n.get("contour")
    if c and any(t["type"] in ("bend", "bend_release", "prebend_release", "vibrato") for t in n.get("techniques", [])):
        vals = c["values"]
        offset = float(n.get("sounding_pitch", n["pitch"]) - n["pitch"])  # pre-bends: contour is relative to the attack
        return [(c["t0"] + i * c["hop"], float(v) + offset) for i, v in enumerate(vals)]
    for t in n.get("techniques", []):
        dur = n["end"] - n["start"]
        st = float(t.get("semitones", 1))
        if t["type"] == "bend":
            return [(n["start"], 0.0), (n["start"] + 0.3 * dur, st), (n["end"], st)]
        if t["type"] == "bend_release":
            return [(n["start"], 0.0), (n["start"] + 0.3 * dur, st), (n["start"] + 0.6 * dur, st), (n["end"], 0.0)]
        if t["type"] == "prebend_release":
            return [(n["start"], st), (n["start"] + 0.4 * dur, st), (n["end"], 0.0)]
    return []


def _pb_value(semitones: float) -> int:
    return int(np.clip(round(semitones / BEND_RANGE * 8191), -8192, 8191))


def export_midi(tracks: list[dict], analysis: dict, quantized: bool = False, per_string: bool | None = None,
                tones: dict | None = None) -> bytes:
    rhythm = analysis["rhythm"]
    duration = analysis["duration"]
    ref = tracks[0] if tracks else {}
    ts = tuple(ref.get("time_signature") or rhythm.get("time_signature", [4, 4]))
    grid = make_grid(rhythm, duration, ts)
    beats = grid["beats"]
    qn_per_beat = grid["ticks_per_beat"] / 960.0
    b0 = float(time_to_beat(0.0, beats))
    per_string = (len(tracks) == 1) if per_string is None else per_string

    snap: dict[str, tuple[int, int]] = {}
    if quantized:
        # constant tempo + notes snapped to the rhythmic grid (bar 0 starts at tick 0)
        from ..guitar.quantize import quantize  # noqa: PLC0415

        tracks = [dict(t, notes=[dict(n) for n in t["notes"]]) for t in tracks]
        firsts = [quantize(t["notes"], grid).get("first_bar_beat", 0) for t in tracks]
        first = min(firsts) if firsts else 0
        for t in tracks:
            quantize(t["notes"], grid, force_first_bar=first)
            for n in t["notes"]:
                if n.get("tick") is not None:
                    snap[n["id"]] = (int(n["tick"] * PPQ / 960), int(n["dur_ticks"] * PPQ / 960))

    def to_tick(t: float) -> int:
        if quantized:
            return int(round(t * grid["bpm"] / 60.0 * qn_per_beat * PPQ))
        return int(round((float(time_to_beat(t, beats)) - b0) * qn_per_beat * PPQ))

    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name="Tempo map", time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=int(ts[0]), denominator=int(ts[1]), time=0))
    events = []
    if quantized:
        events.append((0, mido.bpm2tempo(grid["bpm"] * qn_per_beat)))
    else:
        k0 = int(np.ceil(b0))
        first = beats[k0 + 1] - beats[k0] if k0 + 1 < len(beats) else 60 / grid["bpm"]
        events.append((0, int(first / qn_per_beat * 1e6)))
        for k in range(k0, len(beats) - 1):
            if beats[k] > duration + 1:
                break
            tick = int(round((k - b0) * qn_per_beat * PPQ))
            us = int((beats[k + 1] - beats[k]) / qn_per_beat * 1e6)
            events.append((max(0, tick), max(150000, min(4000000, us))))
    last = 0
    for tick, us in events:
        meta.append(mido.MetaMessage("set_tempo", tempo=us, time=tick - last))
        last = tick

    next_channel = 0

    def alloc() -> int:
        nonlocal next_channel
        ch = next_channel
        next_channel += 1
        if next_channel == 9:
            next_channel = 10
        return ch % 16

    for tr in tracks:
        mt = mido.MidiTrack()
        mid.tracks.append(mt)
        mt.append(mido.MetaMessage("track_name", name=tr["name"][:40], time=0))
        program = gm_program(tr, (tones or {}).get(tr.get("stem")))
        n_strings = len(tr["tuning"]["notes"])
        channels = {s: alloc() for s in range(1, n_strings + 1)} if per_string else None
        shared = None if per_string else alloc()
        msgs = []
        used = set(channels.values()) if channels else {shared}
        for ch in used:
            msgs.append((0, 0, mido.Message("program_change", channel=ch, program=program)))
            for ctrl, val in ((101, 0), (100, 0), (6, BEND_RANGE), (38, 0)):
                msgs.append((0, 1, mido.Message("control_change", channel=ch, control=ctrl, value=val)))
        for n in tr["notes"]:
            if n.get("start") is None:
                continue
            ch = channels.get(n.get("string") or 1, shared) if channels else shared
            pitch = int(n["pitch"])  # the fretted pitch; bends are rendered as pitch-wheel curves
            start, end = float(n["start"]), float(n["end"])
            vel = int(n.get("velocity", 90))
            if n.get("dead"):
                end, vel = start + 0.05, 40
            elif any(t["type"] == "palm_mute" for t in n.get("techniques", [])):
                end = min(end, start + 0.18)
            if quantized and n.get("id") in snap:
                t_on, dur = snap[n["id"]]
                t_off = t_on + max(1, dur)
            else:
                t_on, t_off = to_tick(start), max(to_tick(start) + 1, to_tick(end))
            curve = _bend_curve(n)
            if curve:
                for t, v in curve:
                    msgs.append((to_tick(max(start, min(end, t))), 2, mido.Message("pitchwheel", channel=ch, pitch=_pb_value(v))))
                msgs.append((t_off, 0, mido.Message("pitchwheel", channel=ch, pitch=0)))
            msgs.append((t_on, 3, mido.Message("note_on", channel=ch, note=max(0, min(127, pitch)), velocity=max(1, min(127, vel)))))
            msgs.append((t_off, 1, mido.Message("note_off", channel=ch, note=max(0, min(127, pitch)), velocity=0)))
        msgs.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tick, _, msg in msgs:
            msg.time = max(0, tick - last)
            last = max(last, tick)
            mt.append(msg)
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()
