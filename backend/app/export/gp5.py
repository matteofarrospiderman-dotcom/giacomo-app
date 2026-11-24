"""Guitar Pro 5 (.gp5) export via PyGuitarPro.

GP5 is the most widely supported Guitar Pro format: it opens in Guitar Pro 5-8,
TuxGuitar, MuseScore and Songsterr. GP6/7 formats (.gpx/.gp) are proprietary,
compressed containers without a maintained open writer, so GP5 is the reliable choice;
Guitar Pro 7/8 converts it on open.

Preserved: tuning (per track, any number of strings), capo, tempo, time signature,
key, durations incl. dots/triplets and ties, string/fret, velocity, hammer-on/pull-off,
slides (shift/legato/in/out), bends (bend, bend-release, pre-bend release) with
points, vibrato, palm mute, let ring, dead notes, tremolo picking, natural harmonics,
strum direction.
"""

from __future__ import annotations

import io

import guitarpro as gp
from guitarpro import models as m

from .common import aligned_tracks, gm_program, key_fifths, note_index, tech

KEY_BY_FIFTHS_MAJOR = {k.value[0]: k for k in m.KeySignature if k.value[1] == 0}
KEY_BY_FIFTHS_MINOR = {k.value[0]: k for k in m.KeySignature if k.value[1] == 1}


def _bend(t: dict) -> m.BendEffect:
    st = float(t.get("semitones", 1.0))
    if t["type"] == "bend_release":
        pts = [m.BendPoint(0, 0), m.BendPoint(3, st), m.BendPoint(6, st), m.BendPoint(9, 0), m.BendPoint(12, 0)]
        kind = m.BendType.bendRelease
    elif t["type"] == "prebend_release":
        pts = [m.BendPoint(0, st), m.BendPoint(4, st), m.BendPoint(8, 0), m.BendPoint(12, 0)]
        kind = m.BendType.prebendRelease
    else:
        pts = [m.BendPoint(0, 0), m.BendPoint(4, st), m.BendPoint(12, st)]
        kind = m.BendType.bend
    return m.BendEffect(type=kind, value=int(round(st * 50)), points=pts)


def _note_effect(n: dict, tied: bool) -> m.NoteEffect:
    eff = m.NoteEffect()
    if tied:
        return eff
    for t in n.get("techniques", []):
        k = t["type"]
        if k in ("hammer_on", "pull_off"):
            pass  # in GP the *previous* note carries the hammer flag (set by caller)
        elif k == "slide":
            eff.slides.append(m.SlideType.legatoSlideTo if t.get("legato") else m.SlideType.shiftSlideTo)
        elif k == "slide_in":
            eff.slides.append(m.SlideType.intoFromBelow if t.get("direction") == "from_below" else m.SlideType.intoFromAbove)
        elif k in ("bend", "bend_release", "prebend_release"):
            eff.bend = _bend(t)
        elif k == "vibrato":
            eff.vibrato = True
        elif k == "palm_mute":
            eff.palmMute = True
        elif k == "let_ring":
            eff.letRing = True
        elif k == "tremolo_picking":
            eff.tremoloPicking = m.TremoloPickingEffect(duration=m.Duration(value=16))
        elif k == "harmonic":
            eff.harmonic = m.NaturalHarmonic()
    return eff


def build_song(tracks: list[dict], analysis: dict, title: str, tones: dict | None = None) -> m.Song:
    rhythm = analysis["rhythm"]
    duration = analysis["duration"]
    aligned, grid, measures = aligned_tracks(tracks, rhythm, duration)
    song = m.Song(title=title, tab="TabForge AI transcription", tempo=int(round(grid.get("bpm", rhythm["bpm"]))))
    song.tracks = []
    fifths, mode = key_fifths(analysis.get("key"))
    keysig = (KEY_BY_FIFTHS_MINOR if mode == "minor" else KEY_BY_FIFTHS_MAJOR).get(fifths, m.KeySignature.CMajor)
    num, den = grid["time_signature"]
    headers = []
    start = m.Duration.quarterTime
    for i, meas in enumerate(measures):
        h = m.MeasureHeader(number=i + 1, start=start, keySignature=keysig,
                            timeSignature=m.TimeSignature(numerator=num, denominator=m.Duration(value=den)))
        if i == 0 and analysis.get("rhythm", {}).get("feel") == "swing/triplet":
            h.tripletFeel = m.TripletFeel.eighth
        headers.append(h)
        start += meas["ticks"]
    song.measureHeaders = headers

    for ti, tr in enumerate(aligned):
        tuning = tr["_tuning"]
        channel = 2 * ti if 2 * ti < 9 else 2 * ti + 1
        track = m.Track(song, number=ti + 1, name=tr["name"][:40], fretCount=int(tr.get("max_fret", 24)),
                        offset=int(tr.get("capo", 0)),
                        channel=m.MidiChannel(channel=channel, effectChannel=channel + 1,
                                              instrument=gm_program(tr, (tones or {}).get(tr["stem"]))))
        track.strings = [m.GuitarString(s, tuning.open_midi(s)) for s in range(1, tuning.strings + 1)]
        track.measures = [m.Measure(track, h) for h in headers]
        idx = note_index(tr)
        prev_by_string: dict[int, m.Note] = {}
        for mi, meas in enumerate(tr["notation"]):
            voice = track.measures[mi].voices[0]
            for b in meas["beats"]:
                dur = m.Duration(value=b["value"], isDotted=b["dotted"],
                                 tuplet=m.Tuplet(3, 2) if b.get("tuplet") else m.Tuplet(1, 1))
                beat = m.Beat(voice, duration=dur)
                beat.status = m.BeatStatus.rest if b["rest"] else m.BeatStatus.normal
                for ref in b["notes"]:
                    n = idx.get(ref["id"])
                    if n is None or n.get("string") is None or n.get("fret") is None:
                        continue
                    if n.get("dead"):
                        note = m.Note(beat, value=0, string=int(n["string"]), velocity=int(n.get("velocity", 90)),
                                      type=m.NoteType.dead)
                    else:
                        note = m.Note(beat, value=int(n["fret"]), string=int(n["string"]),
                                      velocity=int(n.get("velocity", 90)),
                                      type=m.NoteType.tie if ref["tie"] else m.NoteType.normal,
                                      effect=_note_effect(n, ref["tie"]))
                    if not ref["tie"] and tech(n, "hammer_on") or (not ref["tie"] and tech(n, "pull_off")):
                        p = prev_by_string.get(int(n["string"]))
                        if p is not None:
                            p.effect.hammer = True
                    beat.notes.append(note)
                    prev_by_string[int(n["string"])] = note
                strum = next((t for ref in b["notes"] if (nn := idx.get(ref["id"])) and not ref["tie"]
                              for t in nn.get("techniques", []) if t["type"] == "strum"), None)
                if strum and len(beat.notes) >= 3:
                    beat.effect.stroke = m.BeatStroke(direction=m.BeatStrokeDirection.down if strum["direction"] == "down"
                                                      else m.BeatStrokeDirection.up, value=16)
                if not beat.notes and not b["rest"]:
                    beat.status = m.BeatStatus.rest
                voice.beats.append(beat)
            if not voice.beats:
                voice.beats.append(m.Beat(voice, duration=m.Duration(value=1), status=m.BeatStatus.rest))
        song.tracks.append(track)
    return song


def export_gp5(tracks: list[dict], analysis: dict, title: str, tones: dict | None = None) -> bytes:
    song = build_song(tracks, analysis, title, tones)
    buf = io.BytesIO()
    gp.write(song, buf, version=(5, 1, 0))
    return buf.getvalue()
