"""MusicXML 4.0 export with standard notation + TAB staff per guitar part.

Each track becomes one part with two staves: staff 1 is standard notation (treble
clef 8vb for guitar, bass clef 8vb for bass), staff 2 is a TAB staff whose
``staff-details`` carry the exact tuning. Every note carries ``<string>``/``<fret>`` so
MuseScore, Finale, Sibelius, Dorico and Guitar Pro import the fingering, not just the
pitches. Techniques map to native MusicXML: hammer-on/pull-off, slide, bend (with
release / pre-bend), vibrato (wavy line), palm mute and let ring (directions), dead
notes (x noteheads), ties, triplets.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

from .common import aligned_tracks, key_fifths, note_index, spell, tech

TYPE_NAMES = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd", 64: "64th"}


def _note_xml(n: dict | None, b: dict, staff: int, voice: int, chord: bool, tie: bool, flats: bool, tab: bool,
              next_legato: dict | None, tie_start: bool = False) -> str:
    x = ["<note>"]
    if chord:
        x.append("<chord/>")
    if n is None:
        x.append("<rest/>")
    else:
        step, alter, octave = spell(int(n["pitch"]), flats)
        x.append(f"<pitch><step>{step}</step>{f'<alter>{alter}</alter>' if alter else ''}<octave>{octave}</octave></pitch>")
    x.append(f"<duration>{b['ticks']}</duration>")
    if tie:
        x.append('<tie type="stop"/>')
    if n is not None and tie_start:
        x.append('<tie type="start"/>')
    x.append(f"<voice>{voice}</voice><type>{TYPE_NAMES.get(b['value'], 'quarter')}</type>")
    if b.get("dotted"):
        x.append("<dot/>")
    if b.get("tuplet"):
        x.append("<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>")
    if n is not None and n.get("dead"):
        x.append("<notehead>x</notehead>")
    x.append(f"<staff>{staff}</staff>")
    if n is not None:
        nots = []
        if tie:
            nots.append('<tied type="stop"/>')
        if tie_start:
            nots.append('<tied type="start"/>')
        technical = []
        if tab:
            technical.append(f"<string>{n['string']}</string><fret>{n['fret']}</fret>")
        if not tie:
            for kind, tag in (("hammer_on", "hammer-on"), ("pull_off", "pull-off")):
                if tech(n, kind):
                    technical.append(f'<{tag} type="stop"/>')
            if next_legato:
                tag = "hammer-on" if next_legato["type"] == "hammer_on" else "pull-off"
                technical.append(f'<{tag} type="start">{"H" if tag == "hammer-on" else "P"}</{tag}>')
            for k in ("bend", "bend_release", "prebend_release"):
                t = tech(n, k)
                if t:
                    alter_v = t.get("semitones", 1)
                    if k == "prebend_release":
                        technical.append(f"<bend><bend-alter>{-alter_v}</bend-alter><pre-bend/></bend>")
                    else:
                        technical.append(f"<bend><bend-alter>{alter_v}</bend-alter></bend>")
                        if k == "bend_release":
                            technical.append(f"<bend><bend-alter>{-alter_v}</bend-alter><release/></bend>")
            if tech(n, "harmonic"):
                technical.append("<harmonic><natural/></harmonic>")
        if technical:
            nots.append("<technical>" + "".join(technical) + "</technical>")
        if not tie:
            if tech(n, "slide"):
                nots.append('<slide type="start" line-type="solid"/>')
            if n.get("_slide_stop"):
                nots.append('<slide type="stop" line-type="solid"/>')
            if tech(n, "vibrato"):
                nots.append('<ornaments><wavy-line type="start"/><wavy-line type="stop"/></ornaments>')
            if b.get("tuplet") and n.get("_tuplet_mark"):
                nots.append(f'<tuplet type="{n["_tuplet_mark"]}"/>')
        if nots:
            x.append("<notations>" + "".join(nots) + "</notations>")
    x.append("</note>")
    return "".join(x)


def _direction(text: str, staff: int) -> str:
    return (f'<direction placement="above"><direction-type><words>{escape(text)}</words></direction-type>'
            f"<staff>{staff}</staff></direction>")


def export_musicxml(tracks: list[dict], analysis: dict, title: str) -> str:
    rhythm = analysis["rhythm"]
    aligned, grid, measures = aligned_tracks(tracks, rhythm, analysis["duration"])
    fifths, mode = key_fifths(analysis.get("key"))
    flats = bool(analysis.get("key", {}).get("use_flats"))
    num, den = grid["time_signature"]
    out = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
           '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
           '"http://www.musicxml.org/dtds/partwise.dtd">',
           '<score-partwise version="4.0">',
           f"<work><work-title>{escape(title)}</work-title></work>",
           "<identification><encoding><software>TabForge AI</software></encoding></identification>",
           "<part-list>"]
    for i, t in enumerate(aligned):
        out.append(f'<score-part id="P{i + 1}"><part-name>{escape(t["name"])}</part-name></score-part>')
    out.append("</part-list>")
    for i, t in enumerate(aligned):
        tuning = t["_tuning"]
        is_bass = t.get("instrument") == "bass"
        idx = note_index(t)
        # pre-compute legato starts, ties and slide stops
        by_string_next: dict[str, dict] = {}
        ordered = sorted([n for n in t["notes"] if n.get("string")], key=lambda n: n["start"])
        # an occurrence starts a tie when the same note's next occurrence is a continuation
        tie_starts: set[tuple[int, int, str]] = set()
        last_seen: dict[str, tuple[int, int]] = {}
        for mi_, meas_ in enumerate(t["notation"]):
            for bi_, b_ in enumerate(meas_["beats"]):
                for r in b_["notes"]:
                    if r["tie"] and r["id"] in last_seen:
                        tie_starts.add((*last_seen[r["id"]], r["id"]))
                    last_seen[r["id"]] = (mi_, bi_)
        for n in ordered:
            for tt in n.get("techniques", []):
                if tt["type"] in ("hammer_on", "pull_off"):
                    prev = [m for m in ordered if m["start"] < n["start"] and m.get("string") == n.get("string")]
                    if prev:
                        by_string_next[prev[-1]["id"]] = tt
                if tt["type"] == "slide":
                    nxt = [m for m in ordered if m["start"] > n["start"] and m.get("string") == n.get("string")]
                    if nxt:
                        nxt[0]["_slide_stop"] = True
        out.append(f'<part id="P{i + 1}">')
        prev_pm = False
        for mi, meas in enumerate(t["notation"]):
            out.append(f'<measure number="{mi + 1}">')
            if mi == 0:
                lines = "".join(f'<staff-tuning line="{k + 1}"><tuning-step>{spell(p, False)[0]}</tuning-step>'
                                f'{"<tuning-alter>1</tuning-alter>" if spell(p, False)[1] else ""}'
                                f"<tuning-octave>{spell(p, False)[2]}</tuning-octave></staff-tuning>"
                                for k, p in enumerate(tuning.notes))
                clef1 = ('<clef number="1"><sign>F</sign><line>4</line><clef-octave-change>-1</clef-octave-change></clef>'
                         if is_bass else
                         '<clef number="1"><sign>G</sign><line>2</line><clef-octave-change>-1</clef-octave-change></clef>')
                out.append(f"<attributes><divisions>960</divisions><key><fifths>{fifths}</fifths><mode>{mode}</mode></key>"
                           f"<time><beats>{num}</beats><beat-type>{den}</beat-type></time><staves>2</staves>"
                           f'{clef1}<clef number="2"><sign>TAB</sign><line>5</line></clef>'
                           f'<staff-details number="2"><staff-lines>{tuning.strings}</staff-lines>{lines}'
                           f'{"<capo>%d</capo>" % t["capo"] if t.get("capo") else ""}</staff-details></attributes>')
                out.append(f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                           f'<per-minute>{round(grid["bpm"] * grid["ticks_per_beat"] / 960)}</per-minute></metronome>'
                           f'</direction-type><staff>1</staff><sound tempo="{round(grid["bpm"] * grid["ticks_per_beat"] / 960)}"/></direction>')
                if t.get("capo"):
                    out.append(_direction(f"Capo {t['capo']}", 1))
            for staff in (1, 2):
                if staff == 2:
                    total = sum(b["ticks"] for b in meas["beats"])
                    out.append(f"<backup><duration>{total}</duration></backup>")
                triplet_run = 0
                for bi, b in enumerate(meas["beats"]):
                    notes = [idx[r["id"]] for r in b["notes"] if r["id"] in idx and idx[r["id"]].get("string")]
                    ties = {r["id"]: r["tie"] for r in b["notes"]}
                    if staff == 1 and notes:
                        pm = any(tech(n, "palm_mute") for n in notes)
                        if pm and not prev_pm:
                            out.append(_direction("P.M.", 1))
                        prev_pm = pm
                        if any(tech(n, "let_ring") for n in notes) and not any(ties.get(n["id"]) for n in notes):
                            out.append(_direction("let ring", 1))
                    if b.get("tuplet"):
                        triplet_run += 1
                        mark = "start" if triplet_run % 3 == 1 else "stop" if triplet_run % 3 == 0 else None
                    else:
                        triplet_run = 0
                        mark = None
                    if not notes:
                        out.append(_note_xml(None, b, staff, staff, False, False, flats, staff == 2, None))
                        continue
                    for k, n in enumerate(sorted(notes, key=lambda n: n["pitch"])):
                        n2 = dict(n)
                        n2["_tuplet_mark"] = mark
                        out.append(_note_xml(n2, b, staff, staff, k > 0, bool(ties.get(n["id"])), flats, staff == 2,
                                             by_string_next.get(n["id"]) if not ties.get(n["id"]) else None,
                                             tie_start=(mi, bi, n["id"]) in tie_starts))
            out.append("</measure>")
        out.append("</part>")
    out.append("</score-partwise>")
    return "\n".join(out)
