"""PDF tablature rendering with ReportLab.

Draws proper tab staves (one line per string), bar lines and measure numbers, fret
numbers on the lines, rhythm stems with flags/beams below the staff (Guitar Pro style
"rhythm in tab"), technique marks (h/p arcs, slides, bend arrows with amounts,
vibrato waves, P.M. brackets, let-ring), chord names above the staff and a header with
tuning, tempo, key and the transcription's confidence summary. Low-confidence notes are
printed in grey so the reader knows what to double-check.
"""

from __future__ import annotations

import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from ..guitar.tunings import midi_to_name
from .common import aligned_tracks, note_index, tech

PAGE_W, PAGE_H = A4
MARGIN = 40
LINE_GAP = 7.0
SYSTEM_GAP = 44


def _beat_x(b: dict, meas_ticks: int, x0: float, w: float) -> float:
    return x0 + 10 + (w - 16) * (b["tick"] / max(1, meas_ticks))


def export_pdf(tracks: list[dict], analysis: dict, title: str) -> bytes:
    aligned, grid, measures = aligned_tracks(tracks, analysis["rhythm"], analysis["duration"])
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(title)
    chords = analysis.get("chords", [])
    for t in aligned:
        _render_track(c, t, measures, grid, analysis, title, chords)
    c.save()
    return buf.getvalue()


def _header(c, t, analysis, title, y):
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, y, title)
    y -= 18
    c.setFont("Helvetica", 9.5)
    tun = t["tuning"]
    info = (f"{t['name']}  ·  Tuning: {tun['name']} ({' '.join(midi_to_name(p) for p in tun['notes'])})"
            + (f"  ·  Capo {t['capo']}" if t.get("capo") else ""))
    c.drawString(MARGIN, y, info)
    y -= 12
    st = t.get("stats", {})
    lv = st.get("levels", {})
    c.drawString(MARGIN, y, f"Tempo ≈ {analysis['rhythm']['bpm']:.0f} BPM  ·  {t['grid']['time_signature'][0]}/"
                            f"{t['grid']['time_signature'][1]}  ·  Key: {analysis.get('key', {}).get('name', '?')}  ·  "
                            f"Confidence: {lv.get('high', 0)} high / {lv.get('medium', 0)} medium / {lv.get('low', 0)} low"
                            " (grey = verify)")
    y -= 12
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(MARGIN, y, "Automatic transcription by TabForge AI – not guaranteed to be correct.")
    return y - 22


def _render_track(c, t, measures, grid, analysis, title, chords):
    n_str = len(t["tuning"]["notes"])
    labels = [midi_to_name(p)[:-1] for p in reversed(t["tuning"]["notes"])]
    idx = note_index(t)
    staff_h = (n_str - 1) * LINE_GAP
    sys_h = staff_h + SYSTEM_GAP + 18
    per_line = 4
    usable_w = PAGE_W - 2 * MARGIN - 14
    meas_w = usable_w / per_line
    y = _header(c, t, analysis, title, PAGE_H - MARGIN)
    notation = t["notation"]
    for li in range(0, len(notation), per_line):
        if y - sys_h < MARGIN:
            c.showPage()
            y = PAGE_H - MARGIN - 10
        top = y - 16  # leave room for chord names / bends above
        # string lines + labels
        c.setLineWidth(0.4)
        c.setStrokeGray(0.45)
        x_start = MARGIN + 14
        x_end = x_start + meas_w * len(notation[li: li + per_line])
        for s in range(n_str):
            yy = top - s * LINE_GAP
            c.line(x_start, yy, x_end, yy)
            c.setFont("Helvetica", 6.5)
            c.setFillGray(0.3)
            c.drawRightString(x_start - 3, yy - 2, labels[s])
        for k, meas in enumerate(notation[li: li + per_line]):
            mi = li + k
            x0 = x_start + k * meas_w
            c.setStrokeGray(0.2)
            c.setLineWidth(0.8)
            c.line(x0, top, x0, top - staff_h)
            c.setFont("Helvetica", 6)
            c.setFillGray(0.4)
            c.drawString(x0 + 1, top + 4, str(mi + 1))
            m = measures[mi]
            # chord names
            for ch in chords:
                if ch.get("root") is None or ch["end"] <= m["start"] or ch["start"] >= m["end"]:
                    continue
                if ch["start"] >= m["start"] - 0.05:
                    frac = (max(ch["start"], m["start"]) - m["start"]) / max(1e-3, m["end"] - m["start"])
                    c.setFont("Helvetica-Bold", 7.5)
                    c.setFillGray(0.1)
                    c.drawString(x0 + 8 + (meas_w - 16) * frac, top + 12, ch["name"])
            pm_x = None
            for b in meas["beats"]:
                bx = _beat_x(b, m["ticks"], x0, meas_w)
                notes = [(idx[r["id"]], r["tie"]) for r in b["notes"] if r["id"] in idx and idx[r["id"]].get("string")]
                # rhythm stems below the staff
                if not b["rest"] and notes and not all(tie for _, tie in notes):
                    _stem(c, bx, top - staff_h - 4, b)
                elif b["rest"]:
                    c.setFont("Helvetica", 6)
                    c.setFillGray(0.5)
                    c.drawCentredString(bx, top - staff_h - 14, "·")
                any_pm = False
                for n, tie in notes:
                    s = int(n["string"]) - 1
                    if s >= n_str:
                        continue
                    yy = top - s * LINE_GAP
                    txt = "x" if n.get("dead") else str(n["fret"])
                    if tie:
                        txt = f"({txt})"
                    c.setFont("Helvetica-Bold" if n.get("edited") else "Helvetica", 7)
                    tw = c.stringWidth(txt, "Helvetica", 7)
                    c.setFillGray(1)
                    c.rect(bx - tw / 2 - 1, yy - 3, tw + 2, 6, stroke=0, fill=1)
                    c.setFillGray(0.0 if n.get("confidence", 1) >= 0.45 or n.get("edited") else 0.55)
                    c.drawCentredString(bx, yy - 2.4, txt)
                    if tie:
                        continue
                    for tt in n.get("techniques", []):
                        k = tt["type"]
                        if k in ("hammer_on", "pull_off"):
                            c.setStrokeGray(0.2)
                            c.setLineWidth(0.5)
                            p = c.beginPath()
                            p.arc(bx - 16, yy + 2, bx - 2, yy + 9, startAng=20, extent=140)
                            c.drawPath(p)
                            c.setFont("Helvetica", 5.5)
                            c.drawCentredString(bx - 9, yy + 9, "H" if k == "hammer_on" else "P")
                        elif k == "slide":
                            c.setLineWidth(0.5)
                            d = 3 if tt.get("direction") == "up" else -3
                            c.line(bx + 5, yy - d, bx + 13, yy + d)
                        elif k in ("bend", "bend_release", "prebend_release"):
                            c.setLineWidth(0.5)
                            c.line(bx + 5, yy, bx + 9, top + 6)
                            amt = tt.get("semitones", 1)
                            label = "full" if amt == 2 else ("½" if amt == 1 else f"{amt / 2:g}")
                            c.setFont("Helvetica", 5.5)
                            c.drawString(bx + 6, top + 8, label + (" R" if k != "bend" else ""))
                        elif k == "vibrato":
                            c.setFont("Helvetica", 7)
                            c.drawString(bx + 4, top + 3, "~~~")
                        elif k == "palm_mute":
                            any_pm = True
                if any_pm and pm_x is None:
                    pm_x = bx
            if pm_x is not None:
                c.setFont("Helvetica", 6)
                c.setFillGray(0.2)
                c.drawString(pm_x - 4, top + 3, "P.M.")
        c.setStrokeGray(0.2)
        c.setLineWidth(0.8)
        c.line(x_end, top, x_end, top - staff_h)
        y -= sys_h
    c.showPage()


def _stem(c, x, y, b):
    c.setStrokeGray(0.15)
    c.setLineWidth(0.6)
    val = b["value"]
    if val >= 2:
        c.line(x, y, x, y - 12)
    flags = {8: 1, 16: 2, 32: 3, 64: 4}.get(val, 0)
    for f in range(flags):
        c.line(x, y - 12 + f * 2.5, x + 4, y - 9 + f * 2.5)
    if b.get("dotted"):
        c.circle(x + 3, y - 13, 0.6, stroke=0, fill=1)
    if b.get("tuplet"):
        c.setFont("Helvetica", 5)
        c.drawCentredString(x, y - 19, "3")
