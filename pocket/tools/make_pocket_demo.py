"""Render the TabForge Pocket demo piece (solo guitar, known ground truth).

96 BPM, 4/4, E minor, standard tuning: Em - C - G - D arpeggios (bars 1-4) followed by a
lead phrase with a hammer-on, a full bend and vibrato over the same chords (bars 5-8).

Writes <out>/demo.mp3 (for the page), <out>/demo.wav, <out>/demo_22k.f32 (mono 22.05 kHz,
for building demo.json with the engine in Node) and <out>/demo.truth.json.

Usage: backend/.venv/bin/python pocket/tools/make_pocket_demo.py <out_dir>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from tests.fx import reverb  # noqa: E402
from tests.synth import render  # noqa: E402

SR = 44100
BPM = 96.0
BEAT = 60.0 / BPM
STD = [40, 45, 50, 55, 59, 64]


def pitch(s: int, f: int) -> int:
    return STD[6 - s] + f


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    start = 0.5
    notes = []
    # arpeggio shapes: (string, fret) in picking order, eighth notes
    shapes = {
        "Em": [(6, 0), (5, 2), (4, 2), (3, 0), (2, 0), (3, 0), (4, 2), (5, 2)],
        "C": [(5, 3), (4, 2), (3, 0), (2, 1), (1, 0), (2, 1), (3, 0), (4, 2)],
        "G": [(6, 3), (5, 2), (4, 0), (3, 0), (2, 3), (3, 0), (4, 0), (5, 2)],
        "D": [(4, 0), (3, 2), (2, 3), (1, 2), (2, 3), (3, 2), (4, 0), (3, 2)],
    }
    prog = ["Em", "C", "G", "D"]
    for bar, ch in enumerate(prog):
        for k, (s, f) in enumerate(shapes[ch]):
            t = start + (bar * 4 + k * 0.5) * BEAT
            notes.append({"pitch": pitch(s, f), "start": t, "end": t + BEAT * 0.95, "vel": 0.65, "string": s, "fret": f})
    # lead phrase: (string, fret, beats, technique)
    phrase = [(2, 8, 1, None), (2, 10, 0.5, None), (1, 7, 0.5, None), (1, 10, 2, "vibrato"),
              (2, 8, 1, None), (2, 10, 1, "hammer_on"), (3, 9, 2, "bend"),
              (3, 7, 1, None), (3, 9, 1, None), (2, 8, 2, "vibrato"),
              (3, 9, 1, None), (2, 7, 1, None), (3, 7, 2, None)]
    t = start + 16 * BEAT
    for s, f, beats, tech in phrase:
        n = {"pitch": pitch(s, f), "start": t, "end": t + beats * BEAT * 0.97, "vel": 0.85, "string": s, "fret": f,
             "technique": tech}
        if tech == "bend":
            n["bend_to"] = n["pitch"] + 2
        if tech == "vibrato":
            n["vibrato"] = (5.5, 0.35)
        notes.append(n)
        t += beats * BEAT
    # bass-register chord roots under the lead (bars 5-8)
    for bar, root in enumerate([(6, 0), (5, 3), (6, 3), (4, 0)]):
        tt = start + (16 + bar * 4) * BEAT
        notes.append({"pitch": pitch(*root), "start": tt, "end": tt + 3.8 * BEAT, "vel": 0.55, "string": root[0], "fret": root[1]})
    dur = start + 32 * BEAT + 1.5
    x = render(notes, dur, pan=0.0)
    x = reverb(x, rt60=1.1, mix=0.18)
    x = x / (np.max(np.abs(x)) * 1.15)
    sf.write(out / "demo.wav", x.T, SR)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(out / "demo.wav"), "-b:a", "128k", str(out / "demo.mp3")], check=True)
    y22 = librosa.resample(x.mean(0), orig_sr=SR, target_sr=22050).astype(np.float32)
    y22.tofile(out / "demo_22k.f32")
    truth = {"bpm": BPM, "notes": [{k: v for k, v in n.items() if k not in ("vel", "vibrato")} for n in notes]}
    (out / "demo.truth.json").write_text(json.dumps(truth, indent=1))
    print(f"demo: {dur:.1f}s, {len(notes)} notes -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
