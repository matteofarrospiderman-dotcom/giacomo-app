"""Generate a demo multitrack song with exact ground truth for testing the app end-to-end.

Contents (110 BPM, 4/4, E minor, standard tuning):
  * rhythm guitar: palm-muted E5/G5/A5/D5 power chords, driven (tanh) and panned left
  * lead guitar: pentatonic melody with a full bend, vibrato and legato, delay, panned right
  * bass: root notes; drums: kick/snare/hat pattern

Writes ``demo_song.wav`` and ``demo_song.truth.json`` (notes with string/fret).

Usage:  python scripts/make_demo_song.py [output_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.fx import delay, distort  # noqa: E402
from tests.synth import render  # noqa: E402

SR = 44100
BPM = 110.0
BEAT = 60.0 / BPM
STD = [40, 45, 50, 55, 59, 64]


def pitch(string: int, fret: int) -> int:
    return STD[6 - string] + fret


def drums(bars: int, start: float) -> np.ndarray:
    n = int((start + bars * 4 * BEAT + 2) * SR)
    y = np.zeros(n, np.float32)
    rng = np.random.default_rng(7)
    for b in range(bars * 8):
        t = start + b * BEAT / 2
        i = int(t * SR)
        L = int(0.12 * SR)
        tt = np.arange(L) / SR
        if b % 4 == 0:  # kick on 1 and 3
            s = np.sin(2 * np.pi * (55 + 60 * np.exp(-tt / 0.02)) * tt) * np.exp(-tt / 0.09) * 0.9
        elif b % 4 == 2:  # snare on 2 and 4
            s = (rng.normal(0, 1, L) * 0.45 + np.sin(2 * np.pi * 190 * tt) * 0.4) * np.exp(-tt / 0.06)
        else:
            s = np.zeros(L)
        hat = rng.normal(0, 1, L) * np.exp(-tt / 0.012) * 0.12
        y[i: i + L] += (s + hat)[: max(0, n - i)]
    return np.stack([y, y])


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    start = BEAT * 2  # two beats count-in of drums
    bars = 8
    rhythm, lead, bass = [], [], []
    shapes = [(6, 0), (6, 3), (6, 5), (5, 5)]  # E5 G5 A5 D5 (root string, fret)
    for bar in range(bars):
        s, f = shapes[bar % 4]
        for k in range(8):  # palm-muted eighth-note chugs
            t = start + bar * 4 * BEAT + k * BEAT / 2
            dur = BEAT / 2 * (0.55 if k < 7 else 0.95)
            root = pitch(s, f)
            for ss, ff in ((s, f), (s - 1, f + 2)):
                rhythm.append({"pitch": pitch(ss, ff), "start": t, "end": t + dur, "vel": 0.7, "string": ss, "fret": ff,
                               "technique": "palm_mute" if k < 7 else None})
            if k % 2 == 0:
                bass.append({"pitch": root - 12, "start": t, "end": t + BEAT * 0.9, "vel": 0.7})
    # lead enters in bar 3: (string, fret, beats, technique)
    phrase = [(2, 8, 1, None), (1, 5, 0.5, None), (1, 8, 0.5, "hammer"), (2, 8, 2, "bend"),
              (2, 5, 1, None), (3, 7, 1, None), (3, 5, 2, "vibrato"),
              (3, 7, 0.5, None), (2, 5, 0.5, None), (2, 8, 1, None), (1, 5, 2, "vibrato"), (2, 5, 2, None)]
    t = start + 2 * 4 * BEAT
    for s, f, beats, tech in phrase:
        n = {"pitch": pitch(s, f), "start": t, "end": t + beats * BEAT * 0.95, "vel": 0.85, "string": s, "fret": f,
             "technique": tech}
        if tech == "bend":
            n["bend_to"] = n["pitch"] + 2
        if tech == "vibrato":
            n["vibrato"] = (5.5, 0.3)
        lead.append(n)
        t += beats * BEAT
    total = start + bars * 4 * BEAT + 2.0
    rh = distort(render(rhythm, total, pan=-0.7), drive=12.0)
    ld = delay(distort(render(lead, total, pan=0.7), drive=6.0) * 1.2, ms=BEAT * 750, feedback=0.35, mix=0.3)
    bs = render(bass, total)
    dr = drums(bars, start)[:, : rh.shape[1]]
    n = min(rh.shape[1], ld.shape[1], bs.shape[1], dr.shape[1])
    mix = 0.55 * rh[:, :n] + 0.6 * ld[:, :n] + 0.45 * bs[:, :n] + 0.5 * dr[:, :n]
    mix /= np.max(np.abs(mix)) * 1.1
    sf.write(out_dir / "demo_song.wav", mix.T, SR)
    truth = {"bpm": BPM, "time_signature": [4, 4], "key": "E minor", "tuning": "standard",
             "rhythm_guitar": [{k: v for k, v in x.items() if k != "vel"} for x in rhythm],
             "lead_guitar": [{k: v for k, v in x.items() if k not in ("vel", "vibrato")} for x in lead],
             "bass": [{k: v for k, v in x.items() if k != "vel"} for x in bass]}
    (out_dir / "demo_song.truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {out_dir / 'demo_song.wav'} ({n / SR:.1f}s) and ground truth")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "demo")
