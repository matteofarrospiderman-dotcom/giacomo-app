"""End-to-end accuracy evaluation against ground truth.

Runs the full pipeline on an audio file with a ``*.truth.json`` next to it (as produced by
``make_demo_song.py``) and reports note F1, fingering agreement, tempo/meter/key/tuning.
Use it to measure the effect of model or parameter changes.

    python scripts/evaluate.py data/demo/demo_song.wav [--separator dsp|demucs|auto]

For real recordings, create a truth file from a verified Guitar Pro/MIDI transcription
(fields: bpm, time_signature, key, tuning, rhythm_guitar/lead_guitar/bass note lists with
pitch/start/end[/string/fret]).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--separator", default="auto")
    ap.add_argument("--tolerance", type=float, default=0.06, help="onset tolerance in seconds")
    a = ap.parse_args()
    truth_path = a.audio.with_suffix("").with_suffix(".truth.json")
    if not truth_path.exists():
        truth_path = a.audio.parent / (a.audio.stem + ".truth.json")
    truth = json.loads(truth_path.read_text())

    import os  # noqa: PLC0415

    os.environ["TABFORGE_DATA_DIR"] = tempfile.mkdtemp(prefix="tabforge-eval-")
    from app import pipeline, storage  # noqa: PLC0415
    from app.config import ensure_dirs  # noqa: PLC0415
    from tests.metrics import note_f1  # noqa: PLC0415

    ensure_dirs()
    meta = storage.create_project(a.audio.stem, {"separator": a.separator, "tuning": "auto"})
    pdir = storage.project_dir(meta["id"])
    shutil.copy(a.audio, pdir / f"source{a.audio.suffix}")
    meta["source"] = {"file": f"source{a.audio.suffix}"}
    storage.save_project(meta)
    t0 = time.time()
    pipeline.process(meta["id"])
    elapsed = time.time() - t0
    an = storage.load_analysis(meta["id"])
    tracks = storage.list_tracks(meta["id"])
    guitar = [n for t in tracks if t["instrument"] == "guitar" for n in t["notes"]]
    bass = [n for t in tracks if t["instrument"] == "bass" for n in t["notes"]]
    gt_guitar = truth.get("rhythm_guitar", []) + truth.get("lead_guitar", [])

    report = {
        "processing_seconds": round(elapsed, 1),
        "separation": an["separation"]["engine"],
        "tempo": {"estimated": an["rhythm"]["bpm"], "truth": truth.get("bpm")},
        "time_signature": {"estimated": an["rhythm"]["time_signature"], "truth": truth.get("time_signature")},
        "key": {"estimated": an["key"]["name"], "truth": truth.get("key")},
        "tuning": {"estimated": an.get("tuning", {}).get("best", {}).get("id"), "truth": truth.get("tuning")},
        "guitar_notes": note_f1(guitar, gt_guitar, a.tolerance),
    }
    for part in ("rhythm_guitar", "lead_guitar"):
        if truth.get(part):
            report[f"{part}_recall"] = note_f1(guitar, truth[part], a.tolerance)["recall"]
    if truth.get("bass"):
        report["bass_notes"] = note_f1(bass, truth["bass"], a.tolerance)
    matched = [(g, n) for g in gt_guitar if "string" in g for n in guitar
               if n["pitch"] == g["pitch"] and abs(n["start"] - g["start"]) < a.tolerance]
    if matched:
        same = sum((n["string"], n["fret"]) == (g["string"], g["fret"]) for g, n in matched)
        report["fingering_identical_to_truth"] = round(same / len(matched), 3)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
