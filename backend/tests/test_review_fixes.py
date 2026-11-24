"""Regression tests for issues found in code review."""

import librosa
import numpy as np
import pytest

from app.analysis.harmony import estimate_key, fuse_key
from app.analysis.rhythm import analyze_rhythm
from app.guitar.quantize import make_grid, measures_info, quantize
from tests.synth import click_track, render


def test_pickup_bar_has_correct_timing():
    beats = list(0.3 + 0.5 * np.arange(30))
    rhythm = {"bpm": 120.0, "beats": beats, "downbeats": [b for b in beats if abs((b - 1.3) / 2 - round((b - 1.3) / 2)) < 1e-6],
              "time_signature": [4, 4]}
    grid = make_grid(rhythm, 15.0)
    notes = [{"id": "a", "start": 0.35, "end": 0.6, "pitch": 60}, {"id": "b", "start": 1.3, "end": 1.6, "pitch": 62}]
    info = quantize(notes, grid)
    m0 = measures_info(grid, info["first_bar_beat"], 2)[0]
    assert m0["bpm"] == pytest.approx(120, abs=1)
    assert m0["start"] == pytest.approx(-0.7, abs=0.02)


def test_silent_chroma_gives_finite_key():
    k = estimate_key(np.zeros(12))
    assert np.isfinite(k["correlation"]) and np.isfinite(k["confidence"])
    fused = fuse_key(k, [{"pitch": p, "start": 0, "end": 1, "confidence": 0.9} for p in (57, 61, 64)])
    assert fused["name"] == "A major" and np.isfinite(fused["confidence"])


def test_tempo_curve_has_no_edge_spikes():
    x = click_track(120, 20.0)
    y = librosa.resample(x.mean(0), orig_sr=44100, target_sr=22050)
    curve = analyze_rhythm(y, 22050)["tempo_curve"]
    assert all(abs(c["bpm"] - 120) < 6 for c in curve)


def test_f0_track_stays_on_time_late_in_a_file():
    from app.transcription import f0_tracker

    t_note = 75.0
    x = render([{"pitch": 64, "start": t_note, "end": t_note + 0.8, "vel": 0.9}], t_note + 1.5)
    y = librosa.resample(x.mean(0), orig_sr=44100, target_sr=22050)
    tr = f0_tracker.track(y)
    segs = [s for s in f0_tracker.segment_notes(tr, voicing_thr=0.5) if abs(s["pitch"] - 64) <= 1]
    assert segs, "note not tracked"
    assert segs[0]["start"] == pytest.approx(t_note, abs=0.06)


def test_broken_pool_is_rebuilt(monkeypatch, tmp_path):
    from concurrent.futures.process import BrokenProcessPool

    from app import storage
    from app.jobs import JobManager

    meta = storage.create_project("pool", {})
    jm = JobManager(1)
    calls = {"n": 0}

    class FakePool:
        def __init__(self, broken):
            self.broken = broken

        def submit(self, *a, **k):
            calls["n"] += 1
            if self.broken:
                raise BrokenProcessPool("dead worker")
            from concurrent.futures import Future

            f = Future()
            f.set_result(None)
            return f

        def shutdown(self, **k):
            pass

    pools = [FakePool(True), FakePool(False)]
    monkeypatch.setattr(jm, "_get_pool", lambda: pools[0] if jm._pool is None and pools[0].broken and calls["n"] == 0 else pools[1])
    jm.submit("process", meta["id"])
    assert calls["n"] == 2
    assert storage.load_project(meta["id"])["status"]["job"]["kind"] == "process"
