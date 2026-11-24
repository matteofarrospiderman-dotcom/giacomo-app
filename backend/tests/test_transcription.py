import librosa
import numpy as np
import pytest

from app.transcription import basic_pitch_onnx, ensemble, techniques
from tests.metrics import note_f1
from tests.synth import render, riff_ground_truth


@pytest.mark.skipif(not basic_pitch_onnx.available()[0], reason="Basic Pitch ONNX model not installed")
def test_ensemble_accuracy_on_synthetic_riff():
    gt = riff_ground_truth()
    x = render(gt, gt[-1]["end"] + 1.0)
    y = librosa.resample(x.mean(0), orig_sr=44100, target_sr=22050)
    res = ensemble.transcribe(y, "guitar")
    est = [{"pitch": n.pitch, "start": n.start} for n in res["notes"]]
    m = note_f1(est, gt)
    assert m["recall"] >= 0.95
    assert m["f1"] >= 0.85
    # harmonic ghosts must be rejected, not silently kept
    assert any("rejected:harmonic_ghost" in n.flags for n in res["rejected"])


@pytest.mark.skipif(not basic_pitch_onnx.available()[0], reason="Basic Pitch ONNX model not installed")
def test_bend_and_vibrato_detection():
    notes = [{"pitch": 64, "start": 0.3, "end": 1.3, "bend_to": 66},
             {"pitch": 67, "start": 1.6, "end": 3.0, "vibrato": (5.5, 0.35)}]
    x = render(notes, 3.6)
    y = librosa.resample(x.mean(0), orig_sr=44100, target_sr=22050)
    res = ensemble.transcribe(y, "guitar")
    techniques.detect(res["notes"], y, res["f0"], res["polyphony"])
    kinds = {t["type"] for n in res["notes"] for t in n.techniques}
    assert "bend" in kinds or "bend_release" in kinds
    assert "vibrato" in kinds
    bend = next(t for n in res["notes"] for t in n.techniques if t["type"].startswith("bend"))
    assert bend["semitones"] == pytest.approx(2.0, abs=0.5)
