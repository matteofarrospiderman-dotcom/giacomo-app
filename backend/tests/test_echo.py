import librosa
import pytest

from app.transcription import basic_pitch_onnx, ensemble
from app.transcription.echo import suppress_echoes
from tests.fx import delay
from tests.metrics import note_f1
from tests.synth import render, riff_ground_truth


def _transcribe(x):
    y = librosa.resample(x.mean(0), orig_sr=44100, target_sr=22050)
    res = ensemble.transcribe(y, "guitar")
    return res, y


@pytest.mark.skipif(not basic_pitch_onnx.available()[0], reason="Basic Pitch ONNX model not installed")
def test_delay_echoes_are_removed_from_lead():
    pitches = [64, 67, 69, 71, 74, 72, 69, 67, 64, 62, 64, 67, 69, 67, 64, 62]
    gt = [{"pitch": p, "start": 0.4 + i * 0.6, "end": 0.4 + i * 0.6 + 0.25, "vel": 0.9} for i, p in enumerate(pitches)]
    x = delay(render(gt, gt[-1]["end"] + 1.5), ms=375, feedback=0.35, mix=0.45)
    res, y = _transcribe(x)
    before = note_f1([{"pitch": n.pitch, "start": n.start} for n in res["notes"]], gt)
    kept, removed, info = suppress_echoes(res["notes"], y, res["salience"])
    after = note_f1([{"pitch": n.pitch, "start": n.start} for n in kept], gt)
    assert info is not None and abs(info["time_s"] - 0.375) < 0.04
    assert after["precision"] > before["precision"]
    assert after["recall"] >= before["recall"] - 0.07  # real notes survive


@pytest.mark.skipif(not basic_pitch_onnx.available()[0], reason="Basic Pitch ONNX model not installed")
def test_no_echoes_invented_for_dry_riff():
    gt = riff_ground_truth(bars=6)
    res, y = _transcribe(render(gt, gt[-1]["end"] + 1.0))
    kept, removed, info = suppress_echoes(res["notes"], y, res["salience"])
    assert info is None and not removed
