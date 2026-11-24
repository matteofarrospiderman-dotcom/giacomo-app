import librosa
import pytest

from app.analysis.rhythm import analyze_rhythm
from tests.synth import click_track


@pytest.mark.parametrize("bpm", [92.0, 110.0, 137.0])
def test_tempo_and_meter(bpm):
    x = click_track(bpm, 20.0)
    y = librosa.resample(x.mean(0), orig_sr=44100, target_sr=22050)
    r = analyze_rhythm(y, 22050)
    assert r["bpm"] == pytest.approx(bpm, abs=0.6)
    assert r["time_signature"] == [4, 4]
    # downbeats fall on the accented (kick) clicks every 4 beats
    period = 60 / bpm
    assert all(abs((d / (4 * period)) - round(d / (4 * period))) < 0.1 for d in r["downbeats"][:4])
