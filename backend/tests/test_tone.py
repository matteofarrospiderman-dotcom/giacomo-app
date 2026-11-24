import pytest

from app.tone.analyzer import analyze_tone
from app.tone.recommend import recommend
from tests.fx import delay, distort, reverb, tremolo
from tests.synth import render


@pytest.fixture(scope="module")
def phrase():
    notes = []
    t = 0.3
    for k in range(8):
        for j, p in enumerate([52, 55, 57, 59]):
            notes.append({"pitch": p + (k % 2) * 2, "start": t, "end": t + (0.45 if j < 3 else 1.2), "confidence": 0.9,
                          "vel": 0.5 + 0.4 * ((k * 4 + j) % 3) / 2})
            t += 0.5 if j < 3 else 1.25
        t += 1.2
    dur = t + 1
    return notes, dur, render(notes, dur)


def _fx(res, key):
    return next(e for e in res["effects"] if e["key"] == key)


def test_clean_has_no_time_effects(phrase):
    notes, dur, x = phrase
    r = analyze_tone(x, 44100, notes, 120, dur)
    for k in ("delay", "tremolo"):
        assert not _fx(r, k)["detected"], k
    assert not _fx(r, "reverb")["detected"]


def test_distortion_raises_gain(phrase):
    notes, dur, x = phrase
    clean = analyze_tone(x, 44100, notes, 120, dur)["gain"]["gain_index"]
    dist = analyze_tone(distort(x), 44100, notes, 120, dur)["gain"]["gain_index"]
    assert dist > clean + 2.5


def test_reverb_rt60(phrase):
    notes, dur, x = phrase
    r = _fx(analyze_tone(reverb(x, 1.8, 0.4), 44100, notes, 120, dur), "reverb")
    assert r["detected"]
    assert r["details"]["rt60_s"] == pytest.approx(1.8, rel=0.35)


def test_delay_time_and_feedback(phrase):
    notes, dur, x = phrase
    r = _fx(analyze_tone(delay(x, 375, 0.45, 0.4), 44100, notes, 120, dur), "delay")
    assert r["detected"]
    assert r["details"]["time_ms"] == pytest.approx(375, abs=15)
    assert r["details"]["note_value"] == "dotted 1/8"


def test_tremolo_rate(phrase):
    notes, dur, x = phrase
    r = _fx(analyze_tone(tremolo(x, 5.0, 0.7), 44100, notes, 120, dur), "tremolo")
    assert r["detected"]
    assert r["details"]["rate_hz"] == pytest.approx(5.0, abs=0.4)


def test_recommendation_uses_owned_gear(phrase):
    notes, dur, x = phrase
    tone = analyze_tone(delay(distort(x), 375, 0.45, 0.4), 44100, notes, 120, dur)
    rec = recommend(tone, amp="fender_clean", pedals=["tube_screamer", "delay"])
    blocks = [b["block"] for b in rec["chain"]]
    assert any("Amp" in b for b in blocks)
    assert any(b == "Delay" for b in blocks)
