from app.analysis.tuning import detect_tuning


def _notes(pitches, dt=0.25):
    return [{"pitch": p, "start": i * dt, "end": i * dt + dt * 0.9, "confidence": 0.9, "techniques": []}
            for i, p in enumerate(pitches)]


def test_standard_tuning_for_open_e_riff():
    riff = [40, 43, 45, 47, 50, 52, 47, 45] * 6
    res = detect_tuning(_notes(riff), "guitar")
    assert res["best"]["id"] == "standard"


def test_drop_d_detected_from_low_d_power_chords():
    notes = []
    t = 0.0
    for k in range(24):
        root = 38 if k % 2 == 0 else 41
        for p in (root, root + 7, root + 12):
            notes.append({"pitch": p, "start": t, "end": t + 0.22, "confidence": 0.9, "techniques": []})
        t += 0.25
    res = detect_tuning(notes, "guitar")
    assert res["best"]["id"] in ("drop_d", "d_standard", "double_drop_d")
    assert res["best"]["notes"][0] == 38
