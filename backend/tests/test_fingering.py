from app.guitar.fingering import FingeringOptimizer, playability_report
from app.guitar.tunings import TUNINGS
from tests.synth import riff_ground_truth


def _notes(seq, dt=0.3):
    return [{"id": f"n{i}", "pitch": p, "start": i * dt, "end": i * dt + dt * 0.9, "confidence": 0.9, "techniques": []}
            for i, p in enumerate(seq)]


def test_open_position_riff_matches_ground_truth():
    gt = riff_ground_truth(bars=8)
    notes = [{"id": i, "pitch": g["pitch"], "start": g["start"], "end": g["end"], "confidence": 0.9, "techniques": []}
             for i, g in enumerate(gt)]
    FingeringOptimizer(TUNINGS["standard"]).optimize(notes)
    exact = sum((n["string"], n["fret"]) == (g["string"], g["fret"]) for n, g in zip(notes, gt))
    assert exact / len(gt) >= 0.95


def test_common_chord_shapes():
    chords = {"C": [48, 52, 55, 60, 64], "G": [43, 47, 50, 55, 59, 67], "E5": [40, 47, 52], "F": [41, 48, 53, 57, 60, 65]}
    expected = {"C": {5: 3, 4: 2, 3: 0, 2: 1, 1: 0}, "E5": {6: 0, 5: 2, 4: 2}, "F": {6: 1, 5: 3, 4: 3, 3: 2, 2: 1, 1: 1}}
    for name, pitches in chords.items():
        notes = [{"pitch": p, "start": 0.0, "end": 1.0, "confidence": 0.9, "techniques": []} for p in pitches]
        FingeringOptimizer(TUNINGS["standard"]).optimize(notes)
        got = {n["string"]: n["fret"] for n in notes}
        assert len(set(got)) == len(pitches)  # one string per note
        if name in expected:
            assert got == expected[name], name


def test_locked_note_is_respected():
    notes = _notes([64, 65, 67])
    notes[1].update({"locked": True, "string": 2, "fret": 6})
    FingeringOptimizer(TUNINGS["standard"]).optimize(notes)
    assert (notes[1]["string"], notes[1]["fret"]) == (2, 6)


def test_unplayable_chord_is_flagged():
    # seven simultaneous notes cannot be played on six strings
    notes = [{"pitch": p, "start": 0.0, "end": 1.0, "confidence": 0.5 + i * 0.01, "techniques": []}
             for i, p in enumerate([40, 45, 50, 55, 59, 64, 69])]
    res = FingeringOptimizer(TUNINGS["standard"]).optimize(notes)
    assert len(res["unplayable"]) >= 1
    assert all("impossible_fingering" in n["flags"] for n in res["unplayable"])


def test_drop_d_low_note():
    notes = _notes([38, 45, 50])
    FingeringOptimizer(TUNINGS["drop_d"]).optimize(notes)
    assert (notes[0]["string"], notes[0]["fret"]) == (6, 0)
    rep = playability_report(notes, TUNINGS["drop_d"])
    assert rep["impossible"] == 0


def test_alternatives_are_offered():
    notes = _notes([57])  # A3 exists on several strings
    FingeringOptimizer(TUNINGS["standard"]).optimize(notes)
    assert len(notes[0]["alternatives"]) >= 2
