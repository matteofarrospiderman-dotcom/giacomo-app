import numpy as np

from app.guitar.quantize import build_measures, decompose, make_grid, measures_info, quantize


def test_decompose_values():
    assert [(d["value"], d["dotted"]) for d in decompose(1440)] == [(4, True)]
    assert [(d["value"], d["tuplet"]) for d in decompose(320)] == [(8, [3, 2])]
    assert sum(d["ticks"] for d in decompose(1600)) == 1600


def _rhythm(bpm=120.0, n=40):
    beats = list(np.arange(n) * 60 / bpm)
    return {"bpm": bpm, "beats": beats, "downbeats": beats[::4], "time_signature": [4, 4]}


def test_straight_and_triplet_quantization():
    rh = _rhythm()
    grid = make_grid(rh, 20.0)
    eighths = [{"id": f"a{i}", "start": i * 0.25 + 0.004, "end": i * 0.25 + 0.2, "pitch": 60} for i in range(8)]
    triplets = [{"id": f"t{i}", "start": 4.0 + i * (0.5 / 3), "end": 4.0 + i * (0.5 / 3) + 0.1, "pitch": 62} for i in range(6)]
    notes = eighths + triplets
    info = quantize(notes, grid)
    ticks = [n["tick"] for n in eighths]
    assert ticks == [i * 480 for i in range(8)]
    tt = [n["tick"] for n in triplets]
    assert all(b - a == 320 for a, b in zip(tt[:-1], tt[1:]))
    meas = measures_info(grid, info["first_bar_beat"], 3)
    notation = build_measures(notes, grid, meas)
    for m in notation:
        assert sum(b["ticks"] for b in m["beats"]) == 3840  # every 4/4 bar is complete
