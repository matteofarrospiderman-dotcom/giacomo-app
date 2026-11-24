"""Full HTTP round-trip: upload -> background processing -> tracks -> edit -> export."""

import io
import time
import zipfile

import guitarpro
import mido
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tests.synth import click_track, render, riff_ground_truth


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def project(client):
    gt = riff_ground_truth(bpm=100, bars=4)
    dur = gt[-1]["end"] + 1.0
    mix = 0.8 * render(gt, dur) + 0.4 * click_track(100, dur)[:, : int(dur * 44100)]
    buf = io.BytesIO()
    sf.write(buf, mix.T, 44100, format="WAV")
    buf.seek(0)
    r = client.post("/api/projects", files={"file": ("riff.wav", buf, "audio/wav")},
                    data={"separator": "dsp", "tuning": "auto"})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        st = client.get(f"/api/projects/{pid}").json()["status"]
        if st["state"] in ("done", "failed"):
            break
        time.sleep(1.0)
    assert st["state"] == "done", st
    return pid


def test_rejects_bad_upload(client):
    r = client.post("/api/projects", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/api/projects", files={"file": ("broken.mp3", b"\x00" * 2000, "audio/mpeg")})
    assert r.status_code == 422


def test_analysis_and_tracks(client, project):
    a = client.get(f"/api/projects/{project}/analysis").json()
    assert abs(a["rhythm"]["bpm"] - 100) < 4
    assert a["rhythm"]["time_signature"] == [4, 4]
    assert "guitar" in a["stems"]
    tracks = client.get(f"/api/projects/{project}/tracks").json()
    gtr = next(t for t in tracks if t["instrument"] == "guitar")
    full = client.get(f"/api/projects/{project}/tracks/{gtr['id']}").json()
    assert full["notes"] and all(n["string"] for n in full["notes"] if "out_of_range" not in n["flags"])
    assert full["tuning"]["id"] == "standard"
    r = client.get(f"/api/projects/{project}/files/{a['stems']['guitar']['preview']}")
    assert r.status_code == 200
    assert client.get(f"/api/projects/{project}/files/../project.json").status_code == 404


def test_edit_and_learning(client, project):
    tid = next(t for t in client.get(f"/api/projects/{project}/tracks").json() if t["instrument"] == "guitar")["id"]
    track = client.get(f"/api/projects/{project}/tracks/{tid}").json()
    n = next(n for n in track["notes"] if n.get("alternatives"))
    alt = n["alternatives"][0]
    r = client.patch(f"/api/projects/{project}/tracks/{tid}/notes/{n['id']}",
                     json={"string": alt["string"], "fret": alt["fret"]}, params={"refinger": True})
    assert r.status_code == 200, r.text
    edited = next(x for x in r.json()["notes"] if x["id"] == n["id"])
    assert (edited["string"], edited["fret"]) == (alt["string"], alt["fret"]) and edited["locked"]
    # impossible string for the pitch is rejected
    r = client.patch(f"/api/projects/{project}/tracks/{tid}/notes/{n['id']}", json={"string": 1, "fret": 30})
    assert r.status_code == 400
    r = client.delete(f"/api/projects/{project}/tracks/{tid}/notes/{track['notes'][-1]['id']}")
    assert r.status_code == 200
    r = client.post(f"/api/projects/{project}/tracks/{tid}/notes", json={"start": 1.0, "end": 1.3, "string": 3, "fret": 2})
    assert r.status_code == 201
    assert client.get("/api/learning").json()["edits"] >= 3
    r = client.post(f"/api/projects/{project}/tracks/{tid}/settings", json={"tuning": "drop_d"})
    assert r.json()["tuning"]["id"] == "drop_d"
    r = client.post(f"/api/projects/{project}/tracks/{tid}/settings", json={"tuning": "standard", "time_signature": [3, 4]})
    assert r.json()["grid"]["time_signature"] == [3, 4]


def test_exports(client, project):
    base = f"/api/projects/{project}/export"
    r = client.get(f"{base}/gp5")
    assert r.status_code == 200
    song = guitarpro.parse(io.BytesIO(r.content))
    assert song.tracks and song.tracks[0].strings
    r = client.get(f"{base}/musicxml")
    assert r.status_code == 200 and b"<technical><string>" in r.content
    r = client.get(f"{base}/midi")
    assert mido.MidiFile(file=io.BytesIO(r.content)).length > 5
    r = client.get(f"{base}/midi-tracks")
    assert zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    r = client.get(f"{base}/txt")
    assert r.status_code == 200 and b"|" in r.content
    r = client.get(f"{base}/pdf")
    assert r.content[:4] == b"%PDF"
    r = client.get(f"{base}/stems")
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert any("guitar" in n for n in names)


def test_tone_recommendation(client, project):
    r = client.get(f"/api/projects/{project}/tone/guitar/recommend", params={"amp": "boss_katana", "pedals": "delay"})
    assert r.status_code == 200
    assert r.json()["chain"]


# ---------------------------------------------------------------- regressions from code review


def _guitar_tid(client, project):
    return next(t for t in client.get(f"/api/projects/{project}/tracks").json() if t["instrument"] == "guitar")["id"]


def test_validation_rejects_bad_input(client, project):
    tid = _guitar_tid(client, project)
    base = f"/api/projects/{project}/tracks/{tid}"
    assert client.post(f"{base}/notes", json={"start": 1.0, "end": 1.2, "string": 3, "fret": -3}).status_code == 422
    assert client.post(f"{base}/notes", json={"start": 1.0, "end": 1.2, "string": 3}).status_code == 400
    note = client.get(base).json()["notes"][0]
    r = client.patch(f"{base}/notes/{note['id']}", json={"techniques": [{"type": "bend", "semitones": "full"}]})
    assert r.status_code == 400
    assert client.post(f"{base}/settings", json={"time_signature": [3]}).status_code == 400
    assert client.post(f"{base}/settings", json={"time_signature": [7, 16]}).status_code == 400
    r = client.post(f"/api/projects/{project}/reprocess", json={"separator": "bogus"})
    assert r.status_code == 400
    assert client.get(f"/api/projects/{project}").json()["options"]["separator"] != "bogus"
    assert client.post(f"/api/projects/{project}/reprocess", json={"strings": 4}).status_code == 400


def test_same_string_collision_is_refused_and_gp5_stays_valid(client, project):
    tid = _guitar_tid(client, project)
    base = f"/api/projects/{project}/tracks/{tid}"
    track = client.get(base).json()
    a = next(n for n in track["notes"] if n.get("string"))
    # a second note on the same string at the same moment is refused
    r = client.post(f"{base}/notes", json={"start": a["start"] + 0.01, "end": a["end"], "string": a["string"], "fret": 7})
    assert r.status_code == 409
    # a fast legato note 45 ms later quantises onto the same tick: export must still parse
    r = client.post(f"{base}/notes", json={"start": a["start"] + 0.045, "end": a["start"] + 0.2, "string": a["string"],
                                           "fret": (a["fret"] or 0) + 2})
    assert r.status_code == 201
    g = client.get(f"/api/projects/{project}/export/gp5")
    assert g.status_code == 200
    guitarpro.parse(io.BytesIO(g.content))


def test_unicode_name_exports(client, project):
    from app import storage

    meta = storage.load_project(project)
    meta["name"] = "Кино – песня"
    storage.save_project(meta)
    for fmt in ("gp5", "musicxml", "txt", "pdf", "midi"):
        r = client.get(f"/api/projects/{project}/export/{fmt}")
        assert r.status_code == 200, fmt
        assert "filename*=UTF-8''" in r.headers["content-disposition"]


def test_export_respects_track_order_and_quantized_midi_snaps(client, project):
    ids = [t["id"] for t in client.get(f"/api/projects/{project}/tracks").json()]
    if len(ids) >= 2:
        r = client.get(f"/api/projects/{project}/export/midi", params={"tracks": f"{ids[1]},{ids[0]}"})
        mf = mido.MidiFile(file=io.BytesIO(r.content))
        names = [m.name for t in mf.tracks for m in t if m.type == "track_name"]
        assert names[1].startswith(client.get(f"/api/projects/{project}/tracks/{ids[1]}").json()["name"][:10])
    r = client.get(f"/api/projects/{project}/export/midi", params={"quantized": True})
    mf = mido.MidiFile(file=io.BytesIO(r.content))
    ticks = []
    for tr in mf.tracks[1:]:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on":
                ticks.append(t)
    # every onset sits on a 1/32 (60-tick at 480 PPQ) or triplet grid position
    assert all(t % 60 == 0 or t % 40 == 0 for t in ticks)
