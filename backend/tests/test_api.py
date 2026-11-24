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
