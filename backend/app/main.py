"""TabForge AI - HTTP API (FastAPI).

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import logging
import math
import re
import secrets
import shutil
import tempfile
import threading
import zipfile
from urllib.parse import quote
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import storage
from .audio import io as aio
from .config import BACKEND_ROOT, ensure_dirs, settings
from .guitar import learning
from .guitar.fingering import all_shapes_for_chord
from .guitar.track import finalize
from .guitar.tunings import TUNINGS, custom_tuning, midi_to_name, tuning_from_dict
from .jobs import JobBusy, jobs
from .tone.recommend import catalog as gear_catalog
from .tone.recommend import recommend

log = logging.getLogger("tabforge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_track_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def track_lock(pid: str, tid: str) -> threading.Lock:
    with _locks_guard:
        return _track_locks.setdefault(f"{pid}/{tid}", threading.Lock())


def _editable(pid: str) -> None:
    """Edits made while a worker rewrites the tracks would be silently overwritten."""
    if jobs.is_running(pid):
        raise HTTPException(409, "The project is being processed; edits are possible once it finishes")


TECH_TYPES = {"hammer_on", "pull_off", "slide", "slide_in", "bend", "bend_release", "prebend_release", "vibrato",
              "palm_mute", "let_ring", "harmonic", "tremolo_picking", "strum"}


def _clean_techniques(raw: list) -> list[dict]:
    """Validate user-supplied techniques so a malformed one can never break an export."""
    out = []
    for t in raw:
        if not isinstance(t, dict) or t.get("type") not in TECH_TYPES:
            raise HTTPException(400, f"unknown technique {t!r}")
        c = {"type": t["type"], "user": True}
        if "bend" in t["type"]:
            try:
                st = float(t.get("semitones", 2))
            except (TypeError, ValueError) as e:
                raise HTTPException(400, "bend semitones must be a number") from e
            if not 0.5 <= st <= 3:
                raise HTTPException(400, "bend semitones must be between 0.5 and 3")
            c["semitones"] = st
        if t["type"] in ("slide", "strum"):
            c["direction"] = t.get("direction") if t.get("direction") in ("up", "down") else "up"
        if t["type"] == "slide_in":
            c["direction"] = t.get("direction") if t.get("direction") in ("from_below", "from_above") else "from_below"
        for k in ("from_pitch", "to_pitch"):
            if isinstance(t.get(k), int):
                c[k] = t[k]
        out.append(c)
    return out


def _string_conflict(track: dict, note: dict, string: int) -> dict | None:
    """Another note already sounding on that string at this note's attack."""
    for m in track["notes"]:
        if m is note or m.get("string") != string or m.get("dead"):
            continue
        if abs(m["start"] - note["start"]) < 0.04:
            return m
    return None


def _finite(*xs: float) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def _lan_address() -> str | None:
    """This machine's address on the local network (no packet is sent)."""
    import socket  # noqa: PLC0415

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            return sk.getsockname()[0]
    except OSError:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    n = jobs.recover()
    if n:
        log.info("re-queued %d interrupted project(s)", n)
    ip = _lan_address()
    if ip:
        log.info("On your phone (same Wi-Fi, server started with --host 0.0.0.0): http://%s:8000", ip)
    yield
    jobs.shutdown()


app = FastAPI(title="TabForge AI", version="1.0.0", lifespan=lifespan,
              description="AI-powered music-to-guitar-tab transcription API")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


def _nf(e: Exception):
    raise HTTPException(404, "Not found") from e


def _project(pid: str) -> dict:
    try:
        return storage.load_project(pid)
    except storage.NotFound as e:
        _nf(e)


def _track(pid: str, tid: str) -> dict:
    try:
        return storage.load_track(pid, tid)
    except storage.NotFound as e:
        _nf(e)


def _analysis(pid: str) -> dict:
    try:
        return storage.load_analysis(pid)
    except storage.NotFound as e:
        raise HTTPException(409, "Analysis not available yet") from e


# ----------------------------------------------------------------------------- system


@app.get("/api/health")
def health() -> dict:
    from .separation.demucs_sep import DemucsSeparator  # noqa: PLC0415
    from .transcription import basic_pitch_onnx  # noqa: PLC0415

    def importable(mod: str) -> bool:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec(mod) is not None

    bp_ok, bp_why = basic_pitch_onnx.available()
    demucs_installed = importable("demucs") and importable("torch")
    return {
        "status": "ok",
        "ffmpeg": aio.has_ffmpeg(),
        "engines": {
            "basic_pitch": {"available": bp_ok, "detail": bp_why},
            "demucs": {"installed": demucs_installed, "model": settings.demucs_model,
                       "local_weights": DemucsSeparator()._local_repo() is not None},
            "crepe": {"installed": importable("torchcrepe")},
            "pyin": {"installed": True},
        },
        "device": settings.resolve_device() if demucs_installed else "cpu",
        "limits": {"max_upload_mb": settings.max_upload_mb, "max_duration_s": settings.max_duration_s,
                   "formats": list(settings.allowed_extensions)},
    }


@app.get("/api/tunings")
def tunings() -> list[dict]:
    return [t.to_dict() for t in TUNINGS.values()]


@app.get("/api/gear")
def gear() -> dict:
    return gear_catalog()


@app.get("/api/learning")
def learning_summary() -> dict:
    return learning.summary()


@app.get("/api/chord-shapes")
def chord_shapes(pitches: str, tuning: str = "standard", max_fret: int = 15) -> dict:
    try:
        ps = [int(p) for p in pitches.split(",") if p.strip()]
    except ValueError as e:
        raise HTTPException(400, "pitches must be comma-separated MIDI numbers") from e
    t = TUNINGS.get(tuning)
    if not t or not 1 <= len(ps) <= t.strings:
        raise HTTPException(400, "invalid tuning or pitch count")
    return {"shapes": all_shapes_for_chord(ps, t, max_fret)}


# ----------------------------------------------------------------------------- projects


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return [{k: p.get(k) for k in ("id", "name", "created_at", "updated_at", "status", "summary", "source")}
            for p in storage.list_projects()]


_SAFE_NAME = re.compile(r"[^\w\-. ()]+")


@app.post("/api/projects", status_code=201)
async def create_project(file: UploadFile = File(...), name: str | None = Form(None),
                         separator: str = Form("auto"), tuning: str = Form("auto"), strings: int | None = Form(None),
                         transcribe_bass: bool = Form(True), split_guitars: bool = Form(True)) -> dict:
    fname = file.filename or "upload"
    ext = Path(fname).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}")
    if separator not in ("auto", "demucs", "dsp"):
        raise HTTPException(400, "separator must be auto|demucs|dsp")
    if tuning != "auto" and tuning not in TUNINGS:
        raise HTTPException(400, "unknown tuning")
    if strings is not None and strings not in (6, 7, 8):
        raise HTTPException(400, "strings must be 6, 7 or 8 (guitar)")
    display = _SAFE_NAME.sub("", name or Path(fname).stem)[:120] or "Untitled"
    meta = storage.create_project(display, {"separator": separator, "tuning": tuning, "strings": strings,
                                            "transcribe_bass": transcribe_bass, "split_guitars": split_guitars})
    pdir = storage.project_dir(meta["id"])
    dest = pdir / f"source{ext}"
    limit = settings.max_upload_mb * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")
        info = aio.probe(dest)
        if info.duration and info.duration > settings.max_duration_s + 1:
            raise HTTPException(413, f"Audio is longer than {settings.max_duration_s / 60:.0f} minutes")
        if info.duration and info.duration < settings.min_duration_s:
            raise HTTPException(422, f"Audio is shorter than {settings.min_duration_s:.0f} s")
    except HTTPException:
        shutil.rmtree(pdir, ignore_errors=True)
        raise
    except aio.AudioError as e:
        shutil.rmtree(pdir, ignore_errors=True)
        raise HTTPException(422, str(e)) from e
    except BaseException:
        # client disconnect, disk full...: never leave a half-created project behind
        shutil.rmtree(pdir, ignore_errors=True)
        raise
    meta = storage.load_project(meta["id"])
    meta["source"] = {"file": dest.name, "original_name": fname, "bytes": size, "probe": info.to_dict()}
    storage.save_project(meta)
    try:
        jobs.submit("process", meta["id"])
    except Exception as e:  # noqa: BLE001
        storage.set_status(meta["id"], "failed", "queued", 0.0, f"Could not start processing: {e}")
        raise HTTPException(503, f"Could not start processing: {e}") from e
    return storage.load_project(meta["id"])


@app.get("/api/projects/{pid}")
def get_project(pid: str) -> dict:
    meta = _project(pid)
    meta["running"] = jobs.is_running(pid)
    return meta


@app.delete("/api/projects/{pid}", status_code=204)
def delete_project(pid: str) -> Response:
    if jobs.is_running(pid):
        raise HTTPException(409, "Project is processing")
    try:
        storage.delete_project(pid)
    except storage.NotFound as e:
        _nf(e)
    return Response(status_code=204)


class ReprocessBody(BaseModel):
    separator: str | None = None
    tuning: str | None = None
    strings: int | None = None
    transcribe_bass: bool | None = None
    split_guitars: bool | None = None


@app.post("/api/projects/{pid}/reprocess")
def reprocess(pid: str, body: ReprocessBody) -> dict:
    meta = _project(pid)
    if jobs.is_running(pid):
        raise HTTPException(409, "A job is already running for this project")
    opts = body.model_dump(exclude_none=True)
    if opts.get("separator", "auto") not in ("auto", "demucs", "dsp"):
        raise HTTPException(400, "separator must be auto|demucs|dsp")
    if opts.get("tuning", "auto") != "auto" and opts["tuning"] not in TUNINGS:
        raise HTTPException(400, "unknown tuning")
    if "strings" in opts and opts["strings"] not in (6, 7, 8):
        raise HTTPException(400, "strings must be 6, 7 or 8 (guitar)")
    meta["options"].update(opts)
    storage.save_project(meta)
    try:
        jobs.submit("process", pid)
    except JobBusy as e:
        raise HTTPException(409, str(e)) from e
    return storage.load_project(pid)


@app.get("/api/projects/{pid}/analysis")
def get_analysis(pid: str) -> dict:
    _project(pid)
    return _analysis(pid)


ALLOWED_FILE = re.compile(r"^(audio|stems)/[A-Za-z0-9_\-]+(\.peaks)?\.(wav|mp3|json)$")


@app.get("/api/projects/{pid}/files/{path:path}")
def get_file(pid: str, path: str):
    _project(pid)
    if not ALLOWED_FILE.match(path):
        raise HTTPException(404, "Not found")
    f = storage.project_dir(pid) / path
    if not f.exists():
        raise HTTPException(404, "Not found")
    media = {"wav": "audio/wav", "mp3": "audio/mpeg", "json": "application/json"}[f.suffix[1:]]
    return FileResponse(f, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


class TranscribeBody(BaseModel):
    stem: str
    instrument: str = Field("guitar", pattern="^(guitar|bass)$")
    tuning: str = "auto"


@app.post("/api/projects/{pid}/transcribe")
def transcribe_stem(pid: str, body: TranscribeBody) -> dict:
    _project(pid)
    analysis = _analysis(pid)
    if body.stem not in analysis.get("stems", {}):
        raise HTTPException(404, "Unknown stem")
    if body.tuning != "auto" and body.tuning not in TUNINGS:
        raise HTTPException(400, "unknown tuning")
    try:
        jobs.submit("transcribe_stem", pid, body.model_dump())
    except JobBusy as e:
        raise HTTPException(409, str(e)) from e
    return storage.load_project(pid)


# ----------------------------------------------------------------------------- tracks


@app.get("/api/projects/{pid}/tracks")
def list_tracks(pid: str) -> list[dict]:
    _project(pid)
    return [{k: t.get(k) for k in ("id", "name", "stem", "instrument", "role", "tuning", "stats", "version")}
            for t in storage.list_tracks(pid)]


@app.get("/api/projects/{pid}/tracks/{tid}")
def get_track(pid: str, tid: str) -> dict:
    _project(pid)
    return _track(pid, tid)


def _refinalize(pid: str, track: dict) -> dict:
    analysis = _analysis(pid)
    priors = learning.load_priors()
    finalize(track, analysis["rhythm"], analysis["duration"], priors=priors, chords=analysis.get("chords"))
    storage.save_track(pid, track)
    return track


def _requantize_only(pid: str, track: dict) -> dict:
    """Cheap update after a local edit: keep all other fingerings, refresh rhythm/notation/review."""
    from .guitar.quantize import build_measures, make_grid, measures_info, quantize  # noqa: PLC0415
    from .guitar.track import build_review  # noqa: PLC0415

    analysis = _analysis(pid)
    rh = analysis["rhythm"]
    grid = make_grid(rh, analysis["duration"], tuple(track.get("time_signature") or rh["time_signature"]),
                     int(track.get("downbeat_offset", 0)))
    info = quantize(track["notes"], grid)
    n_measures = max([n.get("measure", 0) for n in track["notes"]] + [0]) + 1
    track["measures"] = measures_info(grid, info.get("first_bar_beat", 0), n_measures)
    track["notation"] = build_measures(track["notes"], grid, track["measures"])
    track["grid"]["first_bar_beat"] = info.get("first_bar_beat", 0)
    track["review"] = build_review(track)
    lv = track["stats"]["levels"]
    for k in lv:
        lv[k] = sum(1 for n in track["notes"] if n.get("level") == k)
    track["stats"]["notes"] = len(track["notes"])
    track["stats"]["flagged"] = sum(1 for n in track["notes"] if n.get("flags"))
    track["version"] = int(track.get("version", 0)) + 1
    storage.save_track(pid, track)
    return track


class NoteEdit(BaseModel):
    string: int | None = None
    fret: int | None = None
    pitch: int | None = None
    start: float | None = None
    end: float | None = None
    techniques: list[dict] | None = None
    velocity: int | None = None
    locked: bool | None = None
    dead: bool | None = None


def _tuning_eff(track: dict):
    t = tuning_from_dict(track["tuning"])
    capo = int(track.get("capo", 0))
    return t, capo


@app.patch("/api/projects/{pid}/tracks/{tid}/notes/{nid}")
def edit_note(pid: str, tid: str, nid: str, body: NoteEdit, refinger: bool = False) -> dict:
    _editable(pid)
    with track_lock(pid, tid):
        track = _track(pid, tid)
        note = next((n for n in track["notes"] if n["id"] == nid), None)
        if note is None:
            raise HTTPException(404, "Note not found")
        before = dict(note)
        tun, capo = _tuning_eff(track)
        max_fret = int(track.get("max_fret", 24))
        changes = body.model_dump(exclude_none=True)
        moved = False
        # validate and resolve everything first: nothing changes if a check fails
        if "start" in changes or "end" in changes:
            st = changes.get("start", note["start"])
            en = changes.get("end", note["end"])
            if not _finite(st, en) or not 0 <= st < en <= _analysis(pid)["duration"] + 1:
                raise HTTPException(400, "invalid timing")
        if "techniques" in changes:
            changes["techniques"] = _clean_techniques(changes["techniques"])
        if "velocity" in changes and not 1 <= changes["velocity"] <= 127:
            raise HTTPException(400, "velocity must be 1-127")
        new_pitch = note["pitch"]
        if "pitch" in changes:
            if not 20 <= changes["pitch"] <= 108:
                raise HTTPException(400, "pitch out of range")
            new_pitch = changes["pitch"]
        new_string, new_fret = note.get("string"), note.get("fret")
        if "string" in changes or "fret" in changes:
            s_ = changes.get("string")
            if s_ is None and "pitch" in changes and "fret" in changes:
                # pitch + fret: the string that produces that pitch at that fret
                s_ = next((k for k in range(1, tun.strings + 1)
                           if tun.open_midi(k) + capo + changes["fret"] == new_pitch), None)
                if s_ is None:
                    raise HTTPException(400, "no string plays that pitch at that fret")
            if s_ is None:
                s_ = note.get("string") or 1
            if not 1 <= s_ <= tun.strings:
                raise HTTPException(400, "string out of range")
            open_p = tun.open_midi(s_) + capo
            if "fret" in changes:
                f = int(changes["fret"])
                if not 0 <= f <= max_fret:
                    raise HTTPException(400, "fret out of range")
                new_pitch = open_p + f
            else:
                f = new_pitch - open_p
                if not 0 <= f <= max_fret:
                    raise HTTPException(400, f"{midi_to_name(new_pitch)} cannot be played on string {s_}")
            other = _string_conflict(track, note, s_)
            if other is not None:
                raise HTTPException(409, f"String {s_} already plays {other.get('name', '')} at this moment")
            new_string, new_fret, moved = s_, f, True
        elif "pitch" in changes:
            new_string = new_fret = None  # let the optimiser place the corrected pitch
            refinger = True
        # apply
        if new_pitch != note["pitch"]:
            note["pitch"] = new_pitch
            note["name"] = midi_to_name(new_pitch)
            note.pop("sounding_pitch", None)
            # the user confirmed the pitch, so pitch-evidence flags no longer apply
            note["flags"] = [f for f in note.get("flags", []) if f not in
                             ("ambiguous_pitch", "octave_disagreement", "possible_harmonic", "weak_evidence",
                              "single_source", "out_of_range")]
        note["string"], note["fret"] = new_string, new_fret
        if moved:
            learning.record_move(note["pitch"], new_string, new_fret)
            note["flags"] = [f for f in note.get("flags", []) if f not in ("uncertain_string", "impossible_fingering")]
        if "start" in changes or "end" in changes:
            note["start"] = round(changes.get("start", note["start"]), 4)
            note["end"] = round(changes.get("end", note["end"]), 4)
            note["flags"] = [f for f in note.get("flags", []) if f != "unusual_timing"]
        if "techniques" in changes:
            note["techniques"] = changes["techniques"]
        if "velocity" in changes:
            note["velocity"] = int(changes["velocity"])
        if "dead" in changes:
            note["dead"] = bool(changes["dead"])
        note["edited"] = True
        note["locked"] = changes.get("locked", True) if new_string is not None else False
        note["level"] = "user"
        note["confidence"] = max(note.get("confidence", 0), 0.99)
        storage.append_correction(pid, {"track": tid, "action": "edit", "note": nid, "before": before, "after": note})
        return _refinalize(pid, track) if refinger else _requantize_only(pid, track)


class NoteCreate(BaseModel):
    start: float
    end: float
    pitch: int | None = Field(None, ge=20, le=108)
    string: int | None = Field(None, ge=1, le=9)
    fret: int | None = Field(None, ge=0, le=36)
    velocity: int = Field(90, ge=1, le=127)
    techniques: list[dict] = []


@app.post("/api/projects/{pid}/tracks/{tid}/notes", status_code=201)
def add_note(pid: str, tid: str, body: NoteCreate) -> dict:
    _editable(pid)
    with track_lock(pid, tid):
        track = _track(pid, tid)
        tun, capo = _tuning_eff(track)
        if not _finite(body.start, body.end) or body.end <= body.start or body.start < 0 \
                or body.end > _analysis(pid)["duration"] + 1:
            raise HTTPException(400, "invalid timing")
        pitch = body.pitch
        if body.string is not None and body.fret is not None:
            if not 1 <= body.string <= tun.strings:
                raise HTTPException(400, "string out of range")
            if body.fret > int(track.get("max_fret", 24)):
                raise HTTPException(400, "fret out of range")
            pitch = tun.open_midi(body.string) + capo + body.fret
            other = _string_conflict(track, {"start": body.start}, body.string)
            if other is not None:
                raise HTTPException(409, f"String {body.string} already plays {other.get('name', '')} at this moment")
        elif body.string is not None or body.fret is not None:
            raise HTTPException(400, "give both string and fret, or only a pitch")
        if pitch is None:
            raise HTTPException(400, "pitch or string+fret required")
        techniques = _clean_techniques(body.techniques)
        nid = "u" + secrets.token_hex(4)
        note = {"id": nid, "start": round(body.start, 4), "end": round(body.end, 4), "pitch": int(pitch),
                "name": midi_to_name(int(pitch)), "velocity": body.velocity, "confidence": 1.0, "level": "user",
                "sources": {"user": 1.0}, "flags": [], "techniques": techniques, "string": body.string,
                "fret": body.fret, "locked": body.string is not None, "edited": True}
        track["notes"].append(note)
        track["notes"].sort(key=lambda n: (n["start"], n["pitch"]))
        learning.record_add(note)
        storage.append_correction(pid, {"track": tid, "action": "add", "note": nid, "after": note})
        return _refinalize(pid, track)


@app.delete("/api/projects/{pid}/tracks/{tid}/notes/{nid}")
def delete_note(pid: str, tid: str, nid: str) -> dict:
    _editable(pid)
    with track_lock(pid, tid):
        track = _track(pid, tid)
        note = next((n for n in track["notes"] if n["id"] == nid), None)
        if note is None:
            raise HTTPException(404, "Note not found")
        track["notes"] = [n for n in track["notes"] if n["id"] != nid]
        track.setdefault("rejected", []).append({"start": note["start"], "end": note["end"], "pitch": note["pitch"],
                                                 "confidence": note.get("confidence", 0), "reason": "deleted_by_user"})
        learning.record_delete(note)
        storage.append_correction(pid, {"track": tid, "action": "delete", "note": nid, "before": note})
        return _requantize_only(pid, track)


class RestoreBody(BaseModel):
    index: int


@app.post("/api/projects/{pid}/tracks/{tid}/restore")
def restore_rejected(pid: str, tid: str, body: RestoreBody) -> dict:
    _editable(pid)
    with track_lock(pid, tid):
        track = _track(pid, tid)
        rej = track.get("rejected", [])
        if not 0 <= body.index < len(rej):
            raise HTTPException(404, "No such rejected note")
        r = rej.pop(body.index)
        nid = "r" + secrets.token_hex(4)
        storage.append_correction(pid, {"track": tid, "action": "restore", "note": nid, "after": r})
        track["notes"].append({"id": nid, "start": r["start"], "end": r["end"], "pitch": r["pitch"],
                               "name": midi_to_name(r["pitch"]), "velocity": 80, "confidence": r.get("confidence", 0.3),
                               "level": "user", "sources": {"restored": 1.0}, "flags": [], "techniques": [],
                               "string": None, "fret": None, "locked": False, "edited": True})
        track["notes"].sort(key=lambda n: (n["start"], n["pitch"]))
        learning.record_add({"pitch": r["pitch"]})
        return _refinalize(pid, track)


class TrackSettings(BaseModel):
    tuning: str | None = None
    custom_tuning: list[int] | None = None
    capo: int | None = Field(None, ge=0, le=12)
    time_signature: list[int] | None = None
    downbeat_offset: int | None = Field(None, ge=0, le=12)
    profile: str | None = Field(None, pattern="^(balanced|open_position|minimal_movement)$")
    name: str | None = None
    keep_locked: bool = True


@app.post("/api/projects/{pid}/tracks/{tid}/settings")
def track_settings(pid: str, tid: str, body: TrackSettings) -> dict:
    _editable(pid)
    with track_lock(pid, tid):
        track = _track(pid, tid)
        if body.custom_tuning:
            try:
                track["tuning"] = custom_tuning(body.custom_tuning, instrument=track["instrument"]).to_dict()
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            body.keep_locked = False
        elif body.tuning:
            t = TUNINGS.get(body.tuning)
            if t is None:
                raise HTTPException(400, "unknown tuning")
            track["tuning"] = t.to_dict()
            body.keep_locked = False
        if body.capo is not None:
            track["capo"] = body.capo
            body.keep_locked = False
        if body.time_signature:
            if len(body.time_signature) != 2:
                raise HTTPException(400, "time signature must be [beats, beat unit]")
            num, den = body.time_signature
            if not (1 <= num <= 16 and den in (2, 4, 8)):
                raise HTTPException(400, "invalid time signature (beat unit must be 2, 4 or 8)")
            track["time_signature"] = [num, den]
        if body.downbeat_offset is not None:
            track["downbeat_offset"] = body.downbeat_offset
        if body.profile:
            track["profile"] = body.profile
        if body.name:
            track["name"] = _SAFE_NAME.sub("", body.name)[:60] or track["name"]
        if not body.keep_locked:
            # string/fret locks are meaningless after a tuning or capo change
            for n in track["notes"]:
                n["locked"] = False
        return _refinalize(pid, track)


@app.post("/api/projects/{pid}/tracks/{tid}/reoptimize")
def reoptimize(pid: str, tid: str) -> dict:
    """Re-run the fingering optimiser keeping user-locked notes fixed."""
    _editable(pid)
    with track_lock(pid, tid):
        return _refinalize(pid, _track(pid, tid))


# ----------------------------------------------------------------------------- tone


@app.get("/api/projects/{pid}/tone/{stem}/recommend")
def tone_recommend(pid: str, stem: str, amp: str = "generic", pedals: str = "", guitar: str | None = None) -> dict:
    _project(pid)
    analysis = _analysis(pid)
    tone = analysis.get("tone", {}).get(stem)
    if tone is None:
        raise HTTPException(404, "No tone analysis for this stem")
    return recommend(tone, amp, [p for p in pedals.split(",") if p], guitar)


# ----------------------------------------------------------------------------- export


def _export_tracks(pid: str, tracks_param: str | None) -> list[dict]:
    all_tracks = storage.list_tracks(pid)
    if not all_tracks:
        raise HTTPException(409, "No tracks to export")
    if tracks_param:
        want = [t for t in tracks_param.split(",") if t]
        by_id = {t["id"]: t for t in all_tracks}
        sel = [by_id[w] for w in want if w in by_id]  # keep the requested order: the first sets the grid
        if not sel:
            raise HTTPException(404, "No matching tracks")
        return sel
    return all_tracks


def _disposition(name: str) -> dict:
    """Content-Disposition that survives non-Latin song names (RFC 6266 filename*)."""
    ascii_name = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "")
    return {"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"}


def _fname(meta: dict, ext: str, suffix: str = "") -> str:
    base = re.sub(r"[^\w\-]+", "_", meta["name"]).strip("_") or "tabforge"
    return f"{base}{suffix}.{ext}"


@app.get("/api/projects/{pid}/export/{fmt}")
def export(pid: str, fmt: str, tracks: str | None = Query(None), quantized: bool = False):
    meta = _project(pid)
    analysis = _analysis(pid)
    sel = _export_tracks(pid, tracks) if fmt != "stems" else []
    title = meta["name"]
    tones = analysis.get("tone", {})
    if fmt == "gp5":
        from .export.gp5 import export_gp5  # noqa: PLC0415

        if any(len(t["tuning"]["notes"]) > 7 for t in sel):
            raise HTTPException(422, "Guitar Pro 5 supports at most 7 strings; export 8-string tracks as MusicXML or MIDI")

        data = export_gp5(sel, analysis, title, tones)
        return Response(data, media_type="application/octet-stream",
                        headers=_disposition(_fname(meta, "gp5")))
    if fmt == "musicxml":
        from .export.musicxml import export_musicxml  # noqa: PLC0415

        data = export_musicxml(sel, analysis, title).encode("utf-8")
        return Response(data, media_type="application/vnd.recordare.musicxml+xml",
                        headers=_disposition(_fname(meta, "musicxml")))
    if fmt == "midi":
        from .export.midi import export_midi  # noqa: PLC0415

        data = export_midi(sel, analysis, quantized=quantized, tones=tones)
        return Response(data, media_type="audio/midi",
                        headers=_disposition(_fname(meta, "mid")))
    if fmt == "midi-tracks":
        from .export.midi import export_midi  # noqa: PLC0415

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for t in sel:
                z.writestr(_fname(meta, "mid", f"_{t['id']}"), export_midi([t], analysis, quantized=quantized,
                                                                           per_string=True, tones=tones))
        return Response(buf.getvalue(), media_type="application/zip",
                        headers=_disposition(_fname(meta, "zip", "_midi")))
    if fmt == "txt":
        from .export.ascii_tab import export_text  # noqa: PLC0415

        return Response(export_text(sel, analysis, title), media_type="text/plain; charset=utf-8",
                        headers=_disposition(_fname(meta, "txt")))
    if fmt == "pdf":
        from .export.pdf import export_pdf  # noqa: PLC0415

        return Response(export_pdf(sel, analysis, title), media_type="application/pdf",
                        headers=_disposition(_fname(meta, "pdf")))
    if fmt == "stems":
        # stream from a temporary file: stems of a long song are far too big to zip in memory
        pdir = storage.project_dir(pid)
        tmp = tempfile.NamedTemporaryFile(prefix="tabforge-stems-", suffix=".zip", delete=False)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for name, info in analysis.get("stems", {}).items():
                f = pdir / info["wav"]
                if f.exists():
                    z.write(f, f"{_fname(meta, 'wav', '_' + name)}")
        tmp.close()
        return FileResponse(tmp.name, media_type="application/zip", headers=_disposition(_fname(meta, "zip", "_stems")),
                            background=BackgroundTask(lambda p=tmp.name: Path(p).unlink(missing_ok=True)))
    raise HTTPException(404, "Unknown export format")


@app.exception_handler(storage.NotFound)
async def _not_found(_req, _exc):
    return JSONResponse({"detail": "Not found"}, status_code=404)


# ----------------------------------------------------------------------------- frontend (production build)

_dist = BACKEND_ROOT.parent / "frontend" / "dist"
if (_dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=str(_dist / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "Not found")
        f = (_dist / full_path).resolve()
        if full_path and f.is_file() and _dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(_dist / "index.html")
