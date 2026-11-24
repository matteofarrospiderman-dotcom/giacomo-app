"""Filesystem-backed project storage.

Layout of one project::

    data/projects/<id>/
        project.json          metadata + processing status (atomic writes)
        source.<ext>          the original upload, untouched
        audio/mix.wav         decoded 44.1 kHz stereo master
        audio/mix.mp3         browser playback copy
        audio/peaks.json      waveform overview for the UI
        stems/<name>.wav|mp3  separated sources
        analysis.json         tempo, meter, key, chords, tuning, instruments, tone
        tracks/<track>.json   guitar-aware transcriptions (editable)
        corrections.jsonl     append-only log of user edits

JSON is written to a temp file and renamed so a crash never leaves half a file, and
the API process can read status written by worker processes without locking.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .config import settings

_ID_RE = re.compile(r"^[a-f0-9]{12}$")
_TRACK_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


class NotFound(Exception):
    pass


def new_project_id() -> str:
    return uuid.uuid4().hex[:12]


def project_dir(project_id: str) -> Path:
    if not _ID_RE.match(project_id):
        raise NotFound(project_id)
    return settings.projects_dir / project_id


def safe_track_id(track_id: str) -> str:
    if not _TRACK_RE.match(track_id):
        raise NotFound(track_id)
    return track_id


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"), default=_json_default)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _json_default(obj: Any) -> Any:
    # numpy scalars / arrays sneak into analysis results; convert them transparently
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        if default is not None:
            return default
        raise NotFound(str(path)) from None


# ----------------------------------------------------------------------------- projects


def create_project(name: str, options: dict[str, Any]) -> dict[str, Any]:
    pid = new_project_id()
    pdir = project_dir(pid)
    (pdir / "audio").mkdir(parents=True)
    (pdir / "stems").mkdir()
    (pdir / "tracks").mkdir()
    meta = {
        "id": pid,
        "name": name,
        "created_at": time.time(),
        "updated_at": time.time(),
        "options": options,
        "status": {"state": "uploaded", "stage": "queued", "progress": 0.0, "message": "Waiting for worker", "log": []},
        "source": None,
        "summary": None,
    }
    write_json(pdir / "project.json", meta)
    return meta


def load_project(project_id: str) -> dict[str, Any]:
    return read_json(project_dir(project_id) / "project.json")


def save_project(meta: dict[str, Any]) -> None:
    meta["updated_at"] = time.time()
    write_json(project_dir(meta["id"]) / "project.json", meta)


def update_project(project_id: str, **fields: Any) -> dict[str, Any]:
    meta = load_project(project_id)
    meta.update(fields)
    save_project(meta)
    return meta


def list_projects() -> list[dict[str, Any]]:
    out = []
    if not settings.projects_dir.exists():
        return out
    for p in settings.projects_dir.iterdir():
        f = p / "project.json"
        if f.exists():
            try:
                out.append(read_json(f))
            except (json.JSONDecodeError, NotFound):
                continue
    out.sort(key=lambda m: m.get("created_at", 0), reverse=True)
    return out


def delete_project(project_id: str) -> None:
    pdir = project_dir(project_id)
    if not pdir.exists():
        raise NotFound(project_id)
    shutil.rmtree(pdir)


def set_status(project_id: str, state: str, stage: str, progress: float, message: str = "") -> None:
    meta = load_project(project_id)
    st = meta.setdefault("status", {})
    st.update({"state": state, "stage": stage, "progress": round(float(progress), 4), "message": message})
    log = st.setdefault("log", [])
    if not log or log[-1].get("stage") != stage or log[-1].get("message") != message:
        log.append({"t": time.time(), "stage": stage, "message": message})
        del log[:-60]
    save_project(meta)


# ----------------------------------------------------------------------------- analysis/tracks


def load_analysis(project_id: str) -> dict[str, Any]:
    return read_json(project_dir(project_id) / "analysis.json")


def save_analysis(project_id: str, analysis: dict[str, Any]) -> None:
    write_json(project_dir(project_id) / "analysis.json", analysis)


def track_path(project_id: str, track_id: str) -> Path:
    return project_dir(project_id) / "tracks" / f"{safe_track_id(track_id)}.json"


def load_track(project_id: str, track_id: str) -> dict[str, Any]:
    return read_json(track_path(project_id, track_id))


def save_track(project_id: str, track: dict[str, Any]) -> None:
    write_json(track_path(project_id, track["id"]), track)


def list_tracks(project_id: str) -> list[dict[str, Any]]:
    tdir = project_dir(project_id) / "tracks"
    out = []
    if tdir.exists():
        for f in sorted(tdir.glob("*.json")):
            try:
                out.append(read_json(f))
            except (json.JSONDecodeError, NotFound):
                continue
    return out


def append_correction(project_id: str, record: dict[str, Any]) -> None:
    path = project_dir(project_id) / "corrections.jsonl"
    record = {"t": time.time(), **record}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")
