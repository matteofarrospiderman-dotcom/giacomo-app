"""Learning from user corrections.

Every manual edit is logged (``corrections.jsonl`` per project) and folded into a
global priors file that subsequent transcriptions use:

* **String/position preferences** - when a user moves a note to another string, the
  ``string_given_pitch`` counts change; the fingering optimiser converts them into a
  log-prior cost, so the next song is fingered the way *this* guitarist plays.
* **Detection thresholds** - deleting many low-confidence or "possible harmonic" notes
  makes detection stricter (higher minimum confidence / ghost-suppression ratio);
  adding many missed notes makes it more sensitive (lower onset threshold).

Within a project, edited notes are *locked*: re-optimising the fingering keeps them
fixed and re-fingers everything around them, so one correction can fix a whole
passage. This is a lightweight, transparent form of human-in-the-loop learning; a
roadmap item is fine-tuning the transcription network on accumulated corrections.
"""

from __future__ import annotations

import copy
import threading

from ..config import settings
from ..storage import read_json, write_json

_LOCK = threading.Lock()

DEFAULTS = {
    "string_given_pitch": {},
    "fret_zone": {},
    "deleted": {"total": 0, "low_conf": 0, "possible_harmonic": 0, "single_source": 0},
    "added": {"total": 0},
    "moved": 0,
    "edits": 0,
}


def _path():
    return settings.learning_dir / "priors.json"


def load_priors() -> dict:
    data = read_json(_path(), default=copy.deepcopy(DEFAULTS))
    for k, v in DEFAULTS.items():
        data.setdefault(k, copy.deepcopy(v))
    return data


def _save(p: dict) -> None:
    write_json(_path(), p)


def record_move(pitch: int, string: int, fret: int) -> None:
    with _LOCK:
        p = load_priors()
        sp = p["string_given_pitch"].setdefault(str(pitch), {})
        sp[str(string)] = sp.get(str(string), 0) + 1
        z = str(min(fret // 5, 4))
        p["fret_zone"][z] = p["fret_zone"].get(z, 0) + 1
        p["moved"] += 1
        p["edits"] += 1
        _save(p)


def record_delete(note: dict) -> None:
    with _LOCK:
        p = load_priors()
        d = p["deleted"]
        d["total"] += 1
        if note.get("confidence", 1) < 0.45:
            d["low_conf"] += 1
        flags = note.get("flags", [])
        if "possible_harmonic" in flags:
            d["possible_harmonic"] += 1
        if "single_source" in flags:
            d["single_source"] += 1
        p["edits"] += 1
        _save(p)


def record_add(note: dict) -> None:
    with _LOCK:
        p = load_priors()
        p["added"]["total"] += 1
        p["edits"] += 1
        _save(p)


def detection_params() -> dict:
    """Thresholds for the next transcription, nudged by accumulated feedback."""
    p = load_priors()
    deleted = p["deleted"]
    added = p["added"]["total"]
    total = max(1, deleted["total"] + added)
    params = {"onset_threshold": 0.5, "frame_threshold": 0.3, "ghost_ratio": 0.66, "min_confidence": 0.25}
    if total >= 10:
        del_rate = deleted["total"] / total
        add_rate = added / total
        params["onset_threshold"] = round(min(0.65, max(0.35, 0.5 + 0.15 * (del_rate - add_rate))), 3)
        params["min_confidence"] = round(min(0.45, 0.25 + 0.2 * deleted["low_conf"] / total), 3)
        params["ghost_ratio"] = round(min(0.85, 0.66 + 0.2 * deleted["possible_harmonic"] / total), 3)
    return params


def summary() -> dict:
    p = load_priors()
    return {"edits": p["edits"], "moved": p["moved"], "deleted": p["deleted"], "added": p["added"],
            "detection_params": detection_params(),
            "pitches_with_preferences": len(p["string_given_pitch"])}
