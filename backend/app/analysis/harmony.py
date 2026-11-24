"""Key and chord recognition.

Chords
------
* Beat-synchronous CQT chroma, computed on the harmonic content (drums and vocals
  removed when stems exist). A bass chroma from the bass stem (or low-passed harmonic
  content) is computed separately so inversions/slash chords can be named.
* Each beat segment is scored against binary templates for 12 roots x 11 qualities
  (maj, min, 5, 7, maj7, m7, sus2, sus4, dim, aug, add9) with harmonic-weighted
  templates (the 3rd/5th partials of each chord tone also light up chroma bins).
* An HMM with a strong self-transition (chords persist) is decoded with Viterbi so
  single-beat flickers are suppressed.
* Confidence = normalised template margin between the best and runner-up chord.

``cross_validate_with_notes`` later compares these audio-derived chords with the chords
implied by the guitar transcription; agreement raises confidence, disagreement flags.

Key: Krumhansl-Kessler profiles correlated with the duration-weighted chroma.
"""

from __future__ import annotations

import numpy as np
import librosa

from ..guitar.tunings import NOTE_NAMES_FLAT, NOTE_NAMES_SHARP

HOP = 512

QUALITIES: dict[str, list[int]] = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "5": [0, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "add9": [0, 2, 4, 7],
}
# simpler chords are a priori more likely
QUALITY_PRIOR = {"": 1.0, "m": 0.95, "5": 0.8, "7": 0.55, "maj7": 0.4, "m7": 0.5, "sus2": 0.35, "sus4": 0.35,
                 "dim": 0.2, "aug": 0.12, "add9": 0.3}

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"}


def _templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    labels = []
    T = []
    for q, ivs in QUALITIES.items():
        for root in range(12):
            v = np.zeros(12)
            for k, iv in enumerate(ivs):
                pc = (root + iv) % 12
                v[pc] += 1.0 if k == 0 else 0.85
                v[(pc + 7) % 12] += 0.12  # 3rd harmonic leaks into the fifth above
                v[(pc + 4) % 12] += 0.05  # 5th harmonic
            v /= np.linalg.norm(v)
            T.append(v)
            labels.append((root, q))
    return np.array(T), labels


TEMPLATES, LABELS = _templates()


def chord_name(root: int, quality: str, bass: int | None = None, flats: bool = False) -> str:
    names = NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP
    s = names[root] + quality
    if bass is not None and bass != root:
        s += "/" + names[bass]
    return s


def estimate_key(chroma_mean: np.ndarray) -> dict:
    best = None
    scores = []
    for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(np.roll(prof, tonic), chroma_mean)[0, 1])
            scores.append((r, tonic, mode))
    scores.sort(key=lambda x: -x[0])
    r, tonic, mode = scores[0]
    second = scores[1][0]
    name_s = NOTE_NAMES_SHARP[tonic] + ("m" if mode == "minor" else "")
    name_f = NOTE_NAMES_FLAT[tonic] + ("m" if mode == "minor" else "")
    flats = name_f in FLAT_KEYS
    best = {
        "tonic": tonic,
        "mode": mode,
        "name": (NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP)[tonic] + (" minor" if mode == "minor" else " major"),
        "use_flats": flats,
        "correlation": round(r, 3),
        "confidence": round(float(np.clip((r - second) * 6 + 0.25 * max(r, 0), 0, 1)), 3),
        "alternatives": [
            {"name": (NOTE_NAMES_SHARP[t] + (" minor" if m == "minor" else " major")), "correlation": round(c, 3)}
            for c, t, m in scores[1:4]
        ],
        "_corr": {f"{t}{m}": c for c, t, m in scores},
    }
    del name_s
    return best


def fuse_key(chroma_key: dict, notes: list[dict]) -> dict:
    """Combine the chroma key estimate with one from transcribed pitch classes.

    Chroma is biased by overtones (the 5th harmonic of E is G#, pushing E minor towards
    E major); transcribed notes do not have that bias but contain detection errors.
    Correlations from both are averaged per key candidate.
    """
    prof = np.zeros(12)
    for n in notes:
        w = max(0.05, n["end"] - n["start"]) * n.get("confidence", 0.5)
        prof[n["pitch"] % 12] += w
    if prof.sum() <= 0:
        return chroma_key
    best = None
    for mode, kk in (("major", KK_MAJOR), ("minor", KK_MINOR)):
        for tonic in range(12):
            r_notes = float(np.corrcoef(np.roll(kk, tonic), prof)[0, 1])
            r_chroma = chroma_key.get("_corr", {}).get(f"{tonic}{mode}", r_notes)
            # the third decides the mode: transcribed notes have no overtone bias here
            w_maj, w_min = prof[(tonic + 4) % 12], prof[(tonic + 3) % 12]
            third = (w_maj - w_min) / (w_maj + w_min + 1e-9) if (w_maj + w_min) > 0.02 * prof.sum() else 0.0
            r_notes += 0.12 * (third if mode == "major" else -third)
            r = 0.6 * r_notes + 0.4 * r_chroma
            if best is None or r > best[0]:
                best = (r, tonic, mode, r_notes)
    r, tonic, mode, r_notes = best
    name_f = NOTE_NAMES_FLAT[tonic] + ("m" if mode == "minor" else "")
    flats = name_f in FLAT_KEYS
    out = dict(chroma_key)
    out.update({"tonic": tonic, "mode": mode, "use_flats": flats, "correlation": round(r, 3),
                "name": (NOTE_NAMES_FLAT if flats else NOTE_NAMES_SHARP)[tonic] + (" minor" if mode == "minor" else " major"),
                "sources": {"chroma": chroma_key.get("name"), "notes_correlation": round(r_notes, 3)},
                "confidence": round(float(np.clip(chroma_key.get("confidence", 0.5) * 0.5 + 0.5 * max(r, 0), 0, 1)), 3)})
    out.pop("_corr", None)
    return out


def analyze_harmony(y_harm: np.ndarray, sr: int, beats: np.ndarray, duration: float,
                    y_bass: np.ndarray | None = None) -> dict:
    C = np.abs(librosa.cqt(y=y_harm, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C2"), n_bins=180,
                           bins_per_octave=36))
    chroma = librosa.feature.chroma_cqt(C=C, sr=sr, hop_length=HOP, bins_per_octave=36, n_chroma=12, norm=None)
    n_frames = chroma.shape[1]
    tuning_cents = 0.0
    if y_bass is not None and np.sqrt(np.mean(y_bass**2)) > 1e-4:
        bass_chroma = librosa.feature.chroma_cqt(y=y_bass, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C1"),
                                                 n_octaves=3, norm=None)
    else:
        Cb = C[:36]  # C2..B2 region of the harmonic CQT as a weak bass proxy
        bass_chroma = librosa.feature.chroma_cqt(C=Cb, sr=sr, hop_length=HOP, bins_per_octave=36, n_chroma=12,
                                                 norm=None)
    bass_chroma = bass_chroma[:, :n_frames]

    beat_frames = librosa.time_to_frames(np.asarray(beats, float), sr=sr, hop_length=HOP)
    beat_frames = beat_frames[(beat_frames > 0) & (beat_frames < n_frames)]
    beat_frames = np.unique(beat_frames)
    seg_bounds = np.concatenate([[0], beat_frames, [n_frames]])
    seg_bounds = np.unique(seg_bounds)
    sync = librosa.util.sync(chroma, seg_bounds[1:-1], aggregate=np.median) if len(seg_bounds) > 2 else chroma.mean(
        axis=1, keepdims=True)
    bsync = librosa.util.sync(bass_chroma, seg_bounds[1:-1], aggregate=np.median) if len(seg_bounds) > 2 else \
        bass_chroma.mean(axis=1, keepdims=True)
    energy = sync.sum(axis=0)
    normed = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-9)

    # emission scores: cosine similarity with templates + "no chord" state
    sim = TEMPLATES @ normed  # (n_labels, n_segs)
    prior = np.array([QUALITY_PRIOR[q] for _, q in LABELS])[:, None]
    sim = sim * (0.85 + 0.15 * prior)
    silent = energy < (0.08 * np.median(energy[energy > 0]) if np.any(energy > 0) else 1)
    no_chord = np.where(silent, 1.0, 0.35)[None, :]
    emis = np.vstack([sim, no_chord])
    probs = np.exp(emis * 12.0)
    probs /= probs.sum(axis=0, keepdims=True)
    n_states = probs.shape[0]
    trans = librosa.sequence.transition_loop(n_states, 0.82)
    path = librosa.sequence.viterbi(probs, trans)

    seg_times = librosa.frames_to_time(seg_bounds, sr=sr, hop_length=HOP)
    seg_times[-1] = duration
    chords = []
    for i, state in enumerate(path):
        t0, t1 = float(seg_times[i]), float(seg_times[i + 1])
        if state == n_states - 1:
            label = {"label": "N", "root": None, "quality": None, "bass": None}
            conf = float(probs[state, i])
        else:
            root, q = LABELS[state]
            bass_pc = int(np.argmax(bsync[:, i])) if bsync[:, i].sum() > 0 else root
            chord_pcs = {(root + iv) % 12 for iv in QUALITIES[q]}
            bass = bass_pc if bass_pc in chord_pcs else root
            col = np.sort(sim[:, i])
            conf = float(np.clip((col[-1] - col[-2]) * 8 + 0.4 * col[-1], 0, 1))
            label = {"label": None, "root": root, "quality": q, "bass": bass}
        if chords and chords[-1]["_state"] == state and chords[-1]["bass"] == label["bass"]:
            c = chords[-1]
            c["end"] = t1
            c["_confs"].append(conf)
        else:
            chords.append({"start": t0, "end": t1, **label, "_state": int(state), "_confs": [conf]})

    chroma_mean = (chroma * (chroma.sum(axis=0, keepdims=True) > 0)).mean(axis=1)
    key = estimate_key(chroma_mean)
    out = []
    for c in chords:
        if c["end"] - c["start"] < 1e-3:
            continue
        conf = float(np.mean(c["_confs"]))
        if c["root"] is None:
            name = "N"
        else:
            name = chord_name(c["root"], c["quality"], c["bass"], key["use_flats"])
        out.append({
            "start": round(c["start"], 3), "end": round(c["end"], 3), "name": name, "root": c["root"],
            "quality": c["quality"], "bass": c["bass"], "confidence": round(conf, 3),
            "confidence_level": level(conf),
        })
    try:
        tuning_cents = float(librosa.estimate_tuning(y=y_harm, sr=sr) * 100)
    except Exception:  # noqa: BLE001
        tuning_cents = 0.0
    return {"key": key, "chords": out, "reference_deviation_cents": round(tuning_cents, 1)}


def level(conf: float) -> str:
    return "high" if conf >= 0.7 else "medium" if conf >= 0.45 else "low"


def chord_from_pitches(pitches: list[int]) -> tuple[int, str] | None:
    """Name a chord from a set of MIDI pitches (used to cross-check the audio chords)."""
    pcs = {p % 12 for p in pitches}
    if len(pcs) < 2:
        return None
    best = None
    bass_pc = min(pitches) % 12
    for (root, q), _ in zip(LABELS, range(len(LABELS))):
        tpl = {(root + iv) % 12 for iv in QUALITIES[q]}
        inter = len(tpl & pcs)
        extra = len(pcs - tpl)
        missing = len(tpl - pcs)
        score = inter - 0.8 * extra - 0.5 * missing + (0.3 if root == bass_pc else 0) + 0.1 * QUALITY_PRIOR[q]
        if best is None or score > best[0]:
            best = (score, root, q)
    return (best[1], best[2]) if best else None


def cross_validate_with_notes(chords: list[dict], notes: list[dict]) -> dict:
    """Compare audio chords with the pitch content of the transcription in each chord span."""
    agree = 0
    total = 0
    for c in chords:
        if c["root"] is None:
            continue
        pitches = [n["pitch"] for n in notes if n["start"] < c["end"] and n["end"] > c["start"]]
        if len(set(p % 12 for p in pitches)) < 2:
            continue
        total += 1
        pcs = {p % 12 for p in pitches}
        tpl = {(c["root"] + iv) % 12 for iv in QUALITIES[c["quality"]]}
        overlap = len(pcs & tpl) / max(1, len(tpl))
        c["notes_agreement"] = round(overlap, 2)
        if overlap >= 0.66:
            agree += 1
            c["confidence"] = round(min(1.0, c["confidence"] + 0.15), 3)
        else:
            c["confidence"] = round(max(0.0, c["confidence"] - 0.15), 3)
            c["flag"] = "transcription_disagrees"
        c["confidence_level"] = level(c["confidence"])
    return {"chords_checked": total, "agreement": round(agree / total, 3) if total else None}
