"""Spotify Basic Pitch (ICASSP 2022) polyphonic transcription via ONNX Runtime.

Why ONNX: the official ``basic-pitch`` package pins TensorFlow < 2.15, which has no
wheels for modern Python. The package ships an ONNX export of the same network
(``nmp.onnx``), so we run that with onnxruntime (CPU or CUDA) and implement windowing
and note decoding ourselves.

The network outputs three posteriorgrams at 22050/256 = 86.13 frames/s:
  * ``note``    (T, 88)  - a pitch is sounding
  * ``onset``   (T, 88)  - a pitch starts
  * ``contour`` (T, 264) - multi-f0 salience at 1/3 semitone resolution

Decoding (our implementation):
  1. Onset peaks above ``onset_threshold`` seed notes; each note is followed forward
     while the note posterior stays above ``frame_threshold`` (short dips tolerated).
  2. A second "legato" pass creates notes from remaining strong note activations
     that have *no* onset peak - this is what catches hammer-ons, pull-offs and
     slides on guitar, whose attacks are too soft for the onset head.
  3. A fine pitch contour per note is tracked on the ``contour`` output, giving the
     bend / vibrato / slide information used by the technique detector.

Limitations: trained on mixed instruments (GuitarSet is only part of the data);
heavy distortion produces ghost notes on harmonics; very fast tremolo picking can
merge; pitch range limited to A0-C8.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from ..config import settings
from .base import NoteEvent

SR = 22050
FFT_HOP = 256
AUDIO_N_SAMPLES = SR * 2 - FFT_HOP  # 43844
N_OVERLAP_FRAMES = 30
OVERLAP_LEN = N_OVERLAP_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN
FPS = SR / FFT_HOP
MIDI_OFFSET = 21  # bin 0 == A0
CONTOUR_BPS = 3

_SESSION = None


def model_path() -> Path | None:
    local = settings.models_dir / "basic_pitch" / "nmp.onnx"
    if local.exists():
        return local
    spec = importlib.util.find_spec("basic_pitch")
    if spec and spec.submodule_search_locations:
        p = Path(list(spec.submodule_search_locations)[0]) / "saved_models" / "icassp_2022" / "nmp.onnx"
        if p.exists():
            return p
    return None


def available() -> tuple[bool, str]:
    try:
        import onnxruntime  # noqa: F401, PLC0415
    except ImportError:
        return False, "onnxruntime not installed"
    if model_path() is None:
        return False, "Basic Pitch ONNX model not found (pip install basic-pitch --no-deps, or run scripts/download_models.py)"
    return True, "ok"


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort  # noqa: PLC0415

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        _SESSION = ort.InferenceSession(str(model_path()), providers=providers)
    return _SESSION


def run_model(y: np.ndarray, progress=None) -> dict[str, np.ndarray]:
    """Run the network on mono 22.05 kHz audio. Returns posteriorgrams on a uniform frame grid."""
    sess = _session()
    n_orig = len(y)
    padded = np.concatenate([np.zeros(OVERLAP_LEN // 2, np.float32), y.astype(np.float32)])
    starts = list(range(0, len(padded), HOP_SIZE))
    n_frames = int(np.ceil(n_orig / FFT_HOP)) + 1
    out = {"note": np.zeros((n_frames, 88), np.float32), "onset": np.zeros((n_frames, 88), np.float32),
           "contour": np.zeros((n_frames, 264), np.float32)}
    half = N_OVERLAP_FRAMES // 2
    batch = 8
    names = ["StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:0"]
    for b0 in range(0, len(starts), batch):
        chunk_starts = starts[b0: b0 + batch]
        x = np.zeros((len(chunk_starts), AUDIO_N_SAMPLES, 1), np.float32)
        for i, s in enumerate(chunk_starts):
            w = padded[s: s + AUDIO_N_SAMPLES]
            x[i, : len(w), 0] = w
        note, onset, contour = sess.run(names, {"serving_default_input_2:0": x})
        for i, s in enumerate(chunk_starts):
            kept = slice(half, note.shape[1] - half)
            n_kept = note.shape[1] - 2 * half
            # frame k of the kept region is at original-audio sample s + k*FFT_HOP
            idx = np.round((s + np.arange(n_kept) * FFT_HOP) / FFT_HOP).astype(int)
            ok = idx < n_frames
            for key, arr in (("note", note), ("onset", onset), ("contour", contour)):
                seg = arr[i, kept][ok]
                out[key][idx[ok]] = np.maximum(out[key][idx[ok]], seg)
        if progress:
            progress(min(1.0, (b0 + batch) / len(starts)), "Basic Pitch inference")
    return out


def _track_contour(contour: np.ndarray, pitch: int, f0: int, f1: int) -> np.ndarray:
    """Follow the salience peak near ``pitch`` frame by frame; returns semitone offsets."""
    center = (pitch - MIDI_OFFSET) * CONTOUR_BPS + 1
    lo_lim, hi_lim = center - 4, center + 8  # allow ~2.7 st bends up, ~1.3 st down
    cur = center
    out = np.zeros(f1 - f0, np.float32)
    for k, f in enumerate(range(f0, f1)):
        a = max(lo_lim, cur - 2, 0)
        b = min(hi_lim, cur + 2, contour.shape[1] - 1)
        seg = contour[f, a: b + 1]
        if seg.size == 0 or seg.max() < 0.05:
            out[k] = (cur - center) / CONTOUR_BPS
            continue
        j = int(np.argmax(seg))
        peak = a + j
        # parabolic interpolation for sub-bin precision
        if 0 < j < len(seg) - 1:
            y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
            d = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / d if d != 0 else 0.0
        else:
            frac = 0.0
        cur = peak
        out[k] = (peak + frac - center) / CONTOUR_BPS
    # median smooth to remove single-frame jumps
    if len(out) >= 5:
        from scipy.signal import medfilt  # noqa: PLC0415

        out = medfilt(out, 5).astype(np.float32)
    return out


def decode_notes(post: dict[str, np.ndarray], onset_threshold: float = 0.5, frame_threshold: float = 0.3,
                 min_note_s: float = 0.055, energy_tol: int = 8, min_midi: int = 23, max_midi: int = 100,
                 legato_pass: bool = True) -> list[NoteEvent]:
    note = post["note"].copy()
    onset = post["onset"]
    contour = post["contour"]
    T = note.shape[0]
    min_len = max(2, int(round(min_note_s * FPS)))

    # infer extra onsets from sharp rises in the note posterior (as basic-pitch does)
    diff = np.zeros_like(note)
    diff[1:] = np.maximum(note[1:] - note[:-1], 0)
    diff[2:] = np.maximum(diff[2:], note[2:] - note[:-2])
    inferred = np.maximum(onset, diff * (onset.max() / (diff.max() + 1e-9)))

    peaks = (inferred == maximum_filter1d(inferred, size=3, axis=0)) & (inferred > onset_threshold)
    ts, fs = np.nonzero(peaks)
    order = np.argsort(-ts)  # process from the end so earlier notes don't swallow later ones
    notes: list[NoteEvent] = []
    remaining = note.copy()
    for t, f in zip(ts[order], fs[order]):
        midi = f + MIDI_OFFSET
        if midi < min_midi or midi > max_midi:
            continue
        if t >= T - 1:
            continue
        i = t + 1
        k = 0
        while i < T - 1 and k < energy_tol:
            if remaining[i, f] < frame_threshold:
                k += 1
            else:
                k = 0
            i += 1
        i -= k
        if i - t < min_len:
            continue
        remaining[t:i, f] = 0
        if f > 0:
            remaining[t:i, f - 1] = 0
        if f < 87:
            remaining[t:i, f + 1] = 0
        amp = float(np.mean(note[t:i, f]))
        notes.append(NoteEvent(start=t / FPS, end=i / FPS, pitch=int(midi), amplitude=amp,
                               onset_strength=float(min(1.0, inferred[t, f])),
                               sources={"basic_pitch": amp}))

    if legato_pass:
        # notes with strong sustained activation but no onset peak (legato articulations)
        thr = max(frame_threshold + 0.15, 0.45)
        while True:
            mx = remaining.max()
            if mx < thr:
                break
            t, f = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
            midi = f + MIDI_OFFSET
            i = t
            while i < T - 1 and remaining[i + 1, f] > frame_threshold:
                i += 1
            j = t
            while j > 0 and remaining[j - 1, f] > frame_threshold:
                j -= 1
            remaining[j: i + 1, max(0, f - 1): f + 2] = 0
            if i + 1 - j < min_len or not (min_midi <= midi <= max_midi):
                continue
            amp = float(np.mean(note[j: i + 1, f]))
            notes.append(NoteEvent(start=j / FPS, end=(i + 1) / FPS, pitch=int(midi), amplitude=amp,
                                   onset_strength=float(min(1.0, inferred[j, f])),
                                   sources={"basic_pitch": amp}, flags={"no_onset"}))

    for n in notes:
        f0, f1 = int(round(n.start * FPS)), int(round(n.end * FPS))
        n.contour = _track_contour(contour, n.pitch, f0, max(f1, f0 + 1))
        n.contour_t0 = f0 / FPS
        n.contour_hop = 1 / FPS
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def transcribe(y22: np.ndarray, progress=None, **kw) -> tuple[list[NoteEvent], dict[str, np.ndarray]]:
    post = run_model(y22, progress)
    return decode_notes(post, **kw), post
