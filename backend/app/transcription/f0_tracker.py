"""Monophonic fundamental-frequency tracking (CREPE or pYIN).

Used for:
  * cross-validating polyphonic notes in monophonic passages (lead lines, solos);
  * fine pitch contours for bends, vibrato and slides;
  * catching notes the polyphonic model missed in single-note lines.

CREPE (Kim et al. 2018, via ``torchcrepe``) is a CNN trained on pitch-annotated audio
and is the most accurate open monophonic tracker; its weights ship inside the pip
package so no download is needed. pYIN (Mauch & Dixon 2014, librosa) is the
dependency-free fallback. Both report a per-frame voicing probability.

Limitation: in polyphonic passages (chords) an f0 tracker locks onto one note or
nothing; the ensemble only trusts it where the polyphony estimate is ~1.
"""

from __future__ import annotations

import numpy as np
import librosa

from ..config import settings

SR = 22050
HOP = 256  # aligned with Basic Pitch frames


def _crepe(y: np.ndarray, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    import torch  # noqa: PLC0415
    import torchcrepe  # noqa: PLC0415

    sr = 16000
    y16 = librosa.resample(y, orig_sr=SR, target_sr=sr)
    hop16 = int(round(HOP * sr / SR))  # 186 samples ~ 11.6 ms
    audio = torch.from_numpy(y16.astype(np.float32))[None]
    device = settings.resolve_device()
    if device == "mps":
        device = "cpu"
    with torch.no_grad():
        f0, periodicity = torchcrepe.predict(audio, sr, hop16, fmin, fmax, model=settings.crepe_model,
                                             return_periodicity=True, batch_size=1024, device=device,
                                             decoder=torchcrepe.decode.viterbi)
    periodicity = torchcrepe.filter.median(periodicity, 3)
    return f0[0].cpu().numpy(), periodicity[0].cpu().numpy()


def _pyin(y: np.ndarray, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=2048, hop_length=HOP,
                                    fill_na=np.nan)
    return np.nan_to_num(f0, nan=0.0), np.where(voiced, prob, prob * 0.5)


def track(y22: np.ndarray, fmin: float = 60.0, fmax: float = 1500.0) -> dict | None:
    """Return {'t','midi','voicing','engine'} on the 256-sample grid, or None if disabled."""
    pref = settings.f0_tracker
    if pref == "none":
        return None
    engine = "pyin"
    f0 = conf = None
    n = int(np.ceil(len(y22) / HOP)) + 1
    if pref in ("auto", "crepe"):
        try:
            f0, conf = _crepe(y22, fmin, fmax)
            engine = f"crepe-{settings.crepe_model}"
            # CREPE runs at 16 kHz with a 186-sample hop (11.625 ms), not exactly 256/22050 s:
            # resample onto the shared frame grid or the track drifts ~0.2 s over 3 minutes
            hop16 = int(round(HOP * 16000 / SR))
            src_t = np.arange(len(f0)) * hop16 / 16000
            dst_t = np.arange(n) * HOP / SR
            voiced = f0 > 0
            f0 = np.interp(dst_t, src_t, np.where(voiced, f0, 0.0), right=0.0)
            conf = np.interp(dst_t, src_t, conf, right=0.0)
            # do not invent pitches by interpolating between voiced and unvoiced frames
            mask = np.interp(dst_t, src_t, voiced.astype(float), right=0.0) > 0.99
            f0 = np.where(mask, f0, 0.0)
        except Exception:  # noqa: BLE001 - torch/torchcrepe missing -> pyin
            f0 = None
    if f0 is None:
        # pYIN is O(n * thresholds); cap at ~6 minutes of audio for responsiveness
        f0, conf = _pyin(y22[: SR * 360], fmin, fmax)
    f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
    conf = np.pad(conf, (0, max(0, n - len(conf))))[:n]
    midi = np.where(f0 > 0, librosa.hz_to_midi(np.maximum(f0, 1e-3)), np.nan)
    return {"t0": 0.0, "hop": HOP / SR, "midi": midi.astype(np.float32), "voicing": conf.astype(np.float32),
            "engine": engine}


def segment_notes(track_: dict, min_dur: float = 0.06, voicing_thr: float = 0.6) -> list[dict]:
    """Split a voiced f0 track into note segments at pitch jumps > 0.7 semitones."""
    midi = track_["midi"]
    v = track_["voicing"]
    hop = track_["hop"]
    out = []
    i = 0
    n = len(midi)
    while i < n:
        if not (v[i] > voicing_thr and np.isfinite(midi[i])):
            i += 1
            continue
        j = i
        ref = midi[i]
        vals = [midi[i]]
        while j + 1 < n and v[j + 1] > voicing_thr and np.isfinite(midi[j + 1]) and abs(midi[j + 1] - np.median(vals[-8:])) < 0.7:
            j += 1
            vals.append(midi[j])
        dur = (j + 1 - i) * hop
        if dur >= min_dur:
            out.append({"start": i * hop, "end": (j + 1) * hop, "pitch": int(round(float(np.median(vals)))),
                        "voicing": float(np.mean(v[i: j + 1]))})
        del ref
        i = j + 1
    return out
