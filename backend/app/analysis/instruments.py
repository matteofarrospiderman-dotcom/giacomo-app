"""Instrument identification on separated stems.

Neural separation already labels sources (drums, bass, vocals, guitar, piano, other).
On top of that we characterise each stem with interpretable audio descriptors:

* **presence**: share of the mix energy and fraction of time the stem is active;
* **electric vs acoustic guitar**: acoustic guitars have broadband pick/strum transients
  reaching 8-12 kHz, a body resonance region around 100-250 Hz, a high crest factor
  and no amp-cabinet roll-off; electric guitars through a cabinet roll off sharply above
  ~5-6 kHz, and driven amps lower the crest factor;
* **"other" stem content**: sustained flat envelopes and low spectral flux suggest synth
  pads, exponentially decaying harmonic tones suggest keys, noisy/flat spectra suggest FX.

These are heuristics with honest confidence values; the user can override the label.
"""

from __future__ import annotations

import numpy as np
import librosa

STEM_INSTRUMENTS = {
    "guitar": "Guitar",
    "bass": "Bass guitar",
    "drums": "Drums & percussion",
    "vocals": "Vocals",
    "piano": "Piano / keys",
    "other": "Other (synths, keys, extra guitars, FX)",
}


def describe(stem: np.ndarray, mix: np.ndarray, sr: int) -> dict:
    mono = stem.mean(axis=0) if stem.ndim == 2 else stem
    mix_m = mix.mean(axis=0) if mix.ndim == 2 else mix
    e_stem = float(np.sum(mono.astype(np.float64) ** 2))
    e_mix = float(np.sum(mix_m.astype(np.float64) ** 2)) + 1e-12
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=1024)[0]
    rms_db = 20 * np.log10(rms + 1e-9)
    ref = 20 * np.log10(np.max(librosa.feature.rms(y=mix_m, frame_length=2048, hop_length=1024)[0]) + 1e-9)
    active = float(np.mean(rms_db > ref - 35))
    peak = float(np.max(np.abs(mono))) + 1e-9
    crest_db = 20 * np.log10(peak / (np.sqrt(np.mean(mono**2)) + 1e-9))
    S = np.abs(librosa.stft(mono, n_fft=4096, hop_length=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    ltas = S.mean(axis=1) + 1e-12
    ltas_db = 20 * np.log10(ltas)
    peak_db = ltas_db[(freqs > 80) & (freqs < 5000)].max() if np.any((freqs > 80) & (freqs < 5000)) else ltas_db.max()
    above = np.where((ltas_db > peak_db - 30) & (freqs > 1000))[0]
    rolloff_hz = float(freqs[above.max()]) if above.size else 1000.0
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=S)))
    flux = np.mean(np.maximum(np.diff(np.log1p(S), axis=1), 0)) if S.shape[1] > 1 else 0.0
    return {
        "energy_share": round(e_stem / e_mix, 4),
        "active_ratio": round(active, 3),
        "present": bool(e_stem / e_mix > 0.004 and active > 0.08),
        "crest_db": round(float(crest_db), 2),
        "rolloff_hz": round(rolloff_hz, 0),
        "flatness": round(flatness, 4),
        "flux": round(float(flux), 4),
    }


def classify_guitar(desc: dict, hf_transients: float | None = None) -> dict:
    """Electric vs acoustic from cabinet roll-off and crest factor."""
    ac = 0.0
    ac += np.clip((desc["rolloff_hz"] - 6500) / 4000, -1, 1) * 0.5
    ac += np.clip((desc["crest_db"] - 14) / 8, -1, 1) * 0.35
    if hf_transients is not None:
        ac += np.clip(hf_transients - 0.5, -0.5, 0.5) * 0.3
    p_acoustic = float(1 / (1 + np.exp(-3 * ac)))
    kind = "acoustic" if p_acoustic > 0.5 else "electric"
    conf = abs(p_acoustic - 0.5) * 2
    return {"type": kind, "p_acoustic": round(p_acoustic, 3), "confidence": round(float(conf), 3),
            "evidence": [f"High-frequency roll-off ≈ {desc['rolloff_hz']:.0f} Hz "
                         f"({'cabinet-like' if desc['rolloff_hz'] < 7000 else 'full-range'})",
                         f"Crest factor {desc['crest_db']:.1f} dB ({'dynamic' if desc['crest_db'] > 15 else 'compressed/driven'})"]}


def classify_other(desc: dict) -> dict:
    guesses = []
    if desc["flux"] < 0.02 and desc["flatness"] < 0.05:
        guesses.append(("Synth pad / sustained keys", 0.5))
    if desc["flux"] >= 0.02 and desc["flatness"] < 0.05:
        guesses.append(("Keys / plucked instruments / extra guitar", 0.4))
    if desc["flatness"] >= 0.05:
        guesses.append(("Noise-like synths, FX or ambience", 0.35))
    if not guesses:
        guesses.append(("Mixed instruments", 0.2))
    return {"guesses": [{"label": g, "confidence": c} for g, c in guesses]}


def identify(stems: dict[str, np.ndarray], mix: np.ndarray, sr: int) -> dict:
    out = {}
    for name, audio in stems.items():
        d = describe(audio, mix, sr)
        entry = {"name": name, "label": STEM_INSTRUMENTS.get(name, name.title()), **d}
        base = name.split("_")[0]
        if base == "guitar":
            entry["guitar_type"] = classify_guitar(d)
        if base == "other":
            entry["content"] = classify_other(d)
        out[name] = entry
    return out
