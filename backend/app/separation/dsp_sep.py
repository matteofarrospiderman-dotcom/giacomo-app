"""DSP fallback separator (no neural network required).

Used when Demucs/torch or its weights are unavailable. It is clearly labelled
"dsp-approximate" in the UI because it is far below neural separation quality.

Method
------
1. Median-filter harmonic/percussive separation (Fitzgerald 2010) on the STFT
   magnitude: percussive -> ``drums``.
2. The harmonic part is split by frequency with smooth crossovers: content dominated by
   energy below ~160 Hz -> ``bass``.
3. Vocal estimate: REPET-SIM (Rafii & Pardo 2012) finds the repeating accompaniment by
   nearest-neighbour filtering of similar frames; the non-repeating, centre-panned
   residual in the 180 Hz-7 kHz band is assigned to ``vocals``.
4. Everything harmonic that remains is ``guitar`` (really "guitar + keys + synths").

All masks are soft (Wiener-like) and applied to each stereo channel so panning is kept,
which the multi-guitar splitter relies on.
"""

from __future__ import annotations

import numpy as np
import librosa

from .base import ProgressFn, SeparationResult

N_FFT = 4096
HOP = 1024


class DSPSeparator:
    name = "dsp"

    def available(self) -> tuple[bool, str]:
        return True, "ok"

    def separate(self, audio: np.ndarray, sr: int, progress: ProgressFn) -> SeparationResult:
        if audio.shape[0] == 1:
            audio = np.vstack([audio, audio])
        progress(0.05, "STFT")
        X = np.stack([librosa.stft(ch, n_fft=N_FFT, hop_length=HOP) for ch in audio])  # (2, F, T)
        mag_mid = np.abs(X).mean(axis=0)
        progress(0.2, "Harmonic/percussive separation")
        H, P = librosa.decompose.hpss(mag_mid, kernel_size=(17, 31), power=2.0, margin=1.0, mask=False)
        eps = 1e-10
        Hp, Pp = H**2, P**2
        m_harm = Hp / (Hp + Pp + eps)
        m_perc = 1.0 - m_harm

        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
        lp = 1.0 / (1.0 + (freqs / 160.0) ** 6)  # smooth 160 Hz low-pass
        lp = lp[:, None]

        progress(0.45, "Estimating vocals (REPET-SIM)")
        harm_mag = mag_mid * m_harm
        width = max(3, int(librosa.time_to_frames(1.5, sr=sr, hop_length=HOP)))
        try:
            rep = librosa.decompose.nn_filter(harm_mag, aggregate=np.median, metric="cosine", width=width)
            rep = np.minimum(harm_mag, rep)
        except Exception:  # noqa: BLE001 - very short clips
            rep = harm_mag
        nonrep = np.maximum(harm_mag - rep, 0)
        m_voc_raw = (nonrep**2) / (nonrep**2 + rep**2 + eps)
        # vocals are usually centre panned: weight by inter-channel similarity
        L, R = np.abs(X[0]), np.abs(X[1])
        centre = 1.0 - np.abs(L - R) / (L + R + eps)
        band = ((freqs > 180) & (freqs < 7000)).astype(float)[:, None]
        m_voc = m_voc_raw * centre**2 * band * 0.9

        m_bass = m_harm * lp
        m_guitar = np.clip(m_harm * (1 - lp) - m_voc, 0, 1)
        m_voc = np.clip(m_voc * (1 - lp), 0, 1)

        progress(0.75, "Reconstructing stems")
        stems = {}
        for name, mask in (("drums", m_perc), ("bass", m_bass), ("vocals", m_voc), ("guitar", m_guitar)):
            chans = [librosa.istft(X[c] * mask, hop_length=HOP, length=audio.shape[-1]) for c in range(2)]
            stems[name] = np.stack(chans).astype(np.float32)
        progress(1.0, "DSP separation finished")
        return SeparationResult(
            stems=stems, sr=sr, engine="dsp", model="hpss+repet-sim", quality="dsp-approximate",
            notes=[
                "Neural separation (Demucs) was not available, so a DSP approximation was used.",
                "The 'guitar' stem also contains keyboards/synths; expect more false notes.",
            ],
        )
