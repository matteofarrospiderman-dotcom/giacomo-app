"""Audio decoding, validation and file helpers.

FFmpeg is the primary decoder because it handles every container/codec a user is
likely to upload (MP3, AAC/M4A, FLAC, OGG/Opus, WMA, AIFF, video soundtracks...).
``soundfile`` (libsndfile) is a fallback for WAV/FLAC/OGG when FFmpeg is absent.

All analysis runs on float32 arrays shaped ``(channels, samples)``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioError(Exception):
    """Raised for unreadable, corrupt, too long or silent input."""


@dataclass
class ProbeInfo:
    duration: float
    sample_rate: int
    channels: int
    codec: str
    format: str
    bit_rate: int | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path: Path) -> ProbeInfo:
    if has_ffmpeg():
        cmd = [
            "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
            "-select_streams", "a:0", str(path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        except subprocess.TimeoutExpired as e:
            raise AudioError("Timed out while probing the file") from e
        if res.returncode != 0:
            raise AudioError(f"File is not readable audio: {res.stderr.strip()[:300] or 'unknown error'}")
        info = json.loads(res.stdout or "{}")
        streams = info.get("streams") or []
        if not streams:
            raise AudioError("No audio stream found in file")
        s = streams[0]
        fmt = info.get("format", {})
        duration = float(s.get("duration") or fmt.get("duration") or 0.0)
        return ProbeInfo(
            duration=duration,
            sample_rate=int(s.get("sample_rate") or 0),
            channels=int(s.get("channels") or 0),
            codec=str(s.get("codec_name", "unknown")),
            format=str(fmt.get("format_name", "unknown")),
            bit_rate=int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        )
    try:
        i = sf.info(str(path))
    except Exception as e:  # noqa: BLE001
        raise AudioError(f"Unsupported audio file (install FFmpeg for MP3/AAC support): {e}") from e
    return ProbeInfo(i.duration, i.samplerate, i.channels, i.subtype, i.format, None)


def decode(path: Path, sr: int = 44100, channels: int = 2) -> np.ndarray:
    """Decode any audio file to float32 ``(channels, n)`` at ``sr``."""
    if has_ffmpeg():
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-f", "f32le",
            "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "pipe:1",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        except subprocess.TimeoutExpired as e:
            raise AudioError("Timed out while decoding") from e
        if res.returncode != 0 or not res.stdout:
            raise AudioError(f"Decoding failed: {res.stderr.decode(errors='ignore').strip()[:300]}")
        audio = np.frombuffer(res.stdout, dtype=np.float32).reshape(-1, channels).T.copy()
    else:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.T
        if audio.shape[0] == 1 and channels == 2:
            audio = np.vstack([audio, audio])
        elif audio.shape[0] > channels:
            audio = audio[:channels]
        if file_sr != sr:
            import librosa  # noqa: PLC0415

            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
    if not np.all(np.isfinite(audio)):
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio.astype(np.float32, copy=False)


def validate_signal(audio: np.ndarray, sr: int, min_s: float, max_s: float) -> dict:
    dur = audio.shape[-1] / sr
    if dur < min_s:
        raise AudioError(f"Audio is too short ({dur:.2f}s); at least {min_s:.0f}s is required")
    if dur > max_s + 1:
        raise AudioError(f"Audio is too long ({dur / 60:.1f} min); the limit is {max_s / 60:.0f} min")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0
    if peak < 1e-4:
        raise AudioError("Audio appears to be silent")
    clip_ratio = float(np.mean(np.abs(audio) > 0.999))
    return {
        "duration": dur,
        "peak_dbfs": 20 * np.log10(peak + 1e-12),
        "rms_dbfs": 20 * np.log10(rms + 1e-12),
        "clipping_ratio": clip_ratio,
        "is_mono_compatible": bool(audio.shape[0] == 1 or np.allclose(audio[0], audio[-1], atol=1e-3)),
    }


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 1.0:  # avoid wrap-around clipping in int16
        data = data / peak
    sf.write(str(path), data, sr, subtype="PCM_16")


def encode_preview(wav_path: Path, out_path: Path, bitrate: str = "192k") -> Path:
    """Create a compressed copy for smooth browser playback. Falls back to the WAV."""
    if not has_ffmpeg():
        return wav_path
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate,
           str(out_path)]
    res = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    if res.returncode != 0 or not out_path.exists():
        return wav_path
    return out_path


def compute_peaks(audio: np.ndarray, sr: int, points_per_second: int = 100) -> dict:
    """Min/max envelope for waveform drawing (channel-summed)."""
    mono = to_mono(audio)
    hop = max(1, sr // points_per_second)
    n = len(mono) // hop
    if n == 0:
        return {"sr": points_per_second, "min": [], "max": []}
    frames = mono[: n * hop].reshape(n, hop)
    mn = np.round(frames.min(axis=1), 4)
    mx = np.round(frames.max(axis=1), 4)
    return {"sr": points_per_second, "min": mn.tolist(), "max": mx.tolist()}


def rms_envelope(y: np.ndarray, sr: int, hop: int = 512, frame: int = 2048) -> np.ndarray:
    import librosa  # noqa: PLC0415

    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
