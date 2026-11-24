"""Runtime configuration.

Everything is configurable through environment variables prefixed with ``TABFORGE_``
so the same code runs on a laptop CPU or a GPU server without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"TABFORGE_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BACKEND_ROOT / "data"))))
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", str(BACKEND_ROOT / "models"))))

    # Upload validation
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "250"))
    max_duration_s: float = float(_env("MAX_DURATION_S", "900"))  # 15 minutes
    min_duration_s: float = float(_env("MIN_DURATION_S", "1.0"))
    allowed_extensions: tuple[str, ...] = (
        ".mp3", ".wav", ".flac", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aif", ".aiff", ".wma", ".webm", ".mp4",
    )

    # Processing
    analysis_sr: int = 44100
    workers: int = int(_env("WORKERS", "1"))
    # "auto" -> demucs if importable and weights available, else DSP fallback
    separator: str = _env("SEPARATOR", "auto")
    demucs_model: str = _env("DEMUCS_MODEL", "htdemucs_6s")
    device: str = _env("DEVICE", "auto")  # auto | cpu | cuda | mps
    # monophonic f0 tracker: auto (crepe if torch available) | crepe | pyin | none
    f0_tracker: str = _env("F0_TRACKER", "auto")
    crepe_model: str = _env("CREPE_MODEL", "tiny")  # tiny | full

    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()
    )

    @property
    def projects_dir(self) -> Path:
        return self.data_dir / "projects"

    @property
    def learning_dir(self) -> Path:
        return self.data_dir / "learning"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # torch missing or broken -> CPU
            pass
        return "cpu"


settings = Settings()


def ensure_dirs() -> None:
    settings.projects_dir.mkdir(parents=True, exist_ok=True)
    settings.learning_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
