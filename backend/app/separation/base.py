"""Source-separation interface.

Separators are pluggable: anything that turns a stereo mix into named stems can be
registered in :mod:`app.separation` (e.g. a future guitar-specific model, MDX-Net,
BS-RoFormer, or a commercial API) without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

ProgressFn = Callable[[float, str], None]

# Canonical stem names understood by the rest of the pipeline.
STEM_LABELS = {
    "guitar": "Guitar",
    "bass": "Bass",
    "drums": "Drums",
    "vocals": "Vocals",
    "piano": "Piano / Keys",
    "other": "Other (synths, keys, FX)",
}


@dataclass
class SeparationResult:
    stems: dict[str, np.ndarray]  # name -> (channels, samples) float32
    sr: int
    engine: str
    model: str
    quality: str  # "neural" | "dsp-approximate"
    notes: list[str] = field(default_factory=list)


class Separator(Protocol):
    name: str

    def available(self) -> tuple[bool, str]:
        ...

    def separate(self, audio: np.ndarray, sr: int, progress: ProgressFn) -> SeparationResult:
        ...
