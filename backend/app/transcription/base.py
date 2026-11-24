"""Shared note representation for all transcribers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: int  # MIDI
    amplitude: float = 0.5  # model posterior / salience, 0..1
    confidence: float = 0.5
    sources: dict[str, float] = field(default_factory=dict)
    # fine pitch contour relative to ``pitch`` (semitones), sampled at ``contour_hop`` seconds
    contour: np.ndarray | None = None
    contour_t0: float = 0.0
    contour_hop: float = 256 / 22050
    onset_strength: float = 0.5  # how pronounced the attack is (0..1)
    flags: set[str] = field(default_factory=set)
    techniques: list[dict] = field(default_factory=list)
    velocity: int = 90

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlaps(self, other: "NoteEvent", tol: float = 0.0) -> float:
        return max(0.0, min(self.end, other.end) - max(self.start, other.start) + tol)


def confidence_level(c: float) -> str:
    return "high" if c >= 0.7 else "medium" if c >= 0.45 else "low"
