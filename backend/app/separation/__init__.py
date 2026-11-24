"""Separator registry. ``get_separator('auto')`` prefers neural separation."""

from __future__ import annotations

from .base import SeparationResult, Separator, STEM_LABELS  # noqa: F401
from .demucs_sep import DemucsSeparator
from .dsp_sep import DSPSeparator


def get_separator(preference: str = "auto") -> tuple[Separator, list[str]]:
    """Return a usable separator and a list of human-readable notes about the choice."""
    notes: list[str] = []
    if preference in ("auto", "demucs"):
        dm = DemucsSeparator()
        ok, why = dm.available()
        if ok:
            return dm, notes
        notes.append(f"Demucs unavailable: {why}. Falling back to DSP separation.")
        if preference == "demucs":
            notes.append("Install torch + demucs and run scripts/download_models.py for neural separation.")
    return DSPSeparator(), notes
