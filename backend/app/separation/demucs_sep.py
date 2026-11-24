"""Demucs (Hybrid Transformer Demucs, Meta AI) neural source separation.

Default model ``htdemucs_6s`` separates drums / bass / vocals / guitar / piano / other,
which is the only widely available open model with a dedicated *guitar* stem.
``htdemucs_ft`` (4 stems, fine-tuned) gives cleaner vocals/bass/drums but leaves guitar
inside "other".

Weights are fetched by torch.hub on first use (~80 MB for htdemucs_6s) into
``~/.cache/torch/hub/checkpoints``. For offline machines, put the ``.th`` files and the
model's ``.yaml`` into ``$TABFORGE_MODELS_DIR/demucs`` (``scripts/download_models.py``
does this) and they are loaded from there.

Limitations: the guitar stem of htdemucs_6s is noticeably less clean than its
vocals/drums stems (guitar/piano were trained on less data); bleed of keys and synths
into the guitar stem is common, and two guitars always end up in the same stem
(see :mod:`app.separation.guitar_split` for the multi-guitar strategy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import settings
from .base import ProgressFn, SeparationResult

_MODEL_CACHE: dict[str, object] = {}


class DemucsSeparator:
    name = "demucs"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.demucs_model

    def _local_repo(self) -> Path | None:
        repo = settings.models_dir / "demucs"
        if repo.exists() and any(repo.glob("*.th")):
            return repo
        return None

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401, PLC0415
            import demucs.pretrained  # noqa: F401, PLC0415
        except Exception as e:  # noqa: BLE001
            return False, f"demucs/torch not installed ({e.__class__.__name__})"
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(k in msg for k in ("urlopen", "Tunnel", "403", "Connection", "resolve", "timed out")):
                return False, ("Demucs weights could not be downloaded (no internet access to "
                               "dl.fbaipublicfiles.com). Run scripts/download_models.py on a connected machine "
                               "or copy the weights into models/demucs")
            return False, f"Demucs weights unavailable: {msg[:160]}"
        return True, "ok"

    def _load(self):
        if self.model_name in _MODEL_CACHE:
            return _MODEL_CACHE[self.model_name]
        from demucs.pretrained import get_model  # noqa: PLC0415

        model = get_model(self.model_name, repo=self._local_repo())
        model.eval()
        _MODEL_CACHE[self.model_name] = model
        return model

    def separate(self, audio: np.ndarray, sr: int, progress: ProgressFn) -> SeparationResult:
        import torch  # noqa: PLC0415
        from demucs.apply import apply_model  # noqa: PLC0415

        model = self._load()
        msr = int(model.samplerate)
        x = audio
        if sr != msr:
            import librosa  # noqa: PLC0415

            x = librosa.resample(audio, orig_sr=sr, target_sr=msr)
        if x.shape[0] == 1:
            x = np.vstack([x, x])
        device = settings.resolve_device()
        wav = torch.from_numpy(np.ascontiguousarray(x)).float()
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        wav = (wav - mean) / std

        total = {"n": 0}

        def cb(info: dict) -> None:
            # called per processed segment; map to 0..1 heuristically
            total["n"] += 1
            progress(min(0.95, total["n"] / max(1.0, info.get("models", 1) * 40.0)), "Separating sources (Demucs)")

        progress(0.02, f"Running {self.model_name} on {device}")
        with torch.no_grad():
            try:
                out = apply_model(model, wav[None], device=device, shifts=1, split=True, overlap=0.25,
                                  progress=False, callback=cb)
            except TypeError:  # older demucs without callback support
                out = apply_model(model, wav[None], device=device, shifts=1, split=True, overlap=0.25)
        out = out[0] * std + mean
        stems = {}
        for name, src in zip(model.sources, out):
            arr = src.cpu().numpy().astype(np.float32)
            if msr != sr:
                import librosa  # noqa: PLC0415

                arr = librosa.resample(arr, orig_sr=msr, target_sr=sr)
            stems[name] = arr[:, : audio.shape[-1]]
        progress(1.0, "Separation finished")
        notes = []
        if "guitar" not in stems:
            notes.append(f"Model {self.model_name} has no dedicated guitar stem; guitar is inside 'other'.")
        return SeparationResult(stems=stems, sr=sr, engine="demucs", model=self.model_name, quality="neural",
                                notes=notes)
