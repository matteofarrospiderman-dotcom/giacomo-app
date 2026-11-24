"""Download / verify model weights so the app can run fully offline afterwards.

* Basic Pitch: the ONNX network ships inside the ``basic-pitch`` wheel. It is copied to
  ``models/basic_pitch/nmp.onnx`` (also used if the package is later uninstalled).
* Demucs: fetches the ``htdemucs_6s`` (default) or ``--demucs`` model weights through
  torch.hub and copies them into ``models/demucs`` so they load without internet.
* CREPE: weights are bundled with ``torchcrepe``; this only checks they are present.

Usage:  python scripts/download_models.py [--demucs htdemucs_6s] [--skip-demucs]
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


def basic_pitch() -> None:
    dest = settings.models_dir / "basic_pitch" / "nmp.onnx"
    if dest.exists():
        print(f"[ok] Basic Pitch ONNX already at {dest}")
        return
    spec = importlib.util.find_spec("basic_pitch")
    if not spec or not spec.submodule_search_locations:
        print("[!!] basic-pitch is not installed. Run: pip install --no-deps basic-pitch==0.4.0")
        return
    src = Path(list(spec.submodule_search_locations)[0]) / "saved_models" / "icassp_2022" / "nmp.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    print(f"[ok] Basic Pitch ONNX copied to {dest}")


def demucs(name: str) -> None:
    try:
        import torch  # noqa: PLC0415
        from demucs.pretrained import get_model  # noqa: PLC0415
    except ImportError:
        print("[--] torch/demucs not installed: DSP separation fallback will be used (pip install -r requirements-ml.txt)")
        return
    print(f"[..] Fetching Demucs model '{name}' (first time ~80-320 MB)...")
    try:
        get_model(name)
    except Exception as e:  # noqa: BLE001
        print(f"[!!] Could not download Demucs weights: {e}")
        return
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    dest = settings.models_dir / "demucs"
    dest.mkdir(parents=True, exist_ok=True)
    for f in hub.glob("*.th"):
        if not (dest / f.name).exists():
            shutil.copy(f, dest / f.name)
    # copy the bag-of-models yaml so get_model(name, repo=dest) works offline
    import demucs  # noqa: PLC0415

    remote = Path(demucs.__file__).parent / "remote"
    for y in remote.glob("*.yaml"):
        if not (dest / y.name).exists():
            shutil.copy(y, dest / y.name)
    print(f"[ok] Demucs weights cached in {dest}")


def crepe() -> None:
    spec = importlib.util.find_spec("torchcrepe")
    if not spec:
        print("[--] torchcrepe not installed: pYIN will be used for monophonic pitch")
        return
    assets = Path(list(spec.submodule_search_locations)[0]) / "assets"
    print(f"[ok] CREPE weights present: {[p.name for p in assets.glob('*.pth')]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demucs", default=settings.demucs_model)
    ap.add_argument("--skip-demucs", action="store_true")
    a = ap.parse_args()
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    basic_pitch()
    if not a.skip_demucs:
        demucs(a.demucs)
    crepe()
