"""Assemble TabForge Pocket into a deployable folder.

    python pocket/build.py <basic_pitch_model_dir> <demo_dir> [out_dir]

* inlines style.css, engine.js and ui.js into index.html (TF.js loads from jsDelivr);
* packs Basic Pitch's TF.js graph model (model.json + weights) into a single model.json
  with base64 weights, so the page only needs same-origin fetches;
* copies demo.mp3 and demo.json (the demo transcription produced by the same engine in
  Node: see tools/node_transcribe.js).

<basic_pitch_model_dir> is the `model/` folder of the npm package @spotify/basic-pitch
(`npm pack @spotify/basic-pitch@1.0.1` and extract it).
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    model_dir = Path(sys.argv[1])
    demo_dir = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "dist"
    out.mkdir(parents=True, exist_ok=True)
    src = HERE / "src"
    html = (src / "index.html").read_text()
    for marker, name in (("/*__STYLE__*/", "style.css"), ("/*__ENGINE__*/", "engine.js"), ("/*__UI__*/", "ui.js")):
        code = (src / name).read_text()
        if "</script" in code.lower() and name.endswith(".js"):
            raise SystemExit(f"{name} contains a closing script tag")
        html = html.replace(marker, code)
    (out / "index.html").write_text(html)

    mj = json.loads((model_dir / "model.json").read_text())
    weights = b"".join((model_dir / p).read_bytes() for g in mj["weightsManifest"] for p in g["paths"])
    packed = {
        "format": mj["format"],
        "signature": mj.get("signature"),
        "modelTopology": mj["modelTopology"],
        "weightSpecs": [w for g in mj["weightsManifest"] for w in g["weights"]],
        "weightsBase64": base64.b64encode(weights).decode(),
        "source": "Spotify Basic Pitch (ICASSP 2022), Apache-2.0, https://github.com/spotify/basic-pitch",
    }
    (out / "model.json").write_text(json.dumps(packed, separators=(",", ":")))
    for name in ("demo.mp3", "demo.json"):
        shutil.copy(demo_dir / name, out / name)
    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB" for p in out.iterdir()}
    print(f"built {out}: {sizes}")


if __name__ == "__main__":
    main()
