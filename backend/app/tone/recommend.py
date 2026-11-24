"""Turn a tone analysis into a reproducible signal chain for the user's own gear.

The output is a list of chain blocks (in signal order) with concrete settings in knob
positions (0-10) and physical units (ms, s, %), each tagged with the confidence tier of
the analysis it came from. If the user tells us what they own, blocks are mapped onto
that gear (e.g. "use your Tube Screamer as a boost into the amp's crunch channel"); for
modelers we suggest model *families*, not claims about the original recording.
"""

from __future__ import annotations

AMPS = {
    "generic": {"label": "Any amp (generic settings)", "kind": "amp", "channels": ["clean", "crunch", "lead"]},
    "fender_clean": {"label": "Fender-style (Twin, Deluxe, Blues Jr.)", "kind": "amp", "channels": ["clean"],
                     "max_gain": 4.5, "note": "Use a pedal for anything beyond light breakup."},
    "marshall": {"label": "Marshall-style (Plexi, JCM800, DSL)", "kind": "amp", "channels": ["clean", "crunch", "lead"]},
    "vox": {"label": "Vox AC15/AC30-style", "kind": "amp", "channels": ["clean", "crunch"], "max_gain": 6.5},
    "mesa": {"label": "Mesa/Boogie / high-gain (Rectifier, 5150-style)", "kind": "amp", "channels": ["clean", "crunch", "lead"]},
    "boss_katana": {"label": "Boss Katana", "kind": "modeler",
                    "models": {"clean": "Clean", "crunch": "Crunch", "lead": "Lead", "high": "Brown", "acoustic": "Acoustic"}},
    "line6_helix": {"label": "Line 6 Helix / HX / POD Go", "kind": "modeler",
                    "models": {"clean": "US Deluxe Nrm", "crunch": "Essex A30 or Brit Plexi Brt", "lead": "Brit 2204",
                               "high": "Cali Rectifire or PV Panama", "acoustic": "a mic/acoustic sim"}},
    "neural_dsp": {"label": "Neural DSP / amp-sim plugins", "kind": "modeler",
                   "models": {"clean": "the clean amp section", "crunch": "the crunch/rhythm amp", "lead": "the lead amp",
                              "high": "the high-gain amp", "acoustic": "a clean amp with the cab bypassed"}},
    "kemper_qc": {"label": "Kemper / Quad Cortex / Fractal", "kind": "modeler",
                  "models": {"clean": "a clean American profile/capture", "crunch": "a British crunch profile/capture",
                             "lead": "a hot-rodded British lead profile", "high": "a modern high-gain profile",
                             "acoustic": "an acoustic/DI profile"}},
    "spark": {"label": "Positive Grid Spark / BIAS", "kind": "modeler",
              "models": {"clean": "Silver 120 / Twin-style clean", "crunch": "British crunch model", "lead": "Plexi/800-style lead",
                         "high": "Recto/modern high-gain", "acoustic": "acoustic sim"}},
}

PEDALS = {
    "tube_screamer": {"label": "Tube Screamer-style overdrive", "role": "drive", "knobs": ["drive", "tone", "level"], "range": (2, 6)},
    "blues_driver": {"label": "Blues Driver / transparent OD", "role": "drive", "knobs": ["gain", "tone", "level"], "range": (2, 6)},
    "ds1": {"label": "DS-1-style distortion", "role": "drive", "knobs": ["dist", "tone", "level"], "range": (5, 8)},
    "rat": {"label": "RAT-style distortion", "role": "drive", "knobs": ["distortion", "filter", "volume"], "range": (4, 9)},
    "metal_zone": {"label": "High-gain distortion pedal", "role": "drive", "knobs": ["dist", "low", "mid", "high", "level"], "range": (7, 10)},
    "big_muff": {"label": "Big Muff-style fuzz", "role": "fuzz", "knobs": ["sustain", "tone", "volume"], "range": (7, 10)},
    "fuzz_face": {"label": "Fuzz Face-style fuzz", "role": "fuzz", "knobs": ["fuzz", "volume"], "range": (6, 10)},
    "compressor": {"label": "Compressor", "role": "compression"},
    "wah": {"label": "Wah pedal", "role": "wah"},
    "chorus": {"label": "Chorus", "role": "chorus"},
    "phaser": {"label": "Phaser", "role": "phaser"},
    "flanger": {"label": "Flanger", "role": "flanger"},
    "tremolo": {"label": "Tremolo", "role": "tremolo"},
    "octaver": {"label": "Octaver", "role": "octave"},
    "delay": {"label": "Delay", "role": "delay"},
    "reverb": {"label": "Reverb", "role": "reverb"},
    "noise_gate": {"label": "Noise gate", "role": "gate"},
    "multi_fx": {"label": "Multi-effects unit", "role": "multi"},
}


def catalog() -> dict:
    return {"amps": {k: {"label": v["label"], "kind": v["kind"]} for k, v in AMPS.items()},
            "pedals": {k: {"label": v["label"], "role": v["role"]} for k, v in PEDALS.items()}}


def _k(x: float) -> float:
    return round(max(0.0, min(10.0, x)), 1)


def _effect(tone: dict, key: str) -> dict | None:
    for e in tone.get("effects", []):
        if e["key"] == key:
            return e
    return None


def recommend(tone: dict, amp: str = "generic", pedals: list[str] | None = None, guitar: str | None = None) -> dict:
    if not tone or not tone.get("available"):
        return {"chain": [], "notes": ["No tone analysis available for this stem."]}
    pedals = [p for p in (pedals or []) if p in PEDALS]
    has_role = {PEDALS[p]["role"] for p in pedals}
    multi = "multi" in has_role
    amp_info = AMPS.get(amp, AMPS["generic"])
    gain = tone["gain"]["gain_index"]
    eq = tone["eq"]["suggested_amp_eq"]
    chain: list[dict] = []
    notes: list[str] = []

    def have(role: str) -> bool:
        return multi or role in has_role or amp_info["kind"] == "modeler" or not pedals

    comp = _effect(tone, "compression")
    if comp and comp["detected"]:
        amt = comp["details"].get("amount", 0.5)
        chain.append({"block": "Compressor", "tier": comp["tier"], "available": have("compression"),
                      "settings": {"ratio": comp["details"].get("estimated_ratio"), "sustain/threshold": _k(3 + amt * 6),
                                   "attack": "medium (10-30 ms) to keep pick attack", "level": "unity"}})
    wah = _effect(tone, "wah")
    if wah and wah["detected"]:
        chain.append({"block": "Wah", "tier": wah["tier"], "available": have("wah"),
                      "settings": {"sweep": f"≈ {wah['details'].get('sweep_octaves')} octaves", "usage": "rocked with the rhythm"}})
    octv = _effect(tone, "octave")
    if octv and octv["detected"]:
        chain.append({"block": "Octaver", "tier": octv["tier"], "available": have("octave"),
                      "settings": {"sub_octave": "on", "dry": "100%", "octave_down": "40-60%"}})

    # ---- gain staging: pedal(s) + amp
    fuzz = _effect(tone, "fuzz")
    channel = "clean" if gain < 2 else "crunch" if gain < 5.5 else "lead" if gain < 8 else "high"
    amp_gain = gain
    drive_pedals = [p for p in pedals if PEDALS[p]["role"] in ("drive", "fuzz")]
    max_amp_gain = amp_info.get("max_gain", 10)
    if fuzz and fuzz["detected"] and any(PEDALS[p]["role"] == "fuzz" for p in drive_pedals):
        p = next(p for p in drive_pedals if PEDALS[p]["role"] == "fuzz")
        chain.append({"block": PEDALS[p]["label"], "tier": fuzz["tier"], "available": True,
                      "settings": {PEDALS[p]["knobs"][0]: 8, PEDALS[p]["knobs"][1]: _k(eq["treble"]), "volume": "unity+"}})
        amp_gain = 2.5
        channel = "clean"
    elif gain >= 2 and (gain > max_amp_gain or (drive_pedals and gain < 7.5)):
        choice = None
        for p in drive_pedals:
            lo, hi = PEDALS[p].get("range", (0, 10))
            if lo <= gain <= hi + 1:
                choice = p
                break
        choice = choice or (drive_pedals[0] if drive_pedals else None)
        if choice:
            lo, hi = PEDALS[choice].get("range", (0, 10))
            drv = _k((gain - lo) / max(1, hi - lo) * 7 + 2)
            k = PEDALS[choice]["knobs"]
            settings = {k[0]: drv, k[1]: _k(eq["treble"]), k[-1]: "unity or slightly above"}
            chain.append({"block": PEDALS[choice]["label"], "tier": tone["gain"]["tier"], "available": True, "settings": settings})
            amp_gain = min(max_amp_gain, max(2.0, gain - 3))
            channel = "clean" if amp_gain < 2.5 else "crunch"
        elif gain > max_amp_gain:
            notes.append("Your amp cannot reach this much gain on its own: a drive/distortion pedal is recommended.")
            chain.append({"block": "Overdrive/distortion pedal (recommended purchase)", "tier": tone["gain"]["tier"],
                          "available": False, "settings": {"gain": _k(gain), "tone": _k(eq["treble"])}})
            amp_gain = max_amp_gain

    amp_block = {"block": f"Amp – {amp_info['label']}", "tier": tone["gain"]["tier"], "available": True,
                 "settings": {"channel": channel, "gain": _k(amp_gain), "bass": eq["bass"], "mid": eq["mid"],
                              "treble": eq["treble"], "presence": eq["presence"], "master": "to taste"}}
    if amp_info["kind"] == "modeler":
        amp_block["settings"]["model"] = amp_info["models"].get(channel, amp_info["models"].get("crunch"))
        cab = "4x12 closed-back with a dynamic mic near the cap edge" if gain >= 5 else "1x12/2x12 open-back"
        amp_block["settings"]["cab"] = cab
    amp_block["character"] = tone["amp_character"]["style"]
    chain.append(amp_block)

    gate = _effect(tone, "gate")
    if (gate and gate["detected"]) or gain >= 7:
        chain.append({"block": "Noise gate", "tier": gate["tier"] if gate and gate["detected"] else "possible",
                      "available": have("gate"),
                      "settings": {"threshold": "just above the hiss (≈ -50 to -60 dB)", "release": "fast (≈ 50 ms)"
                                   if gain >= 7 else "medium"}})
    for key, label in (("chorus", "Chorus"), ("phaser", "Phaser"), ("flanger", "Flanger"), ("tremolo", "Tremolo")):
        e = _effect(tone, key)
        if e and e["detected"]:
            d = e["details"]
            s = {"rate": f"{d.get('rate_hz')} Hz" if d.get("rate_hz") else "slow", "depth": d.get("depth", "moderate")}
            if key == "tremolo":
                s["depth"] = f"≈ {d.get('depth_db')} dB"
            chain.append({"block": label, "tier": e["tier"], "available": have(key), "settings": s})
    dly = _effect(tone, "delay")
    if dly and dly["detected"]:
        d = dly["details"]
        chain.append({"block": "Delay", "tier": dly["tier"], "available": have("delay"),
                      "settings": {"time": f"{d['time_ms']} ms" + (f" ({d['note_value']}, tap tempo)" if d.get("note_value") else ""),
                                   "feedback": f"{d['feedback_percent']}% (≈ {max(1, round(d['feedback_percent'] / 15))} audible repeats)",
                                   "mix": f"{d['estimated_mix_percent']}%"}})
    rev = _effect(tone, "reverb")
    if rev and rev["detected"]:
        d = rev["details"]
        chain.append({"block": "Reverb", "tier": rev["tier"], "available": have("reverb"),
                      "settings": {"type": d["type"], "decay": f"{d['rt60_s']} s", "mix": f"{d['estimated_mix_percent']}%",
                                   "pre_delay": "10-30 ms"}})

    missing = [b["block"] for b in chain if not b["available"]]
    if pedals and missing:
        notes.append("Not in your gear list: " + ", ".join(missing) + ". A multi-effects unit or plugin can cover these.")
    if guitar == "single_coil":
        notes.append("With single coils, add +1 to gain and cut treble by ~1 to approach a humbucker recording.")
    elif guitar == "humbucker":
        notes.append("With humbuckers, reduce gain ~1 step if the original sounds like single coils (glassy, thin).")
    notes.append(tone.get("disclaimer", ""))
    return {"chain": chain, "notes": [n for n in notes if n], "amp": amp, "pedals": pedals}
