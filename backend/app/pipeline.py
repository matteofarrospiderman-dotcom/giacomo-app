"""End-to-end processing pipeline (runs inside a worker process).

Stages and progress ranges::

    decode            0-5%    FFmpeg -> 44.1 kHz stereo, validation, preview + waveform
    separate          5-40%   Demucs htdemucs_6s (or DSP fallback)
    instruments      40-43%   stem presence & characterisation, multi-guitar pan split
    rhythm           43-47%   tempo, beats, meter, downbeats
    harmony          47-50%   key, chords
    transcribe       50-88%   per stem: ensemble notes -> techniques -> parts ->
                              tuning -> fingering -> quantisation -> review flags
    tone             88-97%   effects analysis per guitar part
    finalize         97-100%  chord cross-validation, save

Large arrays are written to disk as soon as they are produced and only downsampled
mono copies stay in memory, which keeps a 10-minute song well under 2 GB.
"""

from __future__ import annotations

import gc
import time
import traceback

import numpy as np
import librosa

from . import storage
from .analysis import harmony, instruments, rhythm as rhythm_mod, tuning as tuning_mod
from .audio import io as aio
from .config import settings
from .guitar import learning
from .guitar.track import build_track
from .guitar.tunings import TUNINGS, Tuning, custom_tuning, get_tuning
from .separation import STEM_LABELS, get_separator
from .separation.guitar_split import split_guitars
from .tone.analyzer import analyze_tone
from .transcription import dsp_poly, echo, ensemble, parts as parts_mod, techniques

SR = settings.analysis_sr
SR22 = 22050


class Progress:
    def __init__(self, pid: str):
        self.pid = pid
        self.last = 0.0
        self.t_last = 0.0

    def __call__(self, stage: str, frac: float, lo: float, hi: float, msg: str) -> None:
        p = lo + (hi - lo) * max(0.0, min(1.0, frac))
        now = time.time()
        if p - self.last < 0.004 and now - self.t_last < 1.5 and frac < 1:
            return
        self.last, self.t_last = p, now
        storage.set_status(self.pid, "processing", stage, p, msg)

    def sub(self, stage: str, lo: float, hi: float):
        return lambda f, m="": self(stage, f, lo, hi, m)


def _mono22(x: np.ndarray, sr: int = SR) -> np.ndarray:
    m = x.mean(axis=0) if x.ndim == 2 else x
    return librosa.resample(m.astype(np.float32), orig_sr=sr, target_sr=SR22).astype(np.float32)


def _save_stem(pdir, name: str, audio: np.ndarray) -> dict:
    wav = pdir / "stems" / f"{name}.wav"
    aio.write_wav(wav, audio, SR)
    prev = aio.encode_preview(wav, pdir / "stems" / f"{name}.mp3")
    peaks = aio.compute_peaks(audio, SR, 50)
    storage.write_json(pdir / "stems" / f"{name}.peaks.json", peaks)
    return {"wav": f"stems/{name}.wav", "preview": f"stems/{prev.name}" if prev != wav else f"stems/{name}.wav",
            "peaks": f"stems/{name}.peaks.json"}


def _resolve_tuning(opt: str | dict | None, detected: dict | None, instrument: str) -> Tuning:
    if isinstance(opt, dict) and opt.get("notes"):
        return custom_tuning(opt["notes"], opt.get("name", "Custom"), instrument)
    if isinstance(opt, str) and opt in TUNINGS and TUNINGS[opt].instrument == instrument:
        return get_tuning(opt)
    if detected:
        b = detected["best"]
        return TUNINGS.get(b["id"]) or custom_tuning(b["notes"], b["name"], instrument)
    return get_tuning("standard" if instrument == "guitar" else "bass_standard")


def run_ensemble(audio: np.ndarray, instrument: str, prog) -> dict:
    y22 = _mono22(audio)
    res = ensemble.transcribe(y22, instrument, learning.detection_params(), progress=lambda f, m: prog(0.6 * f, m))
    res["y22"] = y22
    res["instrument"] = instrument
    return res


HARMONIC_BLEED = {19: 0.65, 24: 0.65, 28: 0.65, 31: 0.65, 36: 0.65, 12: 0.45}


def cross_stem_cleanup(results: list[dict], approximate_separation: bool = False) -> dict:
    """Remove notes that are bleed from another stem.

    * guitar notes sitting on an upper harmonic of a simultaneous bass note, with weak
      evidence of their own (the octave is treated more leniently: guitars often double
      bass riffs an octave up);
    * bass-stem notes in the guitar register that duplicate a guitar note (same pitch,
      same onset) and are less confident there;
    * guitar notes in the bass register that duplicate a bass note (unison bleed) - only
      trusted strongly when separation was approximate (DSP), where bleed is certain;
    * notes duplicated across guitar parts that came from the stereo splitter: the note is
      kept in the part where the model heard it loudest.
    """
    guitars = [r for r in results if r["instrument"] == "guitar"]
    basses = [r for r in results if r["instrument"] == "bass"]
    removed = {"guitar_bass_harmonics": 0, "bass_guitar_duplicates": 0, "guitar_bass_unison": 0,
               "guitar_part_duplicates": 0}
    unison_conf = 0.95 if approximate_separation else 0.5
    for g in guitars:
        keep = []
        for n in g["notes"]:
            # below standard E2 the bass is by far the likelier owner of a shared note
            dup = n.pitch < 40 and any(
                m.pitch == n.pitch and abs(m.start - n.start) < 0.06
                and n.confidence < (unison_conf if approximate_separation else min(unison_conf, m.confidence))
                for b in basses for m in b["notes"])
            if dup:
                n.flags.add("rejected:bleed_from_bass")
                g["rejected"].append(n)
                removed["guitar_bass_unison"] += 1
            else:
                keep.append(n)
        g["notes"] = keep
    for i, g1 in enumerate(guitars):
        for g2 in guitars[i + 1:]:
            drop1, drop2 = set(), set()
            for a in g1["notes"]:
                for b in g2["notes"]:
                    if a.pitch == b.pitch and abs(a.start - b.start) < 0.05:
                        if a.amplitude >= b.amplitude:
                            drop2.add(id(b))
                        else:
                            drop1.add(id(a))
            for res, drop in ((g1, drop1), (g2, drop2)):
                if drop:
                    res["rejected"].extend(n for n in res["notes"] if id(n) in drop)
                    for n in res["notes"]:
                        if id(n) in drop:
                            n.flags.add("rejected:duplicate_of_other_guitar_part")
                    res["notes"] = [n for n in res["notes"] if id(n) not in drop]
                    removed["guitar_part_duplicates"] += len(drop)
    for g in guitars:
        keep = []
        for n in g["notes"]:
            bleed = False
            for b in basses:
                for m in b["notes"]:
                    iv = n.pitch - m.pitch
                    if iv in HARMONIC_BLEED and abs(n.start - m.start) < 0.06 and n.overlaps(m) > 0.5 * n.duration \
                            and n.confidence < HARMONIC_BLEED[iv]:
                        bleed = True
                        break
                if bleed:
                    break
            if bleed:
                n.flags.add("rejected:bleed_from_bass")
                g["rejected"].append(n)
                removed["guitar_bass_harmonics"] += 1
            else:
                keep.append(n)
        g["notes"] = keep
    for b in basses:
        keep = []
        for m in b["notes"]:
            dup = m.pitch >= 38 and any(
                n.pitch == m.pitch and abs(n.start - m.start) < 0.05 and m.confidence < n.confidence + 0.1
                for g in guitars for n in g["notes"])
            if dup:
                m.flags.add("rejected:bleed_from_guitar")
                b["rejected"].append(m)
                removed["bass_guitar_duplicates"] += 1
            else:
                keep.append(m)
        b["notes"] = keep
    for r in results:
        r["polyphony"] = ensemble._polyphony(r["notes"], len(r["polyphony"]))
        r["stats"]["kept"] = len(r["notes"])
    return removed


def merge_echo_evidence(tone: dict, echo_info: dict | None, bpm: float) -> None:
    """Combine the note-domain delay estimate with the envelope-based one."""
    if not tone.get("available") or not echo_info:
        return
    for e in tone["effects"]:
        if e["key"] != "delay":
            continue
        ms = round(echo_info["time_s"] * 1000)
        d = e["details"]
        if e["detected"] and abs(d.get("time_ms", 0) - ms) < 40:
            e["confidence"] = round(min(0.95, e["confidence"] + 0.15), 2)
        elif not e["detected"]:
            beat = 60.0 / max(bpm, 1)
            names = {0.25: "1/16", 0.5: "1/8", 0.75: "dotted 1/8", 1.0: "1/4", 1 / 3: "1/8 triplet", 1.5: "dotted 1/4"}
            rel = echo_info["time_s"] / beat
            best = min(names, key=lambda r: abs(rel - r))
            synced = abs(rel - best) < 0.06 * best
            d.update({"detected": True, "time_ms": ms, "feedback_percent": d.get("feedback_percent", 30),
                      "estimated_mix_percent": d.get("estimated_mix_percent", 30), "tempo_synced": synced,
                      "note_value": names[best] if synced else None})
            e["detected"] = True
            e["confidence"] = echo_info["confidence"]
            e["summary"] = f"{ms} ms" + (f" ({names[best]})" if synced else "") + " – from repeated quieter notes"
        else:
            continue
        from .tone.analyzer import tier  # noqa: PLC0415

        e["tier"] = tier(e["confidence"])
        d.setdefault("evidence", []).append(
            f"{echo_info['pairs']} notes are repeated {echo_info['time_s'] * 1000:.0f} ms later at lower level "
            f"({echo_info.get('removed', 0)} echo notes removed from the tab)")


def build_part_tracks(pid: str, track_id: str, name: str, stem_name: str, res: dict, instrument: str, rhythm: dict,
                      analysis: dict, tuning_opt, prog, overlap_sal=None, allow_split: bool = True,
                      role: str = "full", strings: int | None = None) -> list[dict]:
    """Techniques -> tuning -> (lead/rhythm split) -> fingering -> tracks."""
    from .guitar.track import note_to_dict  # noqa: PLC0415

    y22 = res["y22"]
    duration = len(y22) / SR22
    priors = learning.load_priors()
    prog(0.6, "Removing delay echoes")
    res["notes"], echoes, echo_info = echo.suppress_echoes(res["notes"], y22, res.get("salience"))
    res["rejected"].extend(echoes)
    prog(0.62, "Detecting playing techniques")
    tech = techniques.detect(res["notes"], y22, res["f0"], res["polyphony"],
                             beat_times=np.array(rhythm.get("beats", [])))
    prog(0.72, "Identifying tuning")
    detected = None
    if tuning_opt in (None, "auto"):
        note_dicts = [note_to_dict(n, i) for i, n in enumerate(res["notes"])]
        detected = tuning_mod.detect_tuning(note_dicts, instrument, strings, analysis.get("key"),
                                            analysis.get("reference_deviation_cents", 0.0))
    tuning = _resolve_tuning(tuning_opt, detected, instrument)

    groups = [(track_id, name, role, res["notes"])]
    split_info = None
    if allow_split and instrument == "guitar":
        tmp = [{"pitch": n.pitch, "start": n.start, "end": n.end, "_ev": n} for n in res["notes"]]
        sp = parts_mod.split_lead_rhythm(tmp, tuning.strings)
        split_info = {k: sp[k] for k in ("adopted", "overlay_notes", "confidence")}
        if sp["adopted"]:
            groups = [(f"{track_id}_rhythm", f"{name} – rhythm", "rhythm", [d["_ev"] for d in sp["rhythm"]]),
                      (f"{track_id}_lead", f"{name} – lead", "lead", [d["_ev"] for d in sp["lead"]])]
    tracks = []
    for k, (tid, tname, trole, evs) in enumerate(groups):
        prog(0.8 + 0.18 * k / len(groups), f"Fingering {tname}")
        extra = {"tuning_detection": detected, "dead_strums": tech["dead_strums"], "strums": tech["strums"],
                 "part_split": split_info, "transcription_stats": res["stats"], "echo_analysis": echo_info}
        tr = build_track(tid, tname, stem_name, instrument, trole, evs, res["rejected"] if k == 0 else [], tuning,
                         rhythm, duration, res["engines"], extra=extra, priors=priors, overlap_salience=overlap_sal,
                         own_salience=res["salience"], chords=analysis.get("chords"))
        storage.save_track(pid, tr)
        tracks.append(tr)
    contours = [n.contour for n in res["notes"] if n.contour is not None and n.duration > 0.7]
    for tr in tracks:
        tr["_contours"] = contours
    return tracks


def transcribe_part(pid: str, track_id: str, name: str, stem_name: str, audio: np.ndarray, instrument: str,
                    rhythm: dict, analysis: dict, tuning_opt, prog, overlap_sal=None, allow_split: bool = True,
                    role: str = "full", strings: int | None = None) -> list[dict]:
    """Single-stem convenience wrapper (used for on-demand transcription)."""
    res = run_ensemble(audio, instrument, prog)
    tracks = build_part_tracks(pid, track_id, name, stem_name, res, instrument, rhythm, analysis, tuning_opt, prog,
                               overlap_sal, allow_split, role, strings)
    del res
    gc.collect()
    return tracks


def process(pid: str) -> None:
    prog = Progress(pid)
    meta = storage.load_project(pid)
    opts = meta.get("options", {})
    pdir = storage.project_dir(pid)
    src = pdir / meta["source"]["file"]
    t_start = time.time()

    # ------------------------------------------------------------------ decode
    prog("decode", 0.1, 0, 0.05, "Decoding audio")
    audio = aio.decode(src, SR, 2)
    q = aio.validate_signal(audio, SR, settings.min_duration_s, settings.max_duration_s)
    aio.write_wav(pdir / "audio" / "mix.wav", audio, SR)
    prev = aio.encode_preview(pdir / "audio" / "mix.wav", pdir / "audio" / "mix.mp3")
    storage.write_json(pdir / "audio" / "peaks.json", aio.compute_peaks(audio, SR, 100))
    duration = audio.shape[-1] / SR
    analysis: dict = {"duration": duration, "signal": q, "mix": {"wav": "audio/mix.wav", "preview": f"audio/{prev.name}",
                                                               "peaks": "audio/peaks.json"}}
    prog("decode", 1.0, 0, 0.05, f"Decoded {duration:.1f}s")

    # ------------------------------------------------------------------ separate
    sep, sep_notes = get_separator(opts.get("separator", settings.separator))
    prog("separate", 0.0, 0.05, 0.4, f"Separating sources ({sep.name})")
    result = sep.separate(audio, SR, prog.sub("separate", 0.05, 0.4))
    analysis["separation"] = {"engine": result.engine, "model": result.model, "quality": result.quality,
                              "notes": sep_notes + result.notes}
    stems_info = {}
    mono = {}
    guitar_stereo = None
    for name, st in result.stems.items():
        stems_info[name] = {"name": name, "label": STEM_LABELS.get(name, name.title()), **_save_stem(pdir, name, st)}
        mono[name] = _mono22(st)
        if name == "guitar":
            guitar_stereo = st
    prog("instruments", 0.2, 0.4, 0.43, "Characterising instruments")
    ident = instruments.identify(result.stems, audio, SR)
    for name, d in ident.items():
        stems_info[name]["analysis"] = d

    # multi-guitar split by stereo position
    guitar_parts = []
    if guitar_stereo is not None and ident.get("guitar", {}).get("present", True):
        sp = split_guitars(guitar_stereo, SR)
        analysis["guitar_split"] = {"analysis": {k: v for k, v in sp["analysis"].items()},
                                    "double_tracked": sp["double_tracked"], "parts": [p["id"] for p in sp["parts"]]}
        if sp["parts"] and not sp["double_tracked"] and opts.get("split_guitars", True):
            for p in sp["parts"]:
                info = _save_stem(pdir, p["id"], p["audio"])
                stems_info[p["id"]] = {"name": p["id"], "label": p["label"], "parent": "guitar", "pan": p["pan"],
                                       "share": p["share"], **info,
                                       "analysis": instruments.describe(p["audio"], audio, SR)}
                guitar_parts.append((p["id"], p["label"], p["audio"]))
        elif sp["double_tracked"]:
            analysis["separation"]["notes"].append(
                "Guitars on the left and right play essentially the same part (double-tracked); transcribed once.")
        del sp
    analysis["stems"] = stems_info
    analysis["instruments"] = ident
    del result
    gc.collect()

    # ------------------------------------------------------------------ rhythm & harmony
    prog("rhythm", 0.1, 0.43, 0.47, "Tracking tempo, beats and meter")
    mix22 = _mono22(audio)
    rh = rhythm_mod.analyze_rhythm(mix22, SR22, mono.get("drums"))
    analysis["rhythm"] = rh
    prog("harmony", 0.1, 0.47, 0.5, "Recognising key and chords")
    harm_src = mix22 - mono.get("drums", 0) - mono.get("vocals", 0) if "drums" in mono else mix22
    hm = harmony.analyze_harmony(harm_src, SR22, np.array(rh["beats"]), duration, mono.get("bass"))
    analysis.update({"key": hm["key"], "chords": hm["chords"], "reference_deviation_cents": hm["reference_deviation_cents"]})

    # salience of competing harmonic instruments, for bleed flags
    competitors = [mono[k] for k in ("piano", "other", "vocals") if k in mono]
    overlap_sal = dsp_poly.salience(np.sum(competitors, axis=0)) if competitors else None
    storage.save_analysis(pid, analysis)

    # ------------------------------------------------------------------ transcribe
    targets = []
    guitar_present = "guitar" in stems_info and ident.get("guitar", {}).get("present", True)
    if guitar_parts:
        for sid, label, aud in guitar_parts:
            targets.append((sid, label, sid, aud, "guitar"))
    elif guitar_present:
        targets.append(("guitar", "Guitar", "guitar", guitar_stereo, "guitar"))
    elif "other" in mono:
        # 4-stem models or no guitar stem: guitar usually lives in "other"
        import soundfile as sf  # noqa: PLC0415

        other, _ = sf.read(str(pdir / "stems" / "other.wav"), dtype="float32", always_2d=True)
        targets.append(("other", "Guitar (from 'other' stem)", "other", other.T, "guitar"))
    if opts.get("transcribe_bass", True) and "bass" in mono and ident.get("bass", {}).get("present", True):
        import soundfile as sf  # noqa: PLC0415

        bass, _ = sf.read(str(pdir / "stems" / "bass.wav"), dtype="float32", always_2d=True)
        targets.append(("bass", "Bass", "bass", bass.T, "bass"))

    tone_results = {}
    all_tracks = []
    span = (0.88 - 0.5) / max(1, len(targets))
    results = []
    for i, (tid, label, stem, aud, inst) in enumerate(targets):
        lo = 0.5 + i * span
        res = run_ensemble(aud, inst, prog.sub("transcribe", lo, lo + 0.6 * span))
        results.append(res)
    analysis["cross_stem_cleanup"] = cross_stem_cleanup(
        results, approximate_separation=analysis["separation"]["quality"] != "neural")
    for i, ((tid, label, stem, aud, inst), res) in enumerate(zip(targets, results)):
        lo = 0.5 + i * span
        sub = prog.sub("transcribe", lo + 0.6 * span, lo + span)
        tun_opt = opts.get("tuning", "auto") if inst == "guitar" else opts.get("bass_tuning", "auto")
        trs = build_part_tracks(pid, tid, label, stem, res, inst, rh, analysis, tun_opt, sub, overlap_sal,
                                allow_split=(inst == "guitar" and not guitar_parts),
                                strings=opts.get("strings") if inst == "guitar" else None)
        all_tracks.extend(trs)
        if inst == "guitar":
            prog("tone", i / max(1, len(targets)), 0.88, 0.97, f"Analysing tone of {label}")
            notes_all = [n for t in trs for n in t["notes"]]
            tone_results[stem] = analyze_tone(aud, SR, notes_all, rh["bpm"], duration, trs[0].get("_contours", []))
            merge_echo_evidence(tone_results[stem], trs[0].get("echo_analysis"), rh["bpm"])
        for t in trs:
            t.pop("_contours", None)
        results[i] = None
        gc.collect()

    # key from the transcription (pitch-class durations) fused with the chroma estimate
    gnotes_all = [n for t in all_tracks for n in t["notes"]]
    if gnotes_all:
        analysis["key"] = harmony.fuse_key(analysis["key"], gnotes_all, analysis.get("chords"))
        meta_key = analysis["key"]["name"]
    else:
        analysis["key"].pop("_corr", None)
        meta_key = analysis["key"]["name"]

    # ------------------------------------------------------------------ finalize
    prog("finalize", 0.3, 0.97, 1.0, "Cross-checking chords with transcription")
    gnotes = [n for t in all_tracks if t["instrument"] == "guitar" for n in t["notes"]]
    analysis["chord_validation"] = harmony.cross_validate_with_notes(analysis["chords"], gnotes)
    analysis["tone"] = tone_results
    analysis["tracks"] = [{"id": t["id"], "name": t["name"], "stem": t["stem"], "instrument": t["instrument"],
                           "role": t["role"], "tuning": t["tuning"], "stats": t["stats"]} for t in all_tracks]
    guitar_tracks = [t for t in all_tracks if t["instrument"] == "guitar"]
    if guitar_tracks:
        analysis["tuning"] = guitar_tracks[0].get("tuning_detection") or {"best": guitar_tracks[0]["tuning"],
                                                                          "confidence": 1.0, "user_selected": True}
    analysis["processing_seconds"] = round(time.time() - t_start, 1)
    storage.save_analysis(pid, analysis)
    meta = storage.load_project(pid)
    meta["summary"] = {
        "duration": duration, "bpm": rh["bpm"], "time_signature": rh["time_signature"], "key": meta_key,
        "tuning": analysis.get("tuning", {}).get("best", {}).get("name"), "tracks": len(all_tracks),
        "separation": analysis["separation"]["engine"],
        "notes": sum(t["stats"]["notes"] for t in all_tracks),
    }
    storage.save_project(meta)
    storage.set_status(pid, "done", "done", 1.0, f"Finished in {analysis['processing_seconds']}s")


def transcribe_stem(pid: str, stem: str, instrument: str, tuning_opt) -> None:
    """On-demand transcription of any stem (e.g. 'other' or 'piano' as guitar)."""
    import soundfile as sf  # noqa: PLC0415

    prog = Progress(pid)
    analysis = storage.load_analysis(pid)
    pdir = storage.project_dir(pid)
    info = analysis["stems"].get(stem)
    if not info:
        raise ValueError(f"Unknown stem {stem}")
    data, sr = sf.read(str(pdir / info["wav"]), dtype="float32", always_2d=True)
    aud = data.T
    if sr != SR:
        aud = librosa.resample(aud, orig_sr=sr, target_sr=SR)
    tid = f"{stem}_{instrument}"
    label = f"{info['label']} ({instrument})"
    trs = transcribe_part(pid, tid, label, stem, aud, instrument, analysis["rhythm"], analysis, tuning_opt,
                          prog.sub("transcribe", 0.05, 0.9), None, allow_split=instrument == "guitar")
    if instrument == "guitar":
        prog("tone", 0.5, 0.9, 0.99, "Analysing tone")
        notes_all = [n for t in trs for n in t["notes"]]
        analysis.setdefault("tone", {})[stem] = analyze_tone(aud, SR, notes_all, analysis["rhythm"]["bpm"],
                                                             analysis["duration"], trs[0].get("_contours", []))
    existing = {t["id"] for t in analysis.get("tracks", [])}
    for t in trs:
        t.pop("_contours", None)
        if t["id"] not in existing:
            analysis.setdefault("tracks", []).append({"id": t["id"], "name": t["name"], "stem": t["stem"],
                                                      "instrument": t["instrument"], "role": t["role"],
                                                      "tuning": t["tuning"], "stats": t["stats"]})
    storage.save_analysis(pid, analysis)
    storage.set_status(pid, "done", "done", 1.0, f"Transcribed {label}")


def run_job(kind: str, pid: str, args: dict) -> None:
    """Entry point executed in the worker process."""
    from .config import ensure_dirs  # noqa: PLC0415

    ensure_dirs()
    try:
        if kind == "process":
            process(pid)
        elif kind == "transcribe_stem":
            transcribe_stem(pid, args["stem"], args.get("instrument", "guitar"), args.get("tuning", "auto"))
        else:
            raise ValueError(kind)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        try:
            meta = storage.load_project(pid)
            meta["status"].update({"state": "failed", "message": f"{type(e).__name__}: {e}", "error": tb[-4000:]})
            storage.save_project(meta)
        except Exception:  # noqa: BLE001
            pass
        raise
