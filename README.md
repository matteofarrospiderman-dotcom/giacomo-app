# TabForge: music-to-guitar-tab transcription

TabForge analyses an uploaded song or guitar recording (MP3, WAV, FLAC, OGG, M4A, AIFF…)
and reconstructs it as **playable guitar tablature**, standard notation, MIDI, separated
instrument stems, chords, tempo/meter, tuning and a **tone/effects analysis** with gear
recommendations. A synchronized editor lets you listen at any speed, review flagged notes
and correct mistakes.

It is not a plain audio-to-MIDI converter. Every note goes through:

1. neural source separation,
2. an ensemble of transcription engines with cross-validation and confidence scoring,
3. effects-aware cleanup (overtone ghosts, delay echoes, bleed from other stems),
4. articulation detection,
5. guitar-specific string/fret optimisation with a model of the fretting hand,
6. rhythm quantisation.

**Accuracy disclaimer:** automatic transcription of mixed commercial recordings is not
solved. TabForge never claims its tabs are correct. Every note has a confidence level and
review flags, and alternatives are offered wherever it is unsure.

---

## Contents

- [Quick start](#quick-start)
- [Setup in detail](#setup-in-detail) (dependencies, models, GPU)
- [Running](#running)
- [Using the app](#using-the-app)
- [Architecture](#architecture)
- [How each feature works](#how-each-feature-works) (algorithm, models, limitations, how to improve)
- [Measured accuracy](#measured-accuracy)
- [API](#api)
- [Testing](#testing)
- [Roadmap for transcription accuracy](#roadmap-for-transcription-accuracy)

---

## Quick start

Requirements: **Python 3.10–3.13**, **Node 18+**, **FFmpeg** on `PATH`
(`brew install ffmpeg` / `sudo apt install ffmpeg` / `winget install ffmpeg`).

```bash
# 1. Backend: venv, dependencies, model weights
cd backend
./scripts/setup.sh            # or: ./scripts/setup.sh --core  (no PyTorch, DSP fallbacks)

# 2. Frontend
cd ../frontend
npm install
npm run build                 # production bundle, served by the backend

# 3. Run (one process serves API + UI)
cd ../backend
. .venv/bin/activate
uvicorn app.main:app --port 8000
# open http://localhost:8000
```

For frontend development with hot reload, run `npm run dev` in `frontend/` (port 5173,
which proxies `/api` to the backend on port 8000) and open http://localhost:5173.

Want a test file with known answers? Run
`python scripts/make_demo_song.py data/demo`. It writes `demo_song.wav` (a drums, bass,
rhythm guitar and lead guitar mix) plus `demo_song.truth.json`.

---

## Setup in detail

### Python dependencies

| File | Contents | Needed for |
|---|---|---|
| `backend/requirements.txt` | numpy, scipy, librosa, soundfile, onnxruntime, FastAPI, mido, PyGuitarPro, reportlab | everything (CPU) |
| `basic-pitch==0.4.0` (installed with `--no-deps`) | Spotify's Basic Pitch package, used only for its ONNX model file | polyphonic transcription |
| `backend/requirements-ml.txt` | torch, torchaudio, demucs, torchcrepe | neural separation + CREPE pitch tracking (strongly recommended) |
| `backend/requirements-dev.txt` | pytest, httpx | tests |

Manual installation, equivalent to `scripts/setup.sh`:

```bash
cd backend
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pip install --no-deps "basic-pitch==0.4.0"     # its TensorFlow pin has no wheels for new Pythons
# CPU-only PyTorch (smaller download):
pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
# or with CUDA (Linux, NVIDIA): pip install -r requirements-ml.txt
python scripts/download_models.py
```

Why Basic Pitch is installed with `--no-deps`: the package pins TensorFlow < 2.15, which
does not install on Python ≥ 3.12. It also ships an ONNX export of the same network, and
TabForge runs that export with onnxruntime using its own windowing and note-decoding code.
You get the same model without TensorFlow.

### Model downloads

`python scripts/download_models.py` prepares everything for offline use:

| Model | Source | Size | Location |
|---|---|---|---|
| Basic Pitch (ICASSP 2022) | `basic-pitch` wheel | 0.2 MB | copied to `backend/models/basic_pitch/nmp.onnx` |
| Demucs `htdemucs_6s` | `dl.fbaipublicfiles.com` via torch.hub | ~80 MB | `~/.cache/torch/hub` + `backend/models/demucs` |
| CREPE tiny/full | bundled in `torchcrepe` | 2 / 85 MB | inside the package |

Other Demucs models: `python scripts/download_models.py --demucs htdemucs_ft`, then set
`TABFORGE_DEMUCS_MODEL=htdemucs_ft`. That model is 4-stem only, so guitar ends up in "other".
For machines without internet access, run the script on a connected machine and copy
`backend/models/` across.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `TABFORGE_DATA_DIR` | `backend/data` | projects, stems, learned priors |
| `TABFORGE_MODELS_DIR` | `backend/models` | local model weights |
| `TABFORGE_SEPARATOR` | `auto` | `auto` / `demucs` / `dsp` |
| `TABFORGE_DEMUCS_MODEL` | `htdemucs_6s` | any Demucs model name |
| `TABFORGE_DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `TABFORGE_F0_TRACKER` | `auto` | `crepe` / `pyin` / `none` |
| `TABFORGE_CREPE_MODEL` | `tiny` | `tiny` (fast) or `full` (more accurate) |
| `TABFORGE_WORKERS` | `1` | parallel background jobs (each loads its own models) |
| `TABFORGE_MAX_UPLOAD_MB` / `TABFORGE_MAX_DURATION_S` | `250` / `900` | upload limits |
| `TABFORGE_CORS_ORIGINS` | `http://localhost:5173,…` | allowed dev origins |

GPU: install CUDA-enabled PyTorch and `onnxruntime-gpu`. Demucs, CREPE and Basic Pitch
pick up the GPU automatically (`/api/health` reports the device). On CPU, Demucs needs
roughly 0.5–2× the song duration.

---

## Running

| Mode | Command | URL |
|---|---|---|
| Production (single process) | `npm run build` in `frontend/`, then `uvicorn app.main:app --port 8000` in `backend/` | http://localhost:8000 |
| Development | `uvicorn app.main:app --reload --port 8000` plus `npm run dev` | http://localhost:5173 |
| API docs (Swagger) | the API server must be running | http://localhost:8000/docs |

Projects that were processing when the server stopped are re-queued automatically on
startup.

---

## Using the app

1. **Dashboard**
   - Drop a file and choose the options: separation engine, tuning (auto or fixed),
     string count, splitting multiple guitars, and bass transcription.
   - The engine status chips show which models are available.
2. **Processing**
   - A live progress view lists each pipeline stage and its log.
3. **Analysis**
   - The waveform shows the beat grid and downbeats, and a chord lane sits underneath.
     Click to seek; drag to set a loop.
   - Tempo, meter, key and tuning cards each show a confidence level and the alternatives.
   - The stem mixer has mute, solo and volume per stem, plus **"Tab as guitar / as bass"**
     buttons to transcribe any stem on demand.
   - Further panels show instrument identification (electric vs acoustic, what the "other"
     stem probably holds), the tuning ranking with its evidence, and the stereo map of the
     guitars.
4. **Tab Editor**
   - **Tab display and playback**
     - The tab scrolls with playback and highlights the notes that are sounding.
     - Note colours show confidence: white = high, amber = medium, red = low,
       violet = edited by you. A dashed box marks a flagged note.
     - Speed runs from 25 % to 150 % with pitch preserved. Drag on the tab or waveform to
       set an A–B loop. A metronome can click on the detected beats.
   - **Fretboard**
     - Shows the sounding notes live.
     - Click a position to move the selected note there; dashed rings are the alternatives.
   - **Note inspector**
     - Change string, fret or pitch, nudge timing and length, toggle techniques, apply an
       alternative fingering, lock or delete the note.
     - The evidence section shows what each engine reported.
   - **Review panel**
     - Lists every flagged note (ambiguous pitch, possible overtone, uncertain string,
       impossible fingering, unusual timing, overlapping instrument, chord disagreement).
     - Click an entry to jump to the note.
   - **Track settings**
     - Tuning (re-fingers the part), capo, time signature, bar-line shift.
     - **Compare alternative fingerings** (open-position / minimal-movement) side by side.
     - **Re-optimise fingering around my edits**.
     - Restore rejected candidates.
   - **Keyboard shortcuts**

     | Key | Action |
     |---|---|
     | `Space` | play / pause |
     | `←` `→` | previous / next note |
     | `↑` `↓` | move the note to another string |
     | `0`–`9` | type a fret |
     | `+` `-` | pitch up / down a semitone |
     | `Del` | delete the note |
     | `L` | loop on / off |
     | `Esc` | deselect |
5. **Tone Reconstruction**
   - Detected effects are grouped into *high confidence / probable / possible / no
     evidence*, each with its measured evidence ("Why?").
   - Also shown: amp character, EQ character and the long-term spectrum.
   - Pick your amp or modeler, your pickups and your pedals to get a concrete signal chain
     with knob settings.
6. **Export**
   - GP5, MusicXML, MIDI (performance timing or quantised), per-instrument MIDI (zip),
     PDF tab, text tab, and WAV/MP3 stems.

---

## Architecture

```
frontend/  React 18 + TypeScript + Vite (no UI framework; SVG/Canvas rendering)
  audio/engine.ts      multi-stem player: HTMLAudioElement (preservesPitch) → WebAudio gain per stem
  audio/pluck.ts       Karplus-Strong synth to audition single notes / chords
  components/          Dashboard, Workspace, AnalysisView, TabView (SVG tab), Fretboard,
                       NoteInspector, ReviewPanel, TrackPanel, ToneView, ExportView

backend/   FastAPI + background worker processes (ProcessPoolExecutor, spawn, recycled)
  app/main.py          REST API, upload validation, editing, exports, static frontend
  app/jobs.py          job queue; workers report progress by atomically rewriting project.json
  app/pipeline.py      orchestration of all stages (below)
  app/audio/io.py      FFmpeg decode / probe / validate, previews, waveform peaks
  app/separation/      demucs_sep (neural), dsp_sep (fallback), guitar_split (stereo multi-guitar)
  app/analysis/        rhythm (tempo, beats, meter, downbeats), harmony (key, chords),
                       tuning (analysis-by-synthesis), instruments (stem characterisation)
  app/transcription/   basic_pitch_onnx, dsp_poly (CQT salience), f0_tracker (CREPE/pYIN),
                       ensemble (fusion + confidence), echo (delay repeats), techniques,
                       parts (lead/rhythm split)
  app/guitar/          tunings, fingering (Viterbi optimiser), quantize, track (document +
                       review flags), learning (priors from user corrections)
  app/tone/            analyzer (effects detection), recommend (signal chain for user gear)
  app/export/          gp5, musicxml, midi, pdf, ascii_tab
  scripts/             setup.sh, download_models.py, make_demo_song.py, evaluate.py
  tests/               28 pytest tests incl. full HTTP round trip and synthetic ground truth
```

**Pipeline** (`app/pipeline.py`):

```
upload ─▶ FFmpeg decode/validate ─▶ source separation ─▶ instrument characterisation
       ─▶ multi-guitar stereo split ─▶ tempo/beats/meter ─▶ key/chords
       ─▶ per stem: ensemble transcription ─▶ cross-stem bleed cleanup ─▶ echo removal
          ─▶ techniques ─▶ tuning detection ─▶ lead/rhythm split ─▶ fingering optimisation
          ─▶ quantisation ─▶ review flags ─▶ tone analysis
       ─▶ chord ↔ transcription cross-validation ─▶ key fusion ─▶ save
```

**Design choices:**

- Every model sits behind a small interface (`Separator`, the transcriber functions, the
  f0 tracker), so it can be swapped or upgraded independently.
- Heavy work runs in separate processes, so a model crash or out-of-memory never takes the
  API down. Workers are recycled after each job to release memory.
- Large arrays are written to disk as soon as they are produced. Only 22 kHz mono copies
  stay in memory, which keeps a 10-minute song under about 2 GB.
- Storage is plain files: JSON plus WAV/MP3 per project, with atomic writes. There is no
  database to run.

---

## How each feature works

For each feature: **how it works**, **which model or algorithm powers it**, **known
limitations**, and **how accuracy can be improved**.

### Audio decoding and validation
- **How:**
  - `ffprobe` checks the container, codec and duration before a job is queued.
  - FFmpeg decodes to 44.1 kHz stereo float32.
  - The signal is validated for silence, length and clipping.
  - An MP3 preview and waveform peaks are generated for the browser.
- **Limits:** DRM-protected files cannot be read. Without FFmpeg, only WAV/FLAC/OGG are
  supported (through libsndfile).

### Instrument separation
- **How:** **Demucs `htdemucs_6s`** (Hybrid Transformer Demucs, Meta) produces drums, bass,
  vocals, **guitar**, piano and other stems.
  - If Demucs is unavailable, a **DSP fallback** takes over:
    - median-filter harmonic/percussive separation for drums,
    - a 160 Hz low-pass band for bass,
    - REPET-SIM plus centre-channel weighting for vocals,
    - the full-range remainder as guitar.
  - The UI labels the fallback clearly as approximate.
- **Limits:**
  - The guitar stem of `htdemucs_6s` is its weakest stem: keys and synths bleed into it.
  - The DSP fallback cannot really separate bass from guitar in the 80–250 Hz overlap.
    The cross-stem cleanup (below) mitigates this.
- **Improve:**
  - Use newer separators such as BS-RoFormer or Mel-RoFormer, or ensembles of several.
  - Fine-tune on guitar-heavy multitracks (MedleyDB, MoisesDB).
  - Run separation twice (on the stem, then on its residual).

### Multiple guitars
- **How:** two complementary strategies.
  1. **Stereo position** (ADRess-style azimuth analysis, `separation/guitar_split.py`):
     - Compute an energy-weighted pan histogram of the guitar stem and find its modes.
     - Resynthesise one stem per mode with soft masks.
     - If the parts' chroma is highly correlated, the part is reported as double-tracked
       and transcribed once.
  2. **Note level** (`transcription/parts.py`):
     - Chords are assigned to rhythm.
     - Melodic notes that start above a still-ringing chord are assigned to lead.
     - Voice-leading continuity decides the remaining single notes.
     - Groups that no single guitar could play are split by register.
  - After either split, notes duplicated across parts stay only in the part where they
    are loudest.
- **Limits:** two guitars at the same pan position, playing in the same register, cannot be
  separated reliably. No open model does this today.
- **Improve:** train a guitar-vs-guitar separation network on synthetic two-guitar mixes;
  use timbre embeddings to cluster notes by instrument.

### Pitch transcription (ensemble)
- **How:** three engines with different failure modes, fused in `transcription/ensemble.py`.
  - **Basic Pitch** (Spotify, ONNX) gives note, onset and contour posteriorgrams at 86 fps.
    - A custom decoder adds a *legato pass* that catches hammer-ons, pull-offs and slides,
      which have no pick attack.
  - **CQT harmonic-salience transcriber** (3 bins per semitone, spectral whitening,
    harmonic summation, sub-harmonic suppression) gives independent support for every
    candidate.
  - **CREPE** (torchcrepe) or **pYIN** gives a precise f0 in monophonic passages. It
    confirms pitches, exposes octave errors and provides fine contours for bends and
    vibrato.
  - Confidence is a weighted vote of the three.
  - Candidates are removed when they are:
    - overtone ghosts (an octave, twelfth or double octave above a louder simultaneous note),
    - harmonics or unisons bleeding from the bass stem,
    - out of the instrument's range,
    - too weak.
  - Removed candidates are **kept as restorable suggestions**, not thrown away.
  - DSP-only or f0-only notes that the network missed are recovered at low confidence.
- **Limits:**
  - Heavy distortion produces intermodulation ghosts.
  - Dense strummed chords lose inner voices.
  - Very fast passages (>14 notes/s) merge.
  - Basic Pitch is a general multi-instrument model, not guitar-specific.
- **Improve:**
  - Plug in a guitar-specific model such as a GuitarSet/GAPS-trained "high-resolution
    guitar transcription" (Riley et al. 2024) or a fine-tuned MT3 variant.
  - Train the fusion weights on labelled data instead of hand-tuning them.

### Effects-aware cleanup (delay echoes)
- **How:** `transcription/echo.py` histograms the lag between same-pitch notes where the
  later one is quieter, measuring loudness by harmonic salience at that pitch.
  - A sharp, well-supported peak (not explained by the riff's own rhythm) is the delay time.
  - Notes at 1× or 2× that lag are rejected as echoes.
  - The estimate also feeds the tone analysis as evidence.
- **Limits:** it needs at least five consistent echo pairs, so short phrases keep their
  echoes. This is a deliberate choice: it prefers keeping a note over deleting a real one.

### Articulations
- **How** (`transcription/techniques.py`, transparent rules over measured evidence):

  | Technique | Evidence |
  |---|---|
  | Bend / release / pre-bend | contour rises ≥ 0.4 semitone and holds, or returns. A "staircase" (E→F→F# without attacks) is merged back into one bent note. |
  | Vibrato | 3.5–9 Hz periodic pitch modulation |
  | Slide | glide in f0 between notes with a weak second attack |
  | Hammer-on / pull-off | legato with no broadband attack, confirmed only if the fingering puts both notes on the same string |
  | Palm mute | fast decay plus a darker attack on low strings |
  | Dead strums | broadband onsets with no pitched notes |
  | Strum direction | onset order across pitches, falling back to metric position |
  | Tremolo picking | fast repeated attacks of one pitch |
  | Let ring | a note sustaining across later attacks on other strings |

- **Limits:**
  - Natural, pinch and artificial harmonics, tapping, rakes and whammy-bar dives are not
    auto-detected. They can be added in the editor.
  - Palm-mute detection depends on timbre and is less reliable on heavily compressed mixes.
- **Improve:** train a frame-level technique classifier (e.g. on the GuitarSet and IDMT-SMT
  Guitar datasets) using the current rules as weak labels.

### Guitar-aware string/fret optimisation (core)
- **How** (`guitar/fingering.py`):
  - Notes within 35 ms form a group (chord, double-stop or single note).
  - For each group, every physically possible fingering is enumerated: one string per
    note, span within reach, at most four fingers, with a barre counting as one finger.
  - Each fingering gets an **intrinsic cost**:
    - stretch and fret height,
    - muted inner strings,
    - a bonus for common open shapes,
    - an impossible bend on an open string,
    - a *learned prior from your own corrections*.
  - Each step between groups gets a **transition cost** from a fretting-hand model:
    - The hand is tracked as an *interval* of feasible index-finger positions; notes
      inside the 4-fret box are free.
    - Shifts cost more when there is less time to make them.
    - Open strings never force a move.
    - Changing strings mid-phrase costs extra.
    - Keeping legato on one string earns a bonus.
    - Cutting off a ringing note costs extra.
    - A repeated chord should keep the same shape.
  - **Viterbi** dynamic programming finds the globally cheapest path.
  - Each note gets its local alternatives with cost deltas. Near-ties are flagged as
    `uncertain_string`.
  - Groups that are unplayable even after dropping their least-confident notes are flagged
    `impossible_fingering`.
  - Three weight profiles give whole alternative transcriptions to compare: *balanced*,
    *open position* and *minimal movement*.
  - Edited notes are locked, and "Re-optimise" re-fingers everything around them.
- **Limits:** the optimiser picks the *most ergonomic* fingering, not necessarily the one
  the recorded player used. Timbre differences between strings (the same note on the B
  string vs the G string) are not yet used as evidence.
- **Improve:**
  - Learn the cost weights and shape priors from large tab corpora (e.g. DadaGP, 26k Guitar
    Pro files).
  - Add a timbre-based string classifier (string-specific spectral fingerprints) as an
    emission probability.

### Tuning identification
- **How:**
  - The reference-pitch offset comes from `librosa.estimate_tuning` and is reported in
    cents and as A4 in Hz.
  - The string tuning is found by **analysis-by-synthesis**: the transcription is fingered
    under every candidate tuning (standard, Eb, drop D/C#/C/B, D/C/B standard, open G/D/E/C,
    DADGAD, 7- and 8-string, bass tunings).
  - Each candidate is scored on: notes below range, playability cost, long ringing notes on
    open strings, chord roots on the lowest open string, and the key tonic, combined with a
    popularity prior.
  - A softmax over the scores gives probabilities, shown with their evidence.
  - Custom tunings can be set in the editor.
- **Limits:** parts that never use the low strings are genuinely ambiguous; the result then
  falls back mostly to the prior, at low confidence.

### Tempo, meter, downbeats
- **How:**
  - Tempo comes from the onset envelope of the drum stem: autocorrelation tempo candidates
    with octave-error resolution, then the Ellis dynamic-programming beat tracker.
  - BPM is refined by a robust linear fit over the beat times.
  - Each (meter, phase) hypothesis is scored from three cues: low-band (kick) accents,
    chord changes (weighted by how harmonic the music is), and overall accents.
  - Compound meters (6/8, 12/8) are detected from triplet-vs-duple subdivision energy.
- **Limits:**
  - Rubato and free time are not handled.
  - Large tempo changes are only partly handled.
  - Odd meters have low priors.
  - When kick and snare are symmetric, the bar phase can be off. Fix it with "Shift
    downbeat".
- **Improve:** use a neural downbeat tracker (e.g. madmom / BeatNet / Beat This!).

### Key and chords
- **How:**
  - Beat-synchronous CQT chroma of the harmonic content, matched against harmonic-weighted
    templates for 11 chord qualities × 12 roots, decoded with an HMM and Viterbi.
  - The bass stem gives the inversion (slash chords).
  - Chords are cross-validated against the guitar transcription, which raises or lowers
    their confidence.
  - The key fuses Krumhansl–Kessler correlation on chroma and on the transcribed notes,
    the third of the scale, and tonic evidence from the chord roots.
- **Limits:** power-chord riffs contain no thirds, so the mode can be ambiguous (e.g.
  E minor vs D major for E5–G5–A5–D5). Alternatives are always listed.

### Tone and effects reconstruction
- **How** (`tone/analyzer.py`): every effect is measured from interpretable evidence on the
  separated stem.

  | Effect | Evidence |
  |---|---|
  | Gain | crest factor, upper-harmonic ratio, mid-band flatness, sustain; confidence drops when these indicators disagree |
  | Compression | spread of note-attack levels, attack-to-sustain ratio, loudness range |
  | Reverb | RT60 from the regression of tail envelopes after phrase endings, L/R tail decorrelation, wet level |
  | Delay | regular decaying echo bumps in tails, plus note-domain echoes; time, feedback, mix, tempo-synced note value |
  | Tremolo / chorus / flanger / phaser / wah | periodic amplitude modulation in note windows without new attacks; slow pitch/partial modulation and stereo decorrelation; sweeping cepstral comb; sweeping spectral notches; moving resonant peak |
  | Noise gate | abrupt drops to a very low floor in a high-gain part |
  | Octaver / harmoniser | sub-octave energy; constant parallel intervals |
  | EQ / cabinet | long-term spectrum vs a reference curve, roll-off frequency |
  | Stereo | width, correlation, pan |

  - Results are tiered **high / probable / possible / no evidence**.
  - The amp is described by *character* (e.g. "classic British crunch"), never as a
    specific product.
  - `tone/recommend.py` maps the analysis onto your gear: amps and modelers (Boss Katana,
    Helix, Kemper/QC, Neural DSP, Spark…) and pedals (TS, BD-2, DS-1, RAT, Muff…), with
    knob positions and physical units.
- **Limits:**
  - Separation artefacts can add their own "reverb" or "flanging".
  - Mastering compression masks the dynamics.
  - A stereo-doubled part can resemble chorus.
  - The exact gear and settings can never be identified from a mix.
- **Improve:**
  - Train effect classifiers and parameter regressors on renders from a large plugin
    corpus (e.g. the IDMT-SMT-Audio-Effects dataset or self-rendered data).
  - Use differentiable DSP to match a parametric amp/effects chain to the stem
    ("tone matching").

### Learning from corrections
- **How:**
  - Each edit is logged per project (`corrections.jsonl`) and folded into global priors
    (`data/learning/priors.json`).
  - **String preference per pitch** and **fret-zone preference** feed the fingering cost.
  - Deletion and addition rates adjust detection thresholds: the onset threshold, minimum
    confidence, and the ghost-suppression ratio.
  - Locked edits re-finger the rest of the passage.
- **Improve:** periodically fine-tune the transcription network on accumulated
  (audio, corrected-tab) pairs. The logs already contain everything needed for that.

### Playback and editor
- **How:**
  - Each stem plays through an `HTMLAudioElement` with `preservesPitch`, so the
    time-stretch keeps the pitch while slowing down. Each element is routed through a
    Web Audio gain for mute, solo and volume.
  - The mix is the clock; stems are re-synced when they drift more than 50 ms.
  - The tab is a static SVG. The playhead and active-note highlighting are updated per
    animation frame by toggling CSS classes, with no React re-render, so playback stays
    smooth even with thousands of notes.
- **Limits:** browser time-stretching sounds grainy below about 40 % speed.

### Exports
- **GP5** (PyGuitarPro) keeps tunings, capo, tempo, time and key signature, ties, triplets,
  hammer-ons, slides, bends with points, vibrato, palm mute, let ring, dead notes, tremolo
  picking and strum direction, all tracks.
  - GPX/GP (Guitar Pro 6–8) is a closed format with no maintained open writer. Guitar Pro
    7/8 open GP5 directly.
- **MusicXML 4.0** has a notation staff and a TAB staff per part with exact
  `staff-tuning`, `<string>`/`<fret>` on every note and native technique elements.
- **MIDI** uses a full tempo map that follows the recording, or a quantised constant tempo.
  - Single-instrument exports use one channel per string, so bends stay independent;
    multi-track files use one channel per track.
  - Bend range is set to ±4 semitones through RPN.
- **PDF** has rhythm stems, chord names and technique marks; low-confidence notes are
  printed in grey.
- **Text tab** is ASCII.
- **Stems** are WAV (zip) or MP3.

---

## Measured accuracy

These numbers were measured in the development environment, where Demucs weights could not
be downloaded, so **all song-level numbers use the DSP separation fallback**. Expect better
results with Demucs. Reproduce them with `pytest` and `python scripts/evaluate.py <file>`.

| Test | Result |
|---|---|
| Single guitar riff (synthetic), note F1 (50 ms onset tolerance) | **0.955** (recall 1.00) |
| String/fret assignment vs ground truth on that riff | **64 / 64** identical |
| Open C, G, E5, F-barre voicings | all identical to the standard shapes |
| Demo song (drums + bass + distorted rhythm + delayed lead), guitar note F1 | **0.81** (rhythm recall 0.91, lead recall 1.00) |
| Demo song tempo / meter / tuning | 110.02 vs 110 BPM · 4/4 ✓ · standard ✓ |
| Demo song key | D major vs true E minor ✗ (power-chord ambiguity, see above) |
| Demo song bass F1 with DSP separation | 0.36 (DSP cannot isolate bass; use Demucs) |
| Reverb RT60 1.8 s / delay 375 ms, 45 % feedback / tremolo 5 Hz | 1.73 s / 374 ms, 45 %, dotted 1/8 / 5.01 Hz |

Synthetic material is cleaner than real recordings. Real-world accuracy depends heavily on
the mix, the amount of distortion and the separation quality.

---

## API

The full OpenAPI schema is at `/docs`. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | engines, device, limits |
| POST | `/api/projects` | multipart upload (`file`, `separator`, `tuning`, `strings`, `transcribe_bass`, `split_guitars`) |
| GET | `/api/projects` · `/api/projects/{id}` | list / status and progress |
| POST | `/api/projects/{id}/reprocess` | re-run with new options |
| GET | `/api/projects/{id}/analysis` | tempo, meter, key, chords, tuning, stems, tone |
| POST | `/api/projects/{id}/transcribe` | transcribe any stem as guitar or bass |
| GET | `/api/projects/{id}/tracks/{tid}` | full editable track |
| PATCH/POST/DELETE | `/api/projects/{id}/tracks/{tid}/notes[/{nid}]` | edit, add or delete notes (`?refinger=true` re-optimises) |
| POST | `/api/projects/{id}/tracks/{tid}/settings` | tuning, capo, time signature, downbeat, fingering profile |
| POST | `/api/projects/{id}/tracks/{tid}/reoptimize` · `/restore` | re-finger around locks / restore a rejected note |
| GET | `/api/projects/{id}/tone/{stem}/recommend?amp=&pedals=&guitar=` | gear-specific signal chain |
| GET | `/api/projects/{id}/export/{gp5\|musicxml\|midi\|midi-tracks\|pdf\|txt\|stems}?tracks=&quantized=` | exports |
| GET | `/api/learning` | what has been learned from your corrections |

---

## Testing

```bash
cd backend && . .venv/bin/activate
pytest -q                                                   # 28 tests, ~1 minute on CPU
python scripts/make_demo_song.py data/demo
python scripts/evaluate.py data/demo/demo_song.wav          # full-pipeline accuracy report
cd ../frontend && npm run typecheck && npm run build
```

The tests use synthetic audio with exact ground truth (Karplus–Strong strings, known
effects) and cover:

- transcription F1, bend and vibrato detection, echo suppression,
- fingering (riff, chord shapes, locks, impossible chords, drop D),
- quantisation (straight and triplet), tempo, meter and downbeats, tuning detection,
- tone detectors (distortion, reverb RT60, delay time, tremolo, recommendations),
- a full HTTP round trip: upload → background processing → edit → every export format,
  each re-parsed.

---

## Roadmap for transcription accuracy

Items are ordered by expected impact.

1. **Guitar-specific transcription model.** Integrate or train a model for guitar audio
   (GuitarSet, GAPS, EGDB, synthetic DadaGP renders) that predicts **string and fret
   directly** (tab-CNN / transformer). Its output then becomes an emission probability in
   the existing Viterbi optimiser, alongside the ergonomic costs.
2. **Better separation.** Use BS-RoFormer / Mel-RoFormer ensembles and a dedicated
   guitar-vs-guitar separator trained on synthetic multi-guitar mixes. Iterate separation
   using the transcription as a mask, i.e. score-informed separation.
3. **Learned fusion.** Replace the hand-weighted confidence vote with a small classifier
   trained on engine features. Calibrate confidences so that "80 %" means 80 % correct.
4. **Data-driven fingering priors.** Learn shape, position and string-transition statistics
   from tab corpora, with per-genre and per-artist style profiles. Add string-timbre
   classification as evidence.
5. **Technique classifier.** Train a frame-level model for bends, slides, legato,
   harmonics, palm mutes and tapping, with the current rule-based detector as weak
   supervision.
6. **Neural beat/downbeat tracking** for rubato and complex meters; tempo-change detection.
7. **Close the loop on corrections.** Export (audio segment, corrected notes) pairs from
   `corrections.jsonl` and fine-tune the transcription and technique models on them
   (active learning: ask the user about the lowest-confidence notes first).
8. **Tone matching.** Fit a differentiable amp + effects chain to the isolated stem.
   Produce downloadable presets for popular modelers.
9. **Evaluation suite.** Build a curated set of real songs with verified Guitar Pro tabs,
   run `scripts/evaluate.py` in CI, and track F1, fingering agreement, technique F1 and
   tuning accuracy over time.
10. **Scale-out.** Use a Celery or RQ worker pool with GPU workers, object storage for
    stems, and caching keyed by audio fingerprint.
