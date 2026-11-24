# TabForge Pocket (on-device phone version)

A single-page web app that runs the core of TabForge entirely on a phone, with no server:

```
audio file ─▶ Web Audio decode (22.05 kHz mono) ─▶ Basic Pitch in TF.js (WebGL / CPU)
          ─▶ note decoding + legato pass + pitch contours (same algorithm as the backend)
          ─▶ duplicate / overtone-ghost removal ─▶ local YIN f0 for single notes
          ─▶ bends, vibrato, hammer-ons/pull-offs ─▶ tempo + beat grid + bar lines
          ─▶ tuning detection ─▶ Viterbi string/fret optimiser (fretting-hand model)
```

What you get on the phone:

- **Playback:**
  - The tab follows the audio and highlights the notes being played; the fretboard shows them.
  - Slow-down from 50 % to 100 % keeps the pitch.
  - Tap a bar number to loop that bar; tap another bar number to extend the loop.
  - A metronome with a one-bar count-in.
- **Editing:**
  - Tap a note to hear it, play from there, move it to another string or fret, fix its pitch or technique, or delete it.
  - The fingering of nearby notes adapts around your edits.
  - Undo is available.
- **Settings:** tuning, capo, fingering style, a faint-note filter, tempo ×2/÷2 and bar-line shift.
- **Saving and sharing:**
  - Songs are saved on the device in IndexedDB.
  - "Copy tab as text" exports the tab.
  - Your string choices are learned and used for future songs.
- **Phone integration:** a screen wake lock while playing, and lock-screen media controls.

What it does **not** do, compared with the desktop app:

- There is no instrument separation, so band mixes produce extra notes. It works best on solo
  guitar or on a guitar stem exported from the desktop app.
- There is no tone or effects analysis, and no GP5, MusicXML or MIDI export.
- Vibrato detection is limited: Basic Pitch's contour is too smooth, and the local f0 tracker
  only refines isolated notes.

## Build

```bash
# 1. Basic Pitch TF.js model (Apache-2.0) from npm
npm pack @spotify/basic-pitch@1.0.1 && mkdir bp && tar -xzf spotify-basic-pitch-1.0.1.tgz -C bp
# 2. demo audio + ground truth (needs the backend venv)
backend/.venv/bin/python pocket/tools/make_pocket_demo.py /tmp/pocket-demo
# 3. demo transcription produced by the same engine in Node
npm install @tensorflow/tfjs@4.22.0           # in any scratch folder
NODE_PATH=<that folder>/node_modules node pocket/tools/node_transcribe.js \
    bp/package/model /tmp/pocket-demo/demo_22k.f32 /tmp/pocket-demo/demo.json /tmp/pocket-demo/demo.truth.json
# 4. assemble
python3 pocket/build.py bp/package/model /tmp/pocket-demo pocket/dist
```

`pocket/dist/` is a static site: `index.html`, `model.json`, `demo.mp3` and `demo.json`.
Serve it from any static host over HTTPS (for example GitHub Pages or Netlify), and it works
offline once it has loaded.

## Measured accuracy

All figures come from synthetic recordings with known answers:

| Test | Result |
|---|---|
| Solo riff, transcribed in Chromium via WebGL | note F1 **0.985**, string/fret identical to the ground truth for **32 of 32** notes, tempo 100.00 BPM (true 100) |
| Demo piece (arpeggios + lead with a bend over sustained bass), Node CPU | note F1 0.86, tempo 96.27 BPM (true 96) |
