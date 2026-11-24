/*
 * TabForge Pocket engine – the on-device subset of the TabForge pipeline.
 *
 *  audio (mono 22.05 kHz) -> Basic Pitch (TF.js) posteriorgrams -> note decoding
 *  (onset peaks + legato pass + pitch contours) -> fragment merge / overtone-ghost
 *  removal -> bends, vibrato, hammer-ons/pull-offs -> tempo + beat grid -> tuning
 *  detection -> Viterbi string/fret optimisation with a fretting-hand model.
 *
 * It mirrors backend/app/transcription/basic_pitch_onnx.py, ensemble.py (single engine),
 * techniques.py and guitar/fingering.py. Works in browsers and in Node (for tests).
 */
(function (root) {
  "use strict";

  const SR = 22050;
  const FFT_HOP = 256;
  const AUDIO_N = SR * 2 - FFT_HOP; // 43844 samples per window
  const OVERLAP_FRAMES = 30;
  const OVERLAP = OVERLAP_FRAMES * FFT_HOP;
  const HOP_SIZE = AUDIO_N - OVERLAP;
  const FPS = SR / FFT_HOP;
  const MIDI_OFFSET = 21;
  const CBPS = 3;
  const P = 88;
  const PC = 264;

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const sleep = () => new Promise((r) => setTimeout(r, 0));

  // ------------------------------------------------------------------ tunings
  const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const midiName = (m) => `${NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`;
  function nameToMidi(n) {
    const m = /^([A-G])([#b]?)(-?\d)$/.exec(n);
    const base = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }[m[1]];
    return (Number(m[3]) + 1) * 12 + base + (m[2] === "#" ? 1 : m[2] === "b" ? -1 : 0);
  }
  const T = (id, name, notes, prior) => ({ id, name, notes: notes.split(" ").map(nameToMidi), prior });
  const TUNINGS = [
    T("standard", "Standard (E A D G B E)", "E2 A2 D3 G3 B3 E4", 0.55),
    T("eb_standard", "Half step down (Eb)", "Eb2 Ab2 Db3 Gb3 Bb3 Eb4", 0.1),
    T("drop_d", "Drop D", "D2 A2 D3 G3 B3 E4", 0.1),
    T("d_standard", "D standard", "D2 G2 C3 F3 A3 D4", 0.04),
    T("drop_c", "Drop C", "C2 G2 C3 F3 A3 D4", 0.04),
    T("open_g", "Open G", "D2 G2 D3 G3 B3 D4", 0.02),
    T("open_d", "Open D", "D2 A2 D3 F#3 A3 D4", 0.015),
    T("dadgad", "DADGAD", "D2 A2 D3 G3 A3 D4", 0.015),
    T("7_standard", "7-string (B E A D G B E)", "B1 E2 A2 D3 G3 B3 E4", 0.03),
  ];
  // open-string pitch for string number s (1 = highest)
  const openMidi = (tun, s) => tun.notes[tun.notes.length - s];

  // ------------------------------------------------------------------ inference
  async function runModel(tf, model, y, onProgress) {
    const nOrig = y.length;
    const padded = new Float32Array(OVERLAP / 2 + nOrig);
    padded.set(y, OVERLAP / 2);
    const starts = [];
    for (let s = 0; s < padded.length; s += HOP_SIZE) starts.push(s);
    const nT = Math.ceil(nOrig / FFT_HOP) + 1;
    const note = new Float32Array(nT * P);
    const onset = new Float32Array(nT * P);
    const contour = new Float32Array(nT * PC);
    const half = OVERLAP_FRAMES / 2;
    const batch = 4;
    for (let b0 = 0; b0 < starts.length; b0 += batch) {
      const cs = starts.slice(b0, b0 + batch);
      const x = new Float32Array(cs.length * AUDIO_N);
      cs.forEach((s, i) => x.set(padded.subarray(s, s + AUDIO_N), i * AUDIO_N));
      const input = tf.tensor3d(x, [cs.length, AUDIO_N, 1]);
      const outs = model.execute(input, ["Identity_1:0", "Identity_2:0", "Identity:0"]);
      const [nA, oA, cA] = await Promise.all(outs.map((t) => t.data()));
      input.dispose();
      outs.forEach((t) => t.dispose());
      const Tw = nA.length / cs.length / P;
      const kept = Tw - 2 * half;
      cs.forEach((s, i) => {
        for (let k = 0; k < kept; k++) {
          const idx = Math.round((s + k * FFT_HOP) / FFT_HOP);
          if (idx >= nT) break;
          const src = i * Tw + k + half;
          for (let p = 0; p < P; p++) {
            const a = idx * P + p;
            const b = src * P + p;
            if (nA[b] > note[a]) note[a] = nA[b];
            if (oA[b] > onset[a]) onset[a] = oA[b];
          }
          for (let p = 0; p < PC; p++) {
            const a = idx * PC + p;
            const b = src * PC + p;
            if (cA[b] > contour[a]) contour[a] = cA[b];
          }
        }
      });
      if (onProgress) onProgress(Math.min(1, (b0 + batch) / starts.length));
      await sleep();
    }
    return { note, onset, contour, T: nT };
  }

  // ------------------------------------------------------------------ note decoding
  function median5(a) {
    if (a.length < 5) return a;
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      const w = [];
      for (let k = -2; k <= 2; k++) w.push(a[clamp(i + k, 0, a.length - 1)]);
      w.sort((x, y) => x - y);
      out[i] = w[2];
    }
    return out;
  }

  function trackContour(contour, pitch, f0, f1) {
    const center = (pitch - MIDI_OFFSET) * CBPS + 1;
    const lo = center - 4;
    const hi = center + 8;
    let cur = center;
    const out = new Float32Array(Math.max(1, f1 - f0));
    for (let k = 0; k < out.length; k++) {
      const f = f0 + k;
      const a = Math.max(lo, cur - 2, 0);
      const b = Math.min(hi, cur + 2, PC - 1);
      let best = -1;
      let j = a;
      for (let q = a; q <= b; q++) {
        const v = contour[f * PC + q];
        if (v > best) { best = v; j = q; }
      }
      if (best < 0.05) {
        out[k] = (cur - center) / CBPS;
        continue;
      }
      let frac = 0;
      if (j > a && j < b) {
        const y0 = contour[f * PC + j - 1], y1 = contour[f * PC + j], y2 = contour[f * PC + j + 1];
        const d = y0 - 2 * y1 + y2;
        frac = d !== 0 ? (0.5 * (y0 - y2)) / d : 0;
      }
      cur = j;
      out[k] = (j + frac - center) / CBPS;
    }
    return median5(out);
  }

  function decodeNotes(post, opts = {}) {
    const onThr = opts.onsetThreshold ?? 0.5;
    const frThr = opts.frameThreshold ?? 0.3;
    const minLen = Math.max(2, Math.round((opts.minNoteS ?? 0.055) * FPS));
    const tol = 8;
    const minMidi = opts.minMidi ?? 26;
    const maxMidi = opts.maxMidi ?? 100;
    const { note, onset, contour, T: nT } = post;
    // inferred onsets from sharp rises of the note posterior
    const inferred = new Float32Array(nT * P);
    let oMax = 0;
    let dMax = 1e-9;
    const diff = new Float32Array(nT * P);
    for (let t = 1; t < nT; t++) {
      for (let p = 0; p < P; p++) {
        let d = Math.max(0, note[t * P + p] - note[(t - 1) * P + p]);
        if (t > 1) d = Math.max(d, note[t * P + p] - note[(t - 2) * P + p]);
        diff[t * P + p] = d;
        if (d > dMax) dMax = d;
      }
    }
    for (let i = 0; i < onset.length; i++) if (onset[i] > oMax) oMax = onset[i];
    const scale = oMax / dMax;
    for (let i = 0; i < inferred.length; i++) inferred[i] = Math.max(onset[i], diff[i] * scale);
    const peaks = [];
    for (let t = 0; t < nT; t++) {
      for (let p = 0; p < P; p++) {
        const v = inferred[t * P + p];
        if (v <= onThr) continue;
        const prev = t > 0 ? inferred[(t - 1) * P + p] : -1;
        const next = t < nT - 1 ? inferred[(t + 1) * P + p] : -1;
        if (v >= prev && v >= next) peaks.push([t, p]);
      }
    }
    peaks.sort((a, b) => b[0] - a[0]);
    const remaining = new Float32Array(note);
    const notes = [];
    const mk = (t0, t1, p, legato) => {
      let amp = 0;
      for (let t = t0; t < t1; t++) amp += note[t * P + p];
      amp /= Math.max(1, t1 - t0);
      notes.push({ start: t0 / FPS, end: t1 / FPS, pitch: p + MIDI_OFFSET, amplitude: amp,
        onsetStrength: Math.min(1, inferred[t0 * P + p]), noOnset: legato, f0: t0, f1: t1 });
    };
    for (const [t, p] of peaks) {
      const midi = p + MIDI_OFFSET;
      if (midi < minMidi || midi > maxMidi || t >= nT - 1) continue;
      let i = t + 1;
      let k = 0;
      while (i < nT - 1 && k < tol) {
        if (remaining[i * P + p] < frThr) k++;
        else k = 0;
        i++;
      }
      i -= k;
      if (i - t < minLen) continue;
      for (let q = t; q < i; q++) for (let pp = Math.max(0, p - 1); pp <= Math.min(P - 1, p + 1); pp++) remaining[q * P + pp] = 0;
      mk(t, i, p, false);
    }
    // legato pass: strong sustained activation without an onset peak
    const thr = Math.max(frThr + 0.15, 0.45);
    // values in `remaining` are only ever zeroed, so visiting cells in descending order of
    // their value is exactly "repeatedly take the arg-max", in linear time
    const cand = [];
    for (let i = 0; i < remaining.length; i++) if (remaining[i] >= thr) cand.push(i);
    cand.sort((a, b) => remaining[b] - remaining[a]);
    for (const at of cand) {
      if (remaining[at] < thr) continue;
      const t = Math.floor(at / P);
      const p = at % P;
      let i = t;
      while (i < nT - 1 && remaining[(i + 1) * P + p] > frThr) i++;
      let j = t;
      while (j > 0 && remaining[(j - 1) * P + p] > frThr) j--;
      for (let q = j; q <= i; q++) for (let pp = Math.max(0, p - 1); pp <= Math.min(P - 1, p + 1); pp++) remaining[q * P + pp] = 0;
      const midi = p + MIDI_OFFSET;
      if (i + 1 - j < minLen || midi < minMidi || midi > maxMidi) continue;
      mk(j, i + 1, p, true);
    }
    for (const n of notes) n.contour = trackContour(contour, n.pitch, n.f0, Math.max(n.f1, n.f0 + 1));
    notes.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
    return notes;
  }

  // ------------------------------------------------------------------ clean-up
  function mergeFragments(notes) {
    const byPitch = new Map();
    for (const n of notes) {
      if (!byPitch.has(n.pitch)) byPitch.set(n.pitch, []);
      byPitch.get(n.pitch).push(n);
    }
    const out = [];
    for (const lst of byPitch.values()) {
      lst.sort((a, b) => a.start - b.start);
      const merged = [];
      for (const n of lst) {
        const prev = merged[merged.length - 1];
        // the same pitch detected twice at (almost) the same moment: keep one note
        if (prev && n.start - prev.start < 0.06) {
          prev.end = Math.max(prev.end, n.end);
          prev.f1 = Math.max(prev.f1, n.f1);
          prev.amplitude = Math.max(prev.amplitude, n.amplitude);
          continue;
        }
        if (prev && n.start - prev.end < 0.04 && (n.onsetStrength < 0.55 || n.noOnset)) {
          const pad = Math.max(0, n.f0 - prev.f1);
          const ext = new Float32Array(prev.contour.length + pad + n.contour.length);
          ext.set(prev.contour);
          ext.fill(prev.contour[prev.contour.length - 1] || 0, prev.contour.length, prev.contour.length + pad);
          ext.set(n.contour, prev.contour.length + pad);
          prev.contour = ext;
          prev.end = Math.max(prev.end, n.end);
          prev.f1 = Math.max(prev.f1, n.f1);
          prev.amplitude = Math.max(prev.amplitude, n.amplitude);
          continue;
        }
        merged.push(n);
      }
      out.push(...merged);
    }
    return out.sort((a, b) => a.start - b.start || a.pitch - b.pitch);
  }

  const GHOST = new Set([12, 19, 24, 28, 31]);
  const overlap = (a, b) => Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start));

  function suppressGhosts(notes, ratio = 0.66) {
    const keep = [];
    const rejected = [];
    for (const b of notes) {
      let ghost = false;
      for (const a of notes) {
        if (a === b || a.pitch >= b.pitch || !GHOST.has(b.pitch - a.pitch)) continue;
        if (Math.abs(a.start - b.start) > 0.06 || overlap(a, b) < 0.6 * (b.end - b.start)) continue;
        if (b.amplitude < ratio * a.amplitude) { ghost = true; break; }
        if (b.amplitude < 0.85 * a.amplitude) b.possibleHarmonic = true;
      }
      if (ghost) rejected.push(b);
      else keep.push(b);
    }
    return { keep, rejected };
  }

  // ------------------------------------------------------------------ techniques
  function detectBend(n) {
    const c = n.contour;
    if (!c || c.length < 6) return null;
    const hop = 1 / FPS;
    const skip = Math.min(c.length - 1, Math.floor(0.025 / hop));
    let peak = -9;
    let kPeak = skip;
    let hold = 0;
    let trough = 9;
    for (let k = skip; k < c.length; k++) {
      if (c[k] > peak) { peak = c[k]; kPeak = k; }
      if (c[k] > 0.35) hold += hop;
      if (c[k] < trough) trough = c[k];
    }
    const startLevel = c.slice(0, skip + 1).reduce((s, v) => s + v, 0) / (skip + 1);
    const q = (st) => clamp(Math.round(st * 2) / 2, 0.5, 3);
    if (peak >= 0.4 && hold >= 0.04 && startLevel <= 0.35) {
      const tail = c.slice(kPeak);
      const released = tail.length > 3 && Math.min(...tail.slice(Math.floor(tail.length / 2))) < 0.25 && peak - tail[tail.length - 1] > 0.35;
      return { type: released ? "bend_release" : "bend", semitones: q(peak), confidence: clamp(0.45 + 0.4 * Math.min(1, hold / 0.12), 0, 0.9) };
    }
    if (trough <= -0.8 && startLevel > -0.25) return { type: "prebend_release", semitones: q(-trough), confidence: 0.4 };
    return null;
  }

  function detectVibrato(n) {
    const c = n.f0contour || n.contour;
    const hop = 1 / FPS;
    const skip = Math.floor(0.08 / hop);
    if (!c || (c.length - skip) * hop < 0.25) return null;
    const seg = Array.from(c.slice(skip));
    // remove a quadratic trend
    const N = seg.length;
    const xs = seg.map((_, i) => (2 * i) / (N - 1) - 1);
    const fit = polyfit2(xs, seg);
    const d = seg.map((v, i) => v - (fit[0] * xs[i] * xs[i] + fit[1] * xs[i] + fit[2]));
    const ac = autocorr(d);
    if (ac[0] <= 1e-9) return null;
    const lo = Math.floor(FPS / 9);
    const hi = Math.min(ac.length - 2, Math.floor(FPS / 3.5));
    if (hi <= lo + 1) return null;
    let k = lo;
    for (let i = lo; i <= hi; i++) if (ac[i] > ac[k]) k = i;
    if (k === lo || k === hi) return null;
    const strength = ac[k] / ac[0];
    const sorted = [...d].sort((a, b) => a - b);
    const depth = sorted[Math.floor(0.95 * (N - 1))] - sorted[Math.floor(0.05 * (N - 1))];
    if (strength < 0.35 || depth < 0.12) return null;
    return { type: "vibrato", rateHz: +(FPS / k).toFixed(2), depth: +depth.toFixed(2), confidence: clamp(0.3 + strength * 0.6, 0, 0.9) };
  }

  function polyfit2(x, y) {
    // least squares for y = a x^2 + b x + c
    let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i], x2 = xi * xi;
      s0 += 1; s1 += xi; s2 += x2; s3 += x2 * xi; s4 += x2 * x2;
      t0 += y[i]; t1 += xi * y[i]; t2 += x2 * y[i];
    }
    const M = [[s4, s3, s2, t2], [s3, s2, s1, t1], [s2, s1, s0, t0]];
    for (let i = 0; i < 3; i++) {
      let piv = i;
      for (let r = i + 1; r < 3; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
      [M[i], M[piv]] = [M[piv], M[i]];
      if (Math.abs(M[i][i]) < 1e-12) return [0, 0, t0 / Math.max(1, s0)];
      for (let r = 0; r < 3; r++) {
        if (r === i) continue;
        const f = M[r][i] / M[i][i];
        for (let c = i; c < 4; c++) M[r][c] -= f * M[i][c];
      }
    }
    return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
  }

  function autocorr(d) {
    const N = d.length;
    const m = d.reduce((s, v) => s + v, 0) / N;
    const x = d.map((v) => v - m);
    const out = new Float64Array(N);
    for (let lag = 0; lag < N; lag++) {
      let s = 0;
      for (let i = 0; i + lag < N; i++) s += x[i] * x[i + lag];
      out[lag] = s;
    }
    return out;
  }

  /** Re-join a bend that the decoder split into a semitone staircase (E->F->F# without attacks). */
  function mergeBendChains(notes) {
    const order = [...notes].sort((a, b) => a.start - b.start);
    const used = new Set();
    const out = [];
    for (let i = 0; i < order.length; i++) {
      const a = order[i];
      if (used.has(a)) continue;
      const chain = [a];
      let peak = a.pitch;
      let rising = true;
      for (;;) {
        const prev = chain[chain.length - 1];
        let cand = null;
        for (let j = i + 1; j < order.length; j++) {
          const b = order[j];
          if (b.start > prev.end + 0.12) break;
          if (used.has(b) || chain.includes(b) || b.start < prev.end - 0.06) continue;
          const step = b.pitch - prev.pitch;
          if (Math.abs(step) >= 1 && Math.abs(step) <= 2) { cand = b; break; }
        }
        if (!cand) break;
        const step = cand.pitch - prev.pitch;
        if (!(cand.noOnset || cand.onsetStrength < 0.5)) break;
        if (rising && step > 0 && cand.pitch - a.pitch <= 3) peak = cand.pitch;
        else if (step < 0 && cand.pitch >= a.pitch) rising = false;
        else break;
        chain.push(cand);
      }
      const peakNote = chain.find((c) => c.pitch === peak);
      const rise = peakNote ? peakNote.start - a.start : 0;
      if (chain.length >= 2 && peak > a.pitch && rise >= 0.12) {
        const parts = [];
        for (const c of chain) {
          const off = c.pitch - a.pitch;
          parts.push(Array.from(c.contour || []).map((v) => v + off));
        }
        a.contour = Float32Array.from(parts.flat());
        a.end = chain[chain.length - 1].end;
        a.amplitude = Math.max(...chain.map((c) => c.amplitude));
        chain.slice(1).forEach((c) => used.add(c));
      }
      out.push(a);
    }
    return out;
  }

  // ------------------------------------------------------------------ monophonic f0 refinement
  /**
   * Second pitch engine for single-note passages (the role CREPE/pYIN play on the desktop):
   * a normalised difference function (YIN family) searched only within +-7% of the note's
   * expected period, so it is cheap and cannot jump octaves. It yields a fine pitch contour
   * (for vibrato/bends, which Basic Pitch's 1/3-semitone contour smooths away) and extends
   * notes whose posterior faded during vibrato.
   */
  function refineMonophonic(notes, y) {
    if (!y) return 0;
    const order = [...notes].sort((a, b) => a.start - b.start);
    let refined = 0;
    for (let i = 0; i < order.length; i++) {
      const n = order[i];
      const others = order.filter((m) => m !== n && m.start < n.end + 1.5 && m.end > n.start - 0.02);
      if (others.some((m) => Math.abs(m.start - n.start) < 0.05)) { n.refineSkip = "chord"; continue; }
      const nextStart = Math.min(n.start + 3.0, ...order.slice(i + 1).filter((m) => m.start > n.start + 0.05).map((m) => m.start));
      if (nextStart - n.start < 0.25) { n.refineSkip = "short"; continue; }
      // a note ringing underneath is fine if it is well below (the period search is local)
      const under = order.filter((m) => m !== n && m.start < n.start && m.end > n.start + 0.1);
      if (under.some((m) => m.pitch > n.pitch - 7)) { n.refineSkip = "under"; continue; }
      const thr = under.length ? 0.65 : 0.4;
      const f0 = 440 * Math.pow(2, (n.pitch - 69) / 12);
      const T0 = SR / f0;
      // about three periods: long enough to measure, short enough to follow vibrato
      const W = Math.round(clamp(3.2 * T0, 384, 1024) / 2) * 2;
      const f0Frame = Math.round(n.start * FPS) + 2;
      // keep the analysis window (W samples + lag) clear of the next attack
      const fEnd = Math.floor((nextStart - (W * 1.1) / SR) * FPS);
      const contour = [];
      let lastGood = f0Frame;
      let prevTau = T0;
      // search periods for pitches from -1.2 to +2.6 semitones around the note
      const tauLo = Math.max(3, Math.floor(T0 / Math.pow(2, 2.6 / 12)));
      const tauHi = Math.ceil(T0 * Math.pow(2, 1.2 / 12));
      const ds = new Float64Array(tauHi + 2);
      for (let f = f0Frame; f < fEnd; f++) {
        const c = f * FFT_HOP;
        if (c + W + tauHi + 2 >= y.length) break;
        for (let tau = tauLo - 1; tau <= tauHi + 1; tau++) {
          let num = 0, e1 = 0, e2 = 0;
          for (let j = 0; j < W; j += 2) {
            const a = y[c + j], b = y[c + j + tau];
            num += (a - b) * (a - b);
            e1 += a * a;
            e2 += b * b;
          }
          ds[tau] = num / (e1 + e2 + 1e-12);
        }
        // the difference function is biased towards short lags, so take a genuine local
        // minimum (never the edge of the search range), the one closest to the previous period
        let bt = 0;
        for (let tau = tauLo; tau <= tauHi; tau++) {
          if (ds[tau] < ds[tau - 1] && ds[tau] <= ds[tau + 1] && ds[tau] < thr) {
            if (!bt || Math.abs(tau - prevTau) < Math.abs(bt - prevTau)) bt = tau;
          }
        }
        if (!bt) {
          contour.push(NaN);
          continue;
        }
        const a = ds[bt - 1], b = ds[bt], cc = ds[bt + 1];
        const den = a - 2 * b + cc;
        const tauF = bt + (den !== 0 ? (0.5 * (a - cc)) / den : 0);
        prevTau = tauF;
        contour.push(12 * Math.log2(T0 / tauF));
        lastGood = f;
      }
      const good = contour.filter((v) => !Number.isNaN(v)).length;
      if (good < 0.6 * contour.length || contour.length < 8) {
        n.refineSkip = `periodic ${good}/${contour.length}`;
        continue;
      }
      // fill gaps; frames start 2 after the onset. Used for vibrato (Basic Pitch's own
      // contour stays the reference for bends, where it is the more reliable of the two).
      let last = 0;
      const filled = contour.map((v) => (Number.isNaN(v) ? last : (last = v)));
      n.f0contour = Float32Array.from([filled[0], filled[0], ...filled]);
      const newEnd = (lastGood + 1) / FPS;
      if (newEnd > n.end + 0.05) n.end = Math.min(newEnd, nextStart);
      refined++;
    }
    return refined;
  }

  function detectTechniques(notes) {
    for (const n of notes) {
      n.techniques = [];
      const b = detectBend(n);
      if (b) n.techniques.push(b);
      const v = detectVibrato(n);
      if (v) n.techniques.push(v);
    }
    const order = [...notes].sort((a, b) => a.start - b.start);
    for (let i = 1; i < order.length; i++) {
      const a = order[i - 1];
      const b = order[i];
      const gap = b.start - a.end;
      if (gap < -0.05 || gap > 0.045 || a.pitch === b.pitch) continue;
      const iv = b.pitch - a.pitch;
      if (Math.abs(iv) > 7) continue;
      // something else starting at the same time means it is not a single-line legato
      if ((order[i + 1] && Math.abs(order[i + 1].start - b.start) < 0.03) || Math.abs(order[i - 1].start - (order[i - 2] || { start: -9 }).start) < 0.03) continue;
      if (b.noOnset || b.onsetStrength < 0.45) {
        b.techniques.push({ type: iv > 0 ? "hammer_on" : "pull_off", fromPitch: a.pitch, confidence: 0.5, provisional: true });
      }
    }
  }

  // ------------------------------------------------------------------ tempo & beats
  function onsetEnvelope(post) {
    const { note, onset, T: nT } = post;
    const env = new Float32Array(nT);
    for (let t = 1; t < nT; t++) {
      let s = 0;
      for (let p = 0; p < P; p++) {
        s += onset[t * P + p];
        const d = note[t * P + p] - note[(t - 1) * P + p];
        if (d > 0) s += 0.5 * d;
      }
      env[t] = s;
    }
    // light smoothing and mean removal
    const out = new Float32Array(nT);
    for (let t = 0; t < nT; t++) out[t] = (env[t] + 0.5 * (env[t - 1] || 0) + 0.5 * (env[t + 1] || 0)) / 2;
    return out;
  }

  function estimateTempo(env) {
    const N = env.length;
    const mean = env.reduce((s, v) => s + v, 0) / Math.max(1, N);
    const x = Array.from(env, (v) => v - mean);
    const minLag = Math.floor((60 / 220) * FPS);
    const maxLag = Math.min(N - 1, Math.ceil((60 / 45) * FPS));
    const ac = new Float64Array(maxLag + 2);
    for (let lag = minLag - 1; lag <= maxLag + 1 && lag < N; lag++) {
      let s = 0;
      for (let i = 0; i + lag < N; i++) s += x[i] * x[i + lag];
      ac[lag] = s;
    }
    let best = null;
    for (let lag = minLag; lag <= maxLag; lag++) {
      if (!(ac[lag] > ac[lag - 1] && ac[lag] >= ac[lag + 1]) || ac[lag] <= 0) continue;
      const a = ac[lag - 1], b = ac[lag], c = ac[lag + 1];
      const d = a - 2 * b + c;
      const l = lag + (d !== 0 ? (0.5 * (a - c)) / d : 0);
      const bpm = (60 * FPS) / l;
      const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 112) / 0.55, 2));
      const score = b * prior;
      if (!best || score > best.score) best = { bpm, score, lag: l };
    }
    return best ? best.bpm : 120;
  }

  /** Ellis-style dynamic-programming beat tracker on the onset envelope. */
  function trackBeats(env, bpm, duration) {
    const N = env.length;
    const period = (60 / bpm) * FPS;
    const tight = 100;
    const score = new Float64Array(N);
    const back = new Int32Array(N).fill(-1);
    const mx = Math.max(...env) || 1;
    for (let t = 0; t < N; t++) {
      const local = env[t] / mx;
      let best = 0;
      let arg = -1;
      const lo = Math.max(0, Math.round(t - 2 * period));
      const hi = Math.round(t - period / 2);
      for (let p = lo; p <= hi; p++) {
        const pen = -tight * Math.pow(Math.log((t - p) / period), 2);
        const v = score[p] + pen;
        if (v > best || arg < 0) { best = v; arg = p; }
      }
      score[t] = local + (arg >= 0 ? best : 0);
      back[t] = arg;
    }
    // best final beat within the last period
    let end = N - 1;
    for (let t = Math.max(0, Math.round(N - period)); t < N; t++) if (score[t] > score[end]) end = t;
    const beats = [];
    for (let t = end; t >= 0; t = back[t]) {
      beats.push(t / FPS);
      if (back[t] < 0) break;
    }
    beats.reverse();
    // refine bpm with a linear fit of beat time vs index; for a steady performance (small
    // residuals) the fitted constant-tempo grid is more accurate than individual DP beats,
    // which are frame-quantised and wobble at the edges of the recording
    let refined = bpm;
    let grid = beats;
    if (beats.length > 4) {
      const n = beats.length;
      const mi = (n - 1) / 2;
      const mt = beats.reduce((s, v) => s + v, 0) / n;
      let num = 0, den = 0;
      beats.forEach((b, i) => { num += (i - mi) * (b - mt); den += (i - mi) * (i - mi); });
      const slope = num / den;
      if (slope > 0) {
        refined = 60 / slope;
        const fitted = beats.map((_, i) => mt + (i - mi) * slope);
        const resid = Math.sqrt(beats.reduce((s, b, i) => s + (b - fitted[i]) ** 2, 0) / n);
        if (resid < 0.03) grid = fitted;
      }
    }
    return { beats: extendGrid(grid, duration, refined), bpm: refined };
  }

  function extendGrid(beats, duration, bpm) {
    const per = 60 / bpm;
    if (beats.length < 2) {
      const g = [];
      for (let t = 0; t < duration + per; t += per) g.push(t);
      return g;
    }
    const out = [...beats];
    while (out[0] - per > -per * 0.999) out.unshift(out[0] - per);
    while (out[out.length - 1] < duration + per) out.push(out[out.length - 1] + per);
    return out;
  }

  function chooseDownbeat(beats, notes, beatsPerBar) {
    // bass notes and chord changes mark bar lines
    const score = new Float64Array(beatsPerBar);
    const groups = new Map();
    for (const n of notes) {
      const i = nearestIdx(beats, n.start);
      if (Math.abs(beats[i] - n.start) > 0.07) continue;
      if (!groups.has(i)) groups.set(i, []);
      groups.get(i).push(n);
    }
    let prevSet = "";
    const idxs = [...groups.keys()].sort((a, b) => a - b);
    for (const i of idxs) {
      const g = groups.get(i);
      const low = Math.min(...g.map((n) => n.pitch));
      const set = [...new Set(g.map((n) => n.pitch % 12))].sort().join(",");
      let s = g.reduce((a, n) => a + (n.end - n.start) * n.amplitude, 0) * (low < 52 ? 1.5 : 1);
      if (set !== prevSet) s += 0.6;
      prevSet = set;
      score[((i % beatsPerBar) + beatsPerBar) % beatsPerBar] += s;
    }
    let best = 0;
    for (let p = 1; p < beatsPerBar; p++) if (score[p] > score[best]) best = p;
    return best;
  }

  function nearestIdx(arr, t) {
    let lo = 0, hi = arr.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    if (lo > 0 && Math.abs(arr[lo - 1] - t) < Math.abs(arr[lo] - t)) return lo - 1;
    return lo;
  }

  function buildMeasures(beats, phase, beatsPerBar, duration) {
    const out = [];
    for (let i = phase; i + beatsPerBar < beats.length; i += beatsPerBar) {
      const start = beats[i];
      const end = beats[i + beatsPerBar];
      if (end <= 0) continue;
      if (start > duration) break;
      out.push({ start, end, beats: beats.slice(i, i + beatsPerBar + 1) });
    }
    // make sure something precedes the first note
    if (phase > 0 && out.length && out[0].start > 0) {
      const per = beats[1] - beats[0];
      const start = out[0].start - beatsPerBar * per;
      const bs = [];
      for (let k = 0; k <= beatsPerBar; k++) bs.push(start + k * per);
      out.unshift({ start, end: out[0].start, beats: bs });
    }
    return out;
  }

  // ------------------------------------------------------------------ fingering
  const PROFILES = {
    balanced: { height: 0.05, shift: 1.0, shiftFixed: 0.6, span: 1.2, open: 0.25, jump: 0.25, gap: 0.7, legato: 0.8, sustain: 1.0, repeat: 0.4, high: 0.15 },
    open_position: { height: 0.22, shift: 1.0, shiftFixed: 0.6, span: 1.2, open: 0.6, jump: 0.2, gap: 0.7, legato: 0.8, sustain: 1.0, repeat: 0.4, high: 0.3 },
    minimal_movement: { height: 0.02, shift: 2.6, shiftFixed: 1.4, span: 1.0, open: 0.1, jump: 0.15, gap: 0.6, legato: 0.8, sustain: 0.8, repeat: 0.6, high: 0.08 },
  };
  const OPEN_SHAPES = [
    [null, 3, 2, 0, 1, 0], [3, 2, 0, 0, 0, 3], [3, 2, 0, 0, 3, 3], [null, null, 0, 2, 3, 2], [0, 2, 2, 1, 0, 0],
    [null, 0, 2, 2, 2, 0], [0, 2, 2, 0, 0, 0], [null, 0, 2, 2, 1, 0], [null, null, 0, 2, 3, 1], [0, 2, 0, 1, 0, 0],
    [null, 0, 2, 0, 2, 0], [null, 0, 2, 2, 0, 0], [0, 0, 2, 2, 0, 0],
  ];

  class Optimizer {
    constructor(tuning, opts = {}) {
      this.t = tuning;
      this.n = tuning.notes.length;
      this.capo = opts.capo || 0;
      this.maxFret = (opts.maxFret || 22) - this.capo;
      this.w = PROFILES[opts.profile || "balanced"] || PROFILES.balanced;
      this.prior = opts.prior || null; // {pitch: {string: count}}
    }

    open(s) { return openMidi(this.t, s) + this.capo; }

    positions(pitch) {
      const out = [];
      for (let s = 1; s <= this.n; s++) {
        const f = pitch - this.open(s);
        if (f >= 0 && f <= this.maxFret) out.push([s, f]);
      }
      return out;
    }

    fingerCount(pos) {
      const fr = pos.filter(([, f]) => f > 0);
      if (!fr.length) return [0, false];
      const minf = Math.min(...fr.map(([, f]) => f));
      const atMin = fr.filter(([, f]) => f === minf).map(([s]) => s);
      let barre = false;
      if (atMin.length >= 2) {
        const lo = Math.min(...atMin), hi = Math.max(...atMin);
        barre = pos.every(([s, f]) => !(s > lo && s < hi && f === 0));
      }
      return [fr.length - (barre ? atMin.length - 1 : 0), barre];
    }

    intrinsic(pos, notes) {
      const w = this.w;
      const fr = pos.map(([, f]) => f).filter((f) => f > 0);
      const minf = fr.length ? Math.min(...fr) : -1;
      const maxf = fr.length ? Math.max(...fr) : -1;
      const span = fr.length ? maxf - minf : 0;
      const maxSpan = minf >= 9 ? 6 : 5;
      const [fingers, barre] = this.fingerCount(pos);
      const feasible = span <= maxSpan && fingers <= 4;
      let cost = span > 3 ? w.span * (span - 3) ** 2 : 0;
      pos.forEach(([s, f], i) => {
        cost += w.height * f;
        if (f > 17) cost += w.high * (f - 17);
        if (f === 0) {
          cost -= w.open;
          if ((notes[i].techniques || []).some((t) => /bend|vibrato/.test(t.type))) cost += 3;
        }
        const pr = this.prior && this.prior[notes[i].pitch];
        if (pr) {
          const tot = Object.values(pr).reduce((a, b) => a + b, 0) + 2;
          const p = ((pr[s] || 0) + 2 / this.n) / tot;
          cost += -0.5 * Math.log(p * this.n);
        }
      });
      if (pos.length >= 3) {
        const ss = pos.map(([s]) => s).sort((a, b) => a - b);
        cost += w.gap * (ss[ss.length - 1] - ss[0] + 1 - ss.length);
        cost += 0.15 * Math.max(0, fingers - 3) + (barre ? 0.2 : 0);
        if (this.n === 6) {
          const fm = Object.fromEntries(pos);
          for (const shape of OPEN_SHAPES) {
            let ok = true, cnt = 0;
            for (let i = 0; i < 6; i++) {
              const s = 6 - i;
              if (shape[i] === null) { if (s in fm) { ok = false; break; } } else if (s in fm) { if (fm[s] !== shape[i]) { ok = false; break; } cnt++; }
            }
            if (ok && cnt === pos.length) { cost -= 0.6; break; }
          }
        }
      } else if (pos.length === 2) cost += 0.1 * Math.max(0, Math.abs(pos[0][0] - pos[1][0]) - 2);
      if (!feasible) cost += 25 + 4 * Math.max(0, fingers - 4) + 3 * Math.max(0, span - maxSpan);
      return { pos, cost, minf, maxf, feasible };
    }

    candidates(notes) {
      const opts = notes.map((n) => (n.locked && n.string ? [[n.string, n.fret]] : this.positions(n.pitch)));
      if (opts.some((o) => !o.length) || notes.length > this.n) return [];
      const order = opts.map((_, i) => i).sort((a, b) => opts[a].length - opts[b].length);
      const chosen = new Array(notes.length);
      const used = new Set();
      const cands = [];
      let limit = 3000;
      const dfs = (k) => {
        if (limit <= 0) return;
        if (k === order.length) { cands.push(this.intrinsic(chosen.slice(), notes)); limit--; return; }
        const i = order[k];
        for (const [s, f] of opts[i]) {
          if (used.has(s)) continue;
          const fr = chosen.filter((x, j) => x && j !== i && x[1] > 0).map((x) => x[1]);
          if (f > 0) fr.push(f);
          if (fr.length && Math.max(...fr) - Math.min(...fr) > 7) continue;
          chosen[i] = [s, f];
          used.add(s);
          dfs(k + 1);
          used.delete(s);
          chosen[i] = undefined;
        }
      };
      dfs(0);
      cands.sort((a, b) => a.cost - b.cost);
      return cands.slice(0, 24);
    }

    reach(c) {
      if (c.minf < 0) return null;
      let lo = Math.max(1, c.maxf - 3);
      let hi = c.minf;
      if (lo > hi) lo = hi = c.minf;
      return [lo, hi];
    }

    window(h, c) {
      const r = this.reach(c);
      if (!r) return [h, 0];
      if (!h) return [r, 0];
      const lo = Math.max(h[0], r[0]), hi = Math.min(h[1], r[1]);
      if (lo <= hi) return [[lo, hi], 0];
      return [r, r[0] > h[1] ? r[0] - h[1] : h[0] - r[1]];
    }

    transition(a, an, ha, b, bn, dt) {
      const w = this.w;
      const [nh, shift] = this.window(ha, b);
      let cost = 0;
      if (shift > 0) cost += w.shiftFixed + w.shift * shift * (0.6 + 0.12 / Math.max(dt, 0.03));
      if (a.pos.length === 1 && b.pos.length === 1) cost += w.jump * Math.max(0, Math.abs(a.pos[0][0] - b.pos[0][0]) - 1);
      const aBy = new Map(an.map((n, i) => [n.pitch, a.pos[i]]));
      bn.forEach((n, i) => {
        for (const t of n.techniques || []) {
          if (t.type === "hammer_on" || t.type === "pull_off") {
            const src = aBy.get(t.fromPitch);
            if (src) cost += src[0] === b.pos[i][0] ? -w.legato : 0.4;
          }
        }
      });
      const bStart = Math.min(...bn.map((n) => n.start));
      an.forEach((n, i) => {
        if (n.end - bStart > 0.08) bn.forEach((m, j) => { if (b.pos[j][0] === a.pos[i][0] && m.pitch !== n.pitch) cost += w.sustain; });
      });
      if (a.pos.length >= 2) {
        const pa = an.map((n) => n.pitch).sort().join(",");
        const pb = bn.map((n) => n.pitch).sort().join(",");
        const sa = a.pos.map((p) => p.join(":")).sort().join(",");
        const sb = b.pos.map((p) => p.join(":")).sort().join(",");
        if (pa === pb && sa !== sb) cost += w.repeat;
      }
      return [cost, nh];
    }

    optimize(notes) {
      const groups = groupNotes(notes.filter((n) => !n.deleted));
      const cl = [];
      const gl = [];
      const unplayable = [];
      for (let g of groups) {
        g = [...g].sort((a, b) => b.pitch - a.pitch);
        let cands = this.candidates(g);
        while ((!cands.length || !cands[0].feasible) && g.length > 1) {
          const worst = g.reduce((m, n) => ((n.confidence ?? 0.5) < (m.confidence ?? 0.5) ? n : m), g[0]);
          g = g.filter((n) => n !== worst);
          worst.string = worst.fret = null;
          unplayable.push(worst);
          cands = this.candidates(g);
        }
        if (!cands.length) { g.forEach((n) => { n.string = n.fret = null; unplayable.push(n); }); continue; }
        cl.push(cands);
        gl.push(g);
      }
      const G = cl.length;
      if (!G) return { unplayable, groups: 0 };
      const dp = cl.map((c) => new Float64Array(c.length).fill(Infinity));
      const bk = cl.map((c) => new Int32Array(c.length));
      const hand = cl.map((c) => new Array(c.length).fill(null));
      cl[0].forEach((c, k) => { dp[0][k] = c.cost; hand[0][k] = this.reach(c); });
      for (let g = 1; g < G; g++) {
        const dt = gl[g][0].start - gl[g - 1][0].start;
        cl[g].forEach((cb, k) => {
          let best = Infinity, arg = 0, bh = null;
          cl[g - 1].forEach((ca, j) => {
            if (!isFinite(dp[g - 1][j])) return;
            const [tc, nh] = this.transition(ca, gl[g - 1], hand[g - 1][j], cb, gl[g], dt);
            const v = dp[g - 1][j] + tc;
            if (v < best) { best = v; arg = j; bh = nh; }
          });
          dp[g][k] = best + cb.cost;
          bk[g][k] = arg;
          hand[g][k] = bh;
        });
      }
      let k = 0;
      for (let i = 1; i < dp[G - 1].length; i++) if (dp[G - 1][i] < dp[G - 1][k]) k = i;
      const path = new Array(G);
      path[G - 1] = k;
      for (let g = G - 1; g > 0; g--) path[g - 1] = bk[g][path[g]];
      for (let g = 0; g < G; g++) {
        const cands = cl[g];
        const chosen = cands[path[g]];
        const notesG = gl[g];
        const prev = g > 0 ? [cl[g - 1][path[g - 1]], gl[g - 1], hand[g - 1][path[g - 1]]] : null;
        const next = g + 1 < G ? [cl[g + 1][path[g + 1]], gl[g + 1]] : null;
        const local = (c) => {
          let v = c.cost;
          let h = this.reach(c);
          if (prev) { const [tc, nh] = this.transition(prev[0], prev[1], prev[2], c, notesG, notesG[0].start - prev[1][0].start); v += tc; h = nh; }
          if (next) v += this.transition(c, notesG, h, next[0], next[1], next[1][0].start - notesG[0].start)[0];
          return v;
        };
        const base = local(chosen);
        const alts = cands.filter((c) => c !== chosen).map((c) => [local(c) - base, c]).sort((a, b) => a[0] - b[0]).slice(0, 4);
        notesG.forEach((n, i) => {
          [n.string, n.fret] = chosen.pos[i];
          n.group = g;
          const seen = new Set([`${n.string}:${n.fret}`]);
          n.alternatives = [];
          for (const [d, c] of alts) {
            const key = c.pos[i].join(":");
            if (!seen.has(key)) { seen.add(key); n.alternatives.push({ string: c.pos[i][0], fret: c.pos[i][1], delta: +d.toFixed(2) }); }
          }
          n.uncertainString = alts.length > 0 && alts[0][0] < 0.35 && !n.locked && n.alternatives.length > 0;
          n.impossible = !chosen.feasible;
        });
      }
      return { unplayable, groups: G, cost: dp[G - 1][k] };
    }
  }

  function groupNotes(notes, tol = 0.035) {
    const groups = [];
    for (const n of [...notes].sort((a, b) => a.start - b.start || b.pitch - a.pitch)) {
      const g = groups[groups.length - 1];
      if (g && n.start - g[0].start <= tol && !g.some((m) => m.pitch === n.pitch)) g.push(n);
      else groups.push([n]);
    }
    return groups;
  }

  /** Confirm legato only when both notes ended up on the same string. */
  function finalizeTechniques(notes) {
    const order = notes.filter((n) => !n.deleted && n.string).sort((a, b) => a.start - b.start);
    order.forEach((n, i) => {
      n.techniques = (n.techniques || []).filter((t) => {
        if (t.type === "hammer_on" || t.type === "pull_off") {
          const prev = order.slice(Math.max(0, i - 6), i).reverse().find((m) => m.pitch === t.fromPitch);
          return prev && prev.string === n.string;
        }
        if (/bend|vibrato/.test(t.type) && n.fret === 0) return false;
        return true;
      });
    });
  }

  // ------------------------------------------------------------------ tuning detection
  function detectTuning(notes) {
    const use = notes.filter((n) => n.confidence >= 0.35);
    if (use.length < 8) return { tuning: TUNINGS[0], confidence: 0.2, ranking: [] };
    const sample = use.length > 300 ? use.slice(0, 300) : use;
    const groups = groupNotes(use);
    const roots = groups.filter((g) => g.length >= 2).map((g) => Math.min(...g.map((n) => n.pitch)));
    const results = TUNINGS.map((t) => {
      const lowest = t.notes[0];
      const below = use.filter((n) => n.pitch < lowest).length / use.length;
      const trial = sample.filter((n) => n.pitch >= lowest).map((n) => ({ ...n, string: null, fret: null, locked: false }));
      const res = trial.length ? new Optimizer(t).optimize(trial) : { cost: 0 };
      const cpn = (res.cost || 0) / Math.max(1, trial.length);
      const lowHits = roots.length ? roots.filter((r) => r === lowest).length / roots.length : use.filter((n) => n.pitch === lowest).length / use.length;
      const longs = use.filter((n) => n.end - n.start > 0.45);
      const openLong = longs.length ? longs.filter((n) => t.notes.includes(n.pitch)).length / longs.length : 0;
      const score = -12 * below - 0.9 * cpn + 1.4 * openLong + 2.2 * lowHits + 0.6 * Math.log(t.prior);
      return { tuning: t, score };
    });
    const mx = Math.max(...results.map((r) => r.score));
    const ex = results.map((r) => Math.exp((r.score - mx) * 1.5));
    const sum = ex.reduce((a, b) => a + b, 0);
    const ranking = results.map((r, i) => ({ tuning: r.tuning, p: ex[i] / sum })).sort((a, b) => b.p - a.p);
    return { tuning: ranking[0].tuning, confidence: ranking[0].p, ranking };
  }

  // ------------------------------------------------------------------ full pipeline
  /**
   * Turn posteriorgrams into a tab document.
   * opts: {sensitivity 0..1, tuningId|'auto', capo, profile, beatsPerBar, bpmOverride, prior}
   */
  function transcribe(post, duration, opts = {}, y = null) {
    const onsetThreshold = 0.5;
    const frameThreshold = 0.3;
    let raw = decodeNotes(post, { onsetThreshold, frameThreshold, minMidi: 26 });
    raw = mergeFragments(raw);
    const { keep, rejected } = suppressGhosts(raw);
    const minConf = 0.2; // the display threshold (doc.minConfidence) hides more without losing data
    let notes = [];
    for (const n of keep) {
      let c = clamp((n.amplitude - 0.22) / 0.5, 0, 1);
      if (n.end - n.start < 0.08) c *= 0.88;
      if (n.possibleHarmonic) c *= 0.8;
      n.confidence = c;
      if (c >= minConf) notes.push(n);
      else rejected.push(n);
    }
    notes = mergeBendChains(notes);
    refineMonophonic(notes, y);
    detectTechniques(notes);
    // tempo / beats
    const env = onsetEnvelope(post);
    let bpm = opts.bpmOverride || estimateTempo(env);
    const tb = trackBeats(env, bpm, duration);
    bpm = opts.bpmOverride || tb.bpm;
    const beats = opts.bpmOverride ? extendGrid([tb.beats[0], tb.beats[0] + 60 / bpm], duration, bpm) : tb.beats;
    const bpb = opts.beatsPerBar || 4;
    // tuning
    let tuning = TUNINGS.find((t) => t.id === opts.tuningId);
    let tuningInfo = null;
    if (!tuning) {
      tuningInfo = detectTuning(notes);
      tuning = tuningInfo.tuning;
    }
    const doc = {
      version: 1,
      duration,
      bpm,
      beats,
      beatsPerBar: bpb,
      tuning,
      tuningInfo: tuningInfo && { id: tuningInfo.tuning.id, confidence: +tuningInfo.confidence.toFixed(2), ranking: tuningInfo.ranking.slice(0, 4).map((r) => ({ id: r.tuning.id, name: r.tuning.name, p: +r.p.toFixed(2) })) },
      capo: opts.capo || 0,
      profile: opts.profile || "balanced",
      minConfidence: opts.minConfidence ?? 0.45,
      notes: notes.map((n, i) => ({
        id: `n${i}`,
        start: +n.start.toFixed(4),
        end: +n.end.toFixed(4),
        pitch: n.pitch,
        confidence: +n.confidence.toFixed(3),
        techniques: n.techniques,
        flags: [n.possibleHarmonic ? "possible_harmonic" : null, n.noOnset ? "legato" : null].filter(Boolean),
        velocity: Math.round(clamp(40 + 87 * (0.6 * n.amplitude + 0.4 * n.onsetStrength), 20, 127)),
      })),
      rejected: rejected.length,
    };
    doc.downbeatPhase = opts.downbeatPhase ?? chooseDownbeat(beats, doc.notes.map((n) => ({ ...n, amplitude: n.confidence })), bpb);
    refinger(doc, opts.prior);
    return doc;
  }

  /** (Re)compute string/fret for a document, keeping locked notes. */
  function refinger(doc, prior) {
    const opt = new Optimizer(doc.tuning, { capo: doc.capo, profile: doc.profile, prior });
    const thr = doc.minConfidence ?? 0.45;
    for (const n of doc.notes) n.hidden = !n.deleted && !n.edited && n.confidence < thr;
    const live = doc.notes.filter((n) => !n.deleted && !n.hidden);
    for (const n of doc.notes) if (!n.locked) n.string = n.fret = null;
    const res = opt.optimize(live);
    finalizeTechniques(live);
    doc.measures = buildMeasures(doc.beats, ((doc.downbeatPhase % doc.beatsPerBar) + doc.beatsPerBar) % doc.beatsPerBar, doc.beatsPerBar, doc.duration);
    doc.unplayable = res.unplayable.length;
    return doc;
  }

  /** Halve (factor 0.5) or double (2) the beat grid, e.g. to fix a tempo octave error. */
  function regrid(doc, factor) {
    const b = doc.beats;
    if (factor === 2) {
      const out = [];
      for (let i = 0; i + 1 < b.length; i++) out.push(b[i], (b[i] + b[i + 1]) / 2);
      out.push(b[b.length - 1]);
      doc.beats = out;
    } else if (factor === 0.5) {
      doc.beats = b.filter((_, i) => i % 2 === 0);
    }
    doc.bpm *= factor;
    doc.downbeatPhase = 0;
    return doc;
  }

  // ------------------------------------------------------------------ text tab
  function toText(doc, title) {
    const tun = doc.tuning;
    const n = tun.notes.length;
    const labels = [...tun.notes].reverse().map((p) => midiName(p).replace(/-?\d+$/, ""));
    const w = Math.max(...labels.map((l) => l.length));
    const lines = [
      `${title}`,
      `Tuning: ${tun.name}${doc.capo ? `   Capo ${doc.capo}` : ""}   Tempo: ~${Math.round(doc.bpm)} BPM`,
      "h hammer-on  p pull-off  b bend  r release  ~ vibrato   (TabForge Pocket, verify by ear)",
      "",
    ];
    const live = doc.notes.filter((x) => !x.deleted && !x.hidden && x.string);
    const per = 4;
    for (let m0 = 0; m0 < doc.measures.length; m0 += per) {
      const rows = labels.map((l) => l.padEnd(w) + "|");
      for (let m = m0; m < Math.min(doc.measures.length, m0 + per); m++) {
        const ms = doc.measures[m];
        const slots = doc.beatsPerBar * 4; // 16th-note grid
        const cells = Array.from({ length: n }, () => new Array(slots).fill(""));
        for (const x of live) {
          if (x.start < ms.start - 0.03 || x.start >= ms.end - 0.03) continue;
          const pos = beatPos(ms.beats, x.start);
          const slot = clamp(Math.round(pos * 4), 0, slots - 1);
          let txt = String(x.fret);
          for (const t of x.techniques || []) {
            if (t.type === "hammer_on") txt = "h" + txt;
            if (t.type === "pull_off") txt = "p" + txt;
            if (t.type === "bend") txt += "b" + (x.fret + Math.round(t.semitones));
            if (t.type === "bend_release") txt += "b" + (x.fret + Math.round(t.semitones)) + "r" + x.fret;
            if (t.type === "vibrato") txt += "~";
          }
          cells[x.string - 1][slot] = txt;
        }
        const widths = Array.from({ length: slots }, (_, k) => Math.max(2, ...cells.map((r) => r[k].length + 1)));
        for (let s = 0; s < n; s++) rows[s] += cells[s].map((c, k) => (c || "").padEnd(widths[k], "-")).join("") + "-|";
      }
      lines.push(...rows, "");
    }
    return lines.join("\n");
  }

  function beatPos(beats, t) {
    for (let i = 0; i + 1 < beats.length; i++) {
      if (t < beats[i + 1] || i + 2 === beats.length) return i + (t - beats[i]) / (beats[i + 1] - beats[i]);
    }
    return 0;
  }

  const api = { SR, FPS, TUNINGS, midiName, openMidi, runModel, decodeNotes, transcribe, refinger, regrid, toText, groupNotes, Optimizer, beatPos, _internal: { mergeFragments, suppressGhosts, mergeBendChains, refineMonophonic, detectBend, detectVibrato } };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TabEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
