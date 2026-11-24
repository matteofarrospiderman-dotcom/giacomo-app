import { audioContext } from "./engine";

/**
 * Karplus-Strong plucked-string synthesis for previewing individual notes and chords.
 * Buffers are cached per pitch so repeated clicks are instant.
 */
const cache = new Map<string, AudioBuffer>();

function render(midi: number, seconds: number, bright: number, palmMute: boolean): AudioBuffer {
  const ctx = audioContext();
  const sr = ctx.sampleRate;
  const freq = 440 * Math.pow(2, (midi - 69) / 12);
  const n = Math.floor(sr * seconds);
  const buf = ctx.createBuffer(1, n, sr);
  const out = buf.getChannelData(0);
  const period = sr / freq;
  const len = Math.max(2, Math.floor(period));
  const frac = period - len;
  const line = new Float32Array(len + 2);
  let prevNoise = 0;
  for (let i = 0; i < line.length; i++) {
    const w = Math.random() * 2 - 1;
    prevNoise = bright * w + (1 - bright) * prevNoise;
    line[i] = prevNoise;
  }
  const decay = palmMute ? 0.94 : 0.996;
  let idx = 0;
  let last = 0;
  for (let i = 0; i < n; i++) {
    const a = line[idx];
    const b = line[(idx + 1) % line.length];
    const s = a * (1 - frac) + b * frac;
    const y = decay * 0.5 * (s + last);
    last = s;
    line[idx] = y;
    idx = (idx + 1) % len;
    const env = Math.min(1, i / (sr * 0.002)) * (i > n - sr * 0.05 ? (n - i) / (sr * 0.05) : 1);
    out[i] = s * env * 0.6;
  }
  return buf;
}

export function playPitches(pitches: number[], opts: { seconds?: number; strum?: number; palmMute?: boolean } = {}) {
  const ctx = audioContext();
  const seconds = opts.seconds ?? 1.4;
  const strum = opts.strum ?? 0.018;
  const master = ctx.createGain();
  master.gain.value = Math.min(0.9, 1.2 / Math.sqrt(Math.max(1, pitches.length)));
  master.connect(ctx.destination);
  [...pitches].sort((a, b) => a - b).forEach((p, i) => {
    const key = `${p}:${seconds}:${opts.palmMute ? 1 : 0}`;
    let buf = cache.get(key);
    if (!buf) {
      buf = render(p, seconds, 0.55, !!opts.palmMute);
      cache.set(key, buf);
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(master);
    src.start(ctx.currentTime + i * strum);
  });
}
