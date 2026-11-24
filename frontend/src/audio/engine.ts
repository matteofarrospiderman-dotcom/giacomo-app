/**
 * Multi-source synchronized player.
 *
 * Each source (original mix + every stem) is an HTMLAudioElement routed through a Web Audio
 * GainNode for mute/solo/volume. HTMLMediaElement is used (instead of AudioBufferSourceNode)
 * because its `preservesPitch` time-stretching lets the user slow down to 25% without changing
 * pitch, and because it streams long files instead of decoding everything into memory.
 * The first source is the clock; the others are re-synced whenever they drift > 50 ms.
 */

export interface SourceSpec {
  id: string;
  label: string;
  url: string;
}

interface Source extends SourceSpec {
  el: HTMLAudioElement;
  node: MediaElementAudioSourceNode | null;
  gain: GainNode | null;
  volume: number;
  muted: boolean;
  solo: boolean;
}

export interface PlayerState {
  playing: boolean;
  rate: number;
  loop: [number, number] | null;
  loopOn: boolean;
  duration: number;
  metronome: boolean;
  sources: { id: string; label: string; volume: number; muted: boolean; solo: boolean }[];
}

type Listener = () => void;
type FrameListener = (t: number) => void;

let sharedCtx: AudioContext | null = null;
export function audioContext(): AudioContext {
  if (!sharedCtx) {
    // iOS: play through the ring/silent switch like a music app
    try {
      const nav = navigator as Navigator & { audioSession?: { type: string } };
      if (nav.audioSession) nav.audioSession.type = "playback";
    } catch {
      /* not supported */
    }
    sharedCtx = new AudioContext({ latencyHint: "interactive" });
  }
  // "interrupted" is iOS after a phone call or backgrounding
  if (sharedCtx.state !== "running") void sharedCtx.resume().catch(() => undefined);
  return sharedCtx;
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && sharedCtx && sharedCtx.state !== "running") void sharedCtx.resume().catch(() => undefined);
});

export class Player {
  private sources: Source[] = [];
  private listeners = new Set<Listener>();
  private frameListeners = new Set<FrameListener>();
  private raf = 0;
  private lastBeatIdx = -1;
  beats: number[] = [];
  downbeats = new Set<number>();
  state: PlayerState = { playing: false, rate: 1, loop: null, loopOn: false, duration: 0, metronome: false, sources: [] };

  load(specs: SourceSpec[], defaults: Record<string, { muted?: boolean; volume?: number }> = {}) {
    this.destroySources();
    this.sources = specs.map((s) => {
      const el = new Audio();
      el.preload = "auto";
      el.src = s.url;
      (el as any).preservesPitch = true;
      (el as any).mozPreservesPitch = true;
      (el as any).webkitPreservesPitch = true;
      el.addEventListener("loadedmetadata", () => {
        if (s === specs[0]) this.set({ duration: el.duration });
      });
      el.addEventListener("ended", () => {
        if (s !== specs[0]) return;
        const { loop, loopOn } = this.state;
        if (loop && loopOn) {
          // a loop that runs to the very end of the song wraps instead of stopping
          this.seek(loop[0]);
          void this.play();
        } else this.pause();
      });
      return { ...s, el, node: null, gain: null, volume: defaults[s.id]?.volume ?? 1, muted: defaults[s.id]?.muted ?? false, solo: false };
    });
    this.emitSources();
    this.startClock();
    this.setupMediaSession(specs[0]?.label ?? "TabForge");
  }

  /** Lock-screen / headphone / Bluetooth pedal controls. */
  private setupMediaSession(title: string) {
    try {
      if (!("mediaSession" in navigator)) return;
      navigator.mediaSession.metadata = new MediaMetadata({ title, artist: "TabForge" });
      navigator.mediaSession.setActionHandler("play", () => void this.play());
      navigator.mediaSession.setActionHandler("pause", () => this.pause());
      navigator.mediaSession.setActionHandler("seekbackward", () => this.seek(this.time - 5));
      navigator.mediaSession.setActionHandler("seekforward", () => this.seek(this.time + 5));
    } catch {
      /* optional */
    }
  }

  private ensureGraph() {
    const ctx = audioContext();
    for (const s of this.sources) {
      if (!s.node) {
        s.node = ctx.createMediaElementSource(s.el);
        s.gain = ctx.createGain();
        s.node.connect(s.gain).connect(ctx.destination);
      }
    }
    this.applyGains();
  }

  private applyGains() {
    const anySolo = this.sources.some((s) => s.solo);
    for (const s of this.sources) {
      const audible = anySolo ? s.solo : !s.muted;
      const g = audible ? s.volume : 0;
      if (s.gain) s.gain.gain.setTargetAtTime(g, audioContext().currentTime, 0.01);
      else s.el.volume = Math.min(1, g);
    }
  }

  get master(): HTMLAudioElement | null {
    return this.sources[0]?.el ?? null;
  }

  get time(): number {
    return this.master?.currentTime ?? 0;
  }

  private playGen = 0;
  private wake: { release: () => Promise<void> } | null = null;
  private lastResync = 0;

  async play() {
    if (!this.master) return;
    const gen = ++this.playGen;
    this.ensureGraph();
    const t = this.master.currentTime;
    for (const s of this.sources) {
      s.el.playbackRate = this.state.rate;
      s.el.currentTime = t;
    }
    const results = await Promise.allSettled(this.sources.map((s) => s.el.play()));
    if (gen !== this.playGen) return; // paused (or played again) while play() was pending
    if (results[0].status === "rejected") {
      // e.g. iOS NotAllowedError: do not claim to be playing when there is no sound
      for (const s of this.sources) s.el.pause();
      this.set({ playing: false });
      return;
    }
    this.lastBeatIdx = this.beatIndex(t);
    this.set({ playing: true });
    this.startClock();
    void this.keepAwake(true);
  }

  pause() {
    this.playGen++;
    for (const s of this.sources) s.el.pause();
    this.set({ playing: false });
    void this.keepAwake(false);
    this.frameListeners.forEach((f) => f(this.time));
  }

  /** Keep the screen on while practising (tolerates browsers that refuse). */
  private async keepAwake(on: boolean) {
    try {
      const nav = navigator as Navigator & { wakeLock?: { request: (t: string) => Promise<{ release: () => Promise<void> }> } };
      if (on && !this.wake && nav.wakeLock) this.wake = await nav.wakeLock.request("screen");
      if (!on && this.wake) {
        await this.wake.release();
        this.wake = null;
      }
    } catch {
      this.wake = null;
    }
  }

  toggle() {
    if (this.state.playing) this.pause();
    else void this.play();
  }

  seek(t: number) {
    const d = this.state.duration || this.master?.duration || 0;
    const tt = Math.max(0, d ? Math.min(t, d - 0.05) : t);
    for (const s of this.sources) s.el.currentTime = tt;
    this.lastBeatIdx = this.beatIndex(tt);
    this.frameListeners.forEach((f) => f(tt));
  }

  setRate(r: number) {
    for (const s of this.sources) s.el.playbackRate = r;
    this.set({ rate: r });
  }

  setLoop(loop: [number, number] | null, on?: boolean) {
    this.set({ loop, loopOn: on ?? (loop ? true : false) });
  }

  toggleLoop() {
    if (this.state.loop) this.set({ loopOn: !this.state.loopOn });
  }

  setMetronome(on: boolean) {
    this.set({ metronome: on });
  }

  setSourceProp(id: string, patch: Partial<Pick<Source, "volume" | "muted" | "solo">>) {
    const s = this.sources.find((x) => x.id === id);
    if (!s) return;
    Object.assign(s, patch);
    this.applyGains();
    this.emitSources();
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  onFrame(fn: FrameListener): () => void {
    this.frameListeners.add(fn);
    return () => this.frameListeners.delete(fn);
  }

  private set(patch: Partial<PlayerState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((l) => l());
  }

  private emitSources() {
    this.set({ sources: this.sources.map(({ id, label, volume, muted, solo }) => ({ id, label, volume, muted, solo })) });
  }

  private beatIndex(t: number): number {
    let lo = 0;
    let hi = this.beats.length - 1;
    if (hi < 0 || t < this.beats[0]) return -1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.beats[mid] <= t) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }

  private click(accent: boolean) {
    const ctx = audioContext();
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.frequency.value = accent ? 1760 : 1175;
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(accent ? 0.5 : 0.3, ctx.currentTime + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.05);
    o.connect(g).connect(ctx.destination);
    o.start();
    o.stop(ctx.currentTime + 0.06);
  }

  /** Animation-frame clock; runs only while playing (no battery drain when paused). */
  private startClock = () => {
    cancelAnimationFrame(this.raf);
    const tick = () => {
      const m = this.master;
      if (m) {
        const t = m.currentTime;
        if (this.state.playing) {
          const now = performance.now();
          if (now - this.lastResync > 500) {
            // re-sync stems that drifted; skip elements still seeking or buffering, otherwise
            // every correction stalls them again and they never catch up
            for (let i = 1; i < this.sources.length; i++) {
              const el = this.sources[i].el;
              if (!el.seeking && el.readyState >= 3 && Math.abs(el.currentTime - t) > 0.1) {
                el.currentTime = t;
                this.lastResync = now;
              }
            }
          }
          const { loop, loopOn } = this.state;
          if (loop && loopOn && t >= loop[1]) this.seek(loop[0]);
          if (this.state.metronome && this.beats.length) {
            const bi = this.beatIndex(t);
            if (bi > this.lastBeatIdx && bi >= 0) this.click(this.downbeats.has(Math.round(this.beats[bi] * 1000)));
            this.lastBeatIdx = bi;
          }
        }
        this.frameListeners.forEach((f) => f(t));
      }
      if (this.state.playing) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  };

  private destroySources() {
    for (const s of this.sources) {
      s.el.pause();
      s.el.removeAttribute("src");
      s.el.load();
      try {
        s.node?.disconnect();
        s.gain?.disconnect();
      } catch {
        /* already disconnected */
      }
    }
    this.sources = [];
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.destroySources();
    this.listeners.clear();
    this.frameListeners.clear();
  }
}
