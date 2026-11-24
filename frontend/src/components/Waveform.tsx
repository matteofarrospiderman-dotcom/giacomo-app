import { useEffect, useRef, useState } from "react";
import type { Player } from "../audio/engine";
import { fetchPeaks } from "../lib/api";

interface Props {
  id: string;
  peaksPath: string;
  duration: number;
  player: Player;
  height?: number;
  color?: string;
  beats?: number[];
  downbeats?: number[];
}

/** Canvas waveform with beat grid, playhead, click-to-seek and drag-to-loop. */
export default function Waveform({ id, peaksPath, duration, player, height = 90, color = "#f59e0b", beats, downbeats }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const [peaks, setPeaks] = useState<{ sr: number; min: number[]; max: number[] } | null>(null);
  const [width, setWidth] = useState(800);
  const drag = useRef<{ x0: number; t0: number } | null>(null);
  const [sel, setSel] = useState<[number, number] | null>(null);

  useEffect(() => {
    fetchPeaks(id, peaksPath).then(setPeaks).catch(() => setPeaks(null));
  }, [id, peaksPath]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const c = canvas.current;
    if (!c || !peaks) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = height * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, width, height);
    const mid = height / 2;
    const n = peaks.max.length;
    let peak = 0.0001;
    for (let i = 0; i < n; i++) peak = Math.max(peak, peaks.max[i], -peaks.min[i]);
    if (beats) {
      g.fillStyle = "rgba(255,255,255,0.05)";
      for (const b of beats) g.fillRect((b / duration) * width, 0, 1, height);
    }
    if (downbeats) {
      g.fillStyle = "rgba(56,189,248,0.22)";
      for (const b of downbeats) g.fillRect((b / duration) * width, 0, 1, height);
    }
    g.fillStyle = color;
    for (let x = 0; x < width; x++) {
      const a = Math.floor((x / width) * n);
      const b = Math.max(a + 1, Math.floor(((x + 1) / width) * n));
      let mn = 0;
      let mx = 0;
      for (let i = a; i < b && i < n; i++) {
        mn = Math.min(mn, peaks.min[i]);
        mx = Math.max(mx, peaks.max[i]);
      }
      const y1 = mid - (mx / peak) * (mid - 2);
      const y2 = mid - (mn / peak) * (mid - 2);
      g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
  }, [peaks, width, height, color, beats, downbeats, duration]);

  useEffect(
    () =>
      player.onFrame((t) => {
        if (head.current) head.current.style.transform = `translateX(${(t / duration) * width}px)`;
      }),
    [player, duration, width],
  );

  useEffect(() => player.subscribe(() => setSel(player.state.loop)), [player]);

  const tAt = (clientX: number) => {
    const r = wrap.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((clientX - r.left) / r.width) * duration));
  };

  return (
    <div
      ref={wrap}
      className="waveform"
      style={{ height }}
      onPointerDown={(e) => {
        // pointer events: the same drag-to-loop works with a mouse, a pen or a finger
        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        drag.current = { x0: e.clientX, t0: tAt(e.clientX) };
      }}
      onPointerMove={(e) => {
        if (drag.current && Math.abs(e.clientX - drag.current.x0) > 4) {
          const t1 = tAt(e.clientX);
          setSel([Math.min(drag.current.t0, t1), Math.max(drag.current.t0, t1)]);
        }
      }}
      onPointerCancel={() => (drag.current = null)}
      onPointerUp={(e) => {
        if (!drag.current) return;
        if (Math.abs(e.clientX - drag.current.x0) > 4) {
          const t1 = tAt(e.clientX);
          const loop: [number, number] = [Math.min(drag.current.t0, t1), Math.max(drag.current.t0, t1)];
          player.setLoop(loop, true);
          player.seek(loop[0]);
        } else player.seek(tAt(e.clientX));
        drag.current = null;
      }}
    >
      <canvas ref={canvas} style={{ height }} />
      {sel && (
        <div style={{ position: "absolute", top: 0, bottom: 0, left: `${(sel[0] / duration) * 100}%`, width: `${((sel[1] - sel[0]) / duration) * 100}%`, background: "rgba(56,189,248,0.14)", borderLeft: "1px solid var(--cyan)", borderRight: "1px solid var(--cyan)", pointerEvents: "none" }} />
      )}
      <div ref={head} className="playhead" style={{ height, left: 0 }} />
      {!peaks && <div className="sub" style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>loading waveform…</div>}
    </div>
  );
}
