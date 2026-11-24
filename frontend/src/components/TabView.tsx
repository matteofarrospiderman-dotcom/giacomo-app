import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Player } from "../audio/engine";
import { midiName } from "../lib/format";
import type { Chord, NotatedBeat, Note, Track } from "../lib/types";

const STR_GAP = 15;
const TOP = 40;
const BOTTOM = 34;
const PAD = 14;
const LEFT = 30;

interface Layout {
  rows: { y: number; h: number; measures: { idx: number; x: number; w: number }[] }[];
  height: number;
  meas: Map<number, { row: number; x: number; w: number }>;
}

function measureNeed(beats: NotatedBeat[]): number {
  const n = beats.filter((b) => !b.rest || b.ticks >= 960).length;
  return Math.max(150, 26 + n * 24);
}

function layout(track: Track, width: number): Layout {
  const usable = Math.max(300, width - LEFT - 10);
  const nStr = track.tuning.strings;
  const h = TOP + (nStr - 1) * STR_GAP + BOTTOM;
  const rows: Layout["rows"] = [];
  let cur: { idx: number; need: number }[] = [];
  let curW = 0;
  const flush = () => {
    if (!cur.length) return;
    const scale = usable / curW;
    let x = LEFT;
    const last = rows.length === 0 ? 0 : rows[rows.length - 1].y + rows[rows.length - 1].h;
    const stretch = cur.length > 1 || curW > usable * 0.5 ? scale : Math.min(scale, 1.6);
    rows.push({ y: last, h, measures: cur.map((m) => { const w = m.need * stretch; const r = { idx: m.idx, x, w }; x += w; return r; }) });
    cur = [];
    curW = 0;
  };
  track.notation.forEach((m, i) => {
    const need = measureNeed(m.beats);
    if (curW + need > usable && cur.length) flush();
    cur.push({ idx: i, need });
    curW += need;
  });
  flush();
  const meas = new Map<number, { row: number; x: number; w: number }>();
  rows.forEach((r, ri) => r.measures.forEach((m) => meas.set(m.idx, { row: ri, x: m.x, w: m.w })));
  return { rows, height: (rows.at(-1)?.y ?? 0) + h + 10, meas };
}

/** Bend size in guitarist's terms: 1 semitone = "½", 2 = "full", 3 = "1½". */
function bendLabel(semitones: number | undefined): string {
  const st = Number(semitones ?? 1);
  if (st === 2) return "full";
  const steps = st / 2;
  const whole = Math.floor(steps);
  const half = steps - whole >= 0.5 ? "½" : "";
  return `${whole || ""}${half}` || "¼";
}

function techLabel(n: Note): { above: string[]; prefix: string; suffix: string } {
  const above: string[] = [];
  let prefix = "";
  let suffix = "";
  for (const t of n.techniques) {
    switch (t.type) {
      case "hammer_on": prefix = "h"; break;
      case "pull_off": prefix = "p"; break;
      case "slide": suffix += t.direction === "down" ? "\\" : "/"; break;
      case "slide_in": prefix = t.direction === "from_above" ? "\\" : "/"; break;
      case "bend": above.push(`↑${bendLabel(t.semitones)}`); break;
      case "bend_release": above.push(`↑↓${bendLabel(t.semitones)}`); break;
      case "harmonic": prefix = "<"; suffix += ">"; break;
      case "prebend_release": above.push("pb↓"); break;
      case "vibrato": above.push("~~"); break;
      case "palm_mute": above.push("PM"); break;
      case "let_ring": above.push("LR"); break;
      case "tremolo_picking": above.push("≡"); break;
    }
  }
  return { above, prefix, suffix };
}

interface Props {
  track: Track;
  chords: Chord[];
  player: Player;
  selectedId: string | null;
  onSelect: (n: Note, opts?: { seek?: boolean }) => void;
  compare: Record<string, [number, number]> | null;
  follow: boolean;
  scrollTo?: string | null;
  /** song bar number of the track's first measure minus one (tab and transport agree) */
  barOffset?: number;
  onBarTap?: (measureIdx: number) => void;
}

function TabViewInner({ track, chords, player, selectedId, onSelect, compare, follow, scrollTo, barOffset = 0, onBarTap }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const head = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(900);
  const drag = useRef<{ x: number; y: number; t: number } | null>(null);
  const [loopSel, setLoopSel] = useState<[number, number] | null>(player.state.loop);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const L = useMemo(() => layout(track, width), [track, width]);
  const noteById = useMemo(() => new Map(track.notes.map((n) => [n.id, n])), [track.notes]);
  const nStr = track.tuning.strings;
  const labels = useMemo(() => [...track.tuning.notes].reverse().map((p) => midiName(p).replace(/-?\d+$/, "")), [track.tuning.notes]);
  const sorted = useMemo(() => [...track.notes].filter((n) => n.string).sort((a, b) => a.start - b.start), [track.notes]);

  // time <-> x mapping via measures
  const timeToPos = (t: number): { x: number; row: number } | null => {
    const ms = track.measures;
    if (!ms.length) return null;
    let lo = 0;
    let hi = ms.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (ms[mid].start <= t) lo = mid;
      else hi = mid - 1;
    }
    const m = ms[lo];
    const geo = L.meas.get(lo);
    if (!geo) return null;
    const frac = Math.max(0, Math.min(1, (t - m.start) / Math.max(1e-3, m.end - m.start)));
    return { x: geo.x + PAD + frac * (geo.w - 2 * PAD), row: geo.row };
  };
  const posToTime = (x: number, y: number): number | null => {
    const ri = L.rows.findIndex((r) => y >= r.y && y < r.y + r.h);
    if (ri < 0) return null;
    const m = L.rows[ri].measures.find((mm) => x >= mm.x && x < mm.x + mm.w);
    if (!m) return null;
    const meas = track.measures[m.idx];
    const frac = Math.max(0, Math.min(1, (x - m.x - PAD) / Math.max(1, m.w - 2 * PAD)));
    return meas.start + frac * (meas.end - meas.start);
  };

  // per-frame playhead + active note highlighting without React re-renders
  useEffect(() => {
    const els = new Map<string, Element[]>();
    svg.current?.querySelectorAll("[data-id]").forEach((el) => {
      const id = el.getAttribute("data-id")!;
      const arr = els.get(id) ?? [];
      arr.push(el);
      els.set(id, arr);
    });
    let lastOn = new Set<string>();
    let lastRow = -1;
    const unsubscribe = player.onFrame((t) => {
      const p = timeToPos(t);
      if (head.current) {
        if (p) {
          const row = L.rows[p.row];
          head.current.style.display = "block";
          head.current.style.transform = `translate(${p.x}px, ${row.y + TOP - 18}px)`;
          head.current.style.height = `${(nStr - 1) * STR_GAP + 36}px`;
          if (follow && player.state.playing && p.row !== lastRow && wrap.current) {
            const sc = wrap.current;
            const top = row.y;
            if (top < sc.scrollTop || top + row.h > sc.scrollTop + sc.clientHeight) sc.scrollTo({ top: Math.max(0, top - 20), behavior: "smooth" });
          }
          lastRow = p.row;
        } else head.current.style.display = "none";
      }
      const on = new Set<string>();
      // binary search first note that could be sounding
      let lo = 0;
      let hi = sorted.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid].start < t - 8) lo = mid + 1;
        else hi = mid;
      }
      for (let i = lo; i < sorted.length && sorted[i].start <= t; i++) if (sorted[i].end > t) on.add(sorted[i].id);
      for (const id of lastOn) if (!on.has(id)) els.get(id)?.forEach((e) => e.classList.remove("on"));
      for (const id of on) if (!lastOn.has(id)) els.get(id)?.forEach((e) => e.classList.add("on"));
      lastOn = on;
    });
    return () => {
      unsubscribe();
      // never leave a note stuck highlighted when the tab re-renders mid-playback
      for (const id of lastOn) els.get(id)?.forEach((e) => e.classList.remove("on"));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [player, L, sorted, follow, nStr]);

  useEffect(() => player.subscribe(() => setLoopSel(player.state.loop)), [player]);

  useEffect(() => {
    if (!scrollTo || !wrap.current) return;
    const el = svg.current?.querySelector(`[data-id="${scrollTo}"]`) as SVGGElement | null;
    if (!el) return;
    const sc = wrap.current;
    // client rects include the row's translate(); getBBox() would not
    const y = el.getBoundingClientRect().top - sc.getBoundingClientRect().top + sc.scrollTop;
    if (y < sc.scrollTop + 20 || y > sc.scrollTop + sc.clientHeight - 40) sc.scrollTo({ top: Math.max(0, y - 120), behavior: "smooth" });
  }, [scrollTo, L]);

  const svgPoint = (e: React.MouseEvent) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const chordMarks = useMemo(() => {
    const out: { x: number; y: number; name: string; key: string }[] = [];
    const ms0 = track.measures[0];
    for (const c of chords) {
      if (c.root == null) continue;
      if (ms0 && c.end <= ms0.start) continue; // intro chords before this part's first bar
      const ms = track.measures;
      const mi = ms.findIndex((m) => m.end > Math.max(c.start, ms0 ? ms0.start : c.start));
      if (mi < 0) continue;
      const m = ms[mi];
      const geo = L.meas.get(mi);
      if (!geo) continue;
      const frac = Math.max(0, (Math.max(c.start, m.start) - m.start) / (m.end - m.start));
      out.push({ x: geo.x + PAD + frac * (geo.w - 2 * PAD), y: L.rows[geo.row].y + 12, name: c.name, key: `${c.start}` });
    }
    // avoid overlapping labels
    out.sort((a, b) => a.y - b.y || a.x - b.x);
    const kept: typeof out = [];
    for (const c of out) {
      const prev = kept.at(-1);
      if (!prev || prev.y !== c.y || c.x - prev.x > 34) kept.push(c);
      else kept[kept.length - 1] = c; // the later chord is the one sounding from here on
    }
    return kept;
  }, [chords, track.measures, L]);

  return (
    <div className="tab-scroll" ref={wrap}>
      <svg
        ref={svg}
        className="tab-svg"
        width={width}
        height={L.height}
        onMouseDown={(e) => {
          if ((e.target as Element).closest(".note")) return;
          const p = svgPoint(e);
          const t = posToTime(p.x, p.y);
          if (t != null) drag.current = { ...p, t };
        }}
        onMouseUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d) return;
          const p = svgPoint(e);
          const t = posToTime(p.x, p.y);
          if (t == null) return;
          if (Math.abs(p.x - d.x) + Math.abs(p.y - d.y) > 8) {
            const loop: [number, number] = [Math.min(d.t, t), Math.max(d.t, t)];
            player.setLoop(loop, true);
            player.seek(loop[0]);
          } else player.seek(t);
        }}
      >
        {L.rows.map((r, ri) => (
          <g key={ri} transform={`translate(0, ${r.y})`}>
            {Array.from({ length: nStr }, (_, s) => (
              <g key={s}>
                <text className="slabel" x={8} y={TOP + s * STR_GAP + 3}>{labels[s]}</text>
                <line className="str" x1={LEFT} x2={r.measures.at(-1)!.x + r.measures.at(-1)!.w} y1={TOP + s * STR_GAP} y2={TOP + s * STR_GAP} />
              </g>
            ))}
            {r.measures.map((m) => (
              <g key={m.idx}>
                <line className="bar" x1={m.x} x2={m.x} y1={TOP} y2={TOP + (nStr - 1) * STR_GAP} />
                <text className="mnum" x={m.x + 2} y={TOP - 22}>{m.idx + 1 + barOffset}</text>
                {onBarTap && (
                  <rect className="mnum-hit" x={m.x} y={TOP - 36} width={Math.min(44, m.w)} height={22}
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); onBarTap(m.idx); }}>
                    <title>Loop this bar (tap another bar number to extend)</title>
                  </rect>
                )}
                {loopSel && (() => {
                  const meas = track.measures[m.idx];
                  const a = Math.max(loopSel[0], meas.start);
                  const b = Math.min(loopSel[1], meas.end);
                  if (b <= a) return null;
                  const fx = (t: number) => m.x + PAD + ((t - meas.start) / (meas.end - meas.start)) * (m.w - 2 * PAD);
                  return <rect className="loopsel" x={fx(a)} width={Math.max(2, fx(b) - fx(a))} y={TOP - 10} height={(nStr - 1) * STR_GAP + 20} />;
                })()}
              </g>
            ))}
            <line className="bar" x1={r.measures.at(-1)!.x + r.measures.at(-1)!.w} x2={r.measures.at(-1)!.x + r.measures.at(-1)!.w} y1={TOP} y2={TOP + (nStr - 1) * STR_GAP} />
            {r.measures.map((m) => {
              const meas = track.notation[m.idx];
              const ticks = track.measures[m.idx]?.ticks ?? 3840;
              return meas.beats.map((b, bi) => {
                const x = m.x + PAD + (b.tick / ticks) * (m.w - 2 * PAD);
                const notes = b.notes.map((ref) => ({ n: noteById.get(ref.id), tie: ref.tie })).filter((v) => v.n && v.n.string);
                const aboveLabels = new Set<string>();
                notes.forEach(({ n, tie }) => { if (!tie) techLabel(n!).above.forEach((l) => aboveLabels.add(l)); });
                const yStemTop = TOP + (nStr - 1) * STR_GAP + 8;
                const flags = { 8: 1, 16: 2, 32: 3, 64: 4 }[b.value] ?? 0;
                return (
                  <g key={`${m.idx}-${bi}`}>
                    {!b.rest && notes.some((v) => !v.tie) && (
                      <g>
                        {b.value >= 2 && <line className="stem" x1={x} x2={x} y1={yStemTop} y2={yStemTop + 14} />}
                        {Array.from({ length: flags }, (_, f) => <line key={f} className="stem" x1={x} x2={x + 5} y1={yStemTop + 14 - f * 3} y2={yStemTop + 10 - f * 3} />)}
                        {b.dotted && <circle cx={x + 4} cy={yStemTop + 16} r={1.2} fill="#8c95aa" />}
                        {b.tuplet && <text className="tech" x={x - 2} y={yStemTop + 25}>3</text>}
                      </g>
                    )}
                    {[...aboveLabels].map((l, k) => <text key={l} className="tech" x={x - 6} y={TOP - 8 - k * 10}>{l}</text>)}
                    {notes.map(({ n, tie }) => {
                      const note = n!;
                      const s = note.string! - 1;
                      if (s >= nStr) return null;
                      const y = TOP + s * STR_GAP;
                      const tl = techLabel(note);
                      const txt = note.dead ? "x" : tie ? `(${note.fret})` : `${tie ? "" : tl.prefix}${note.fret}${tie ? "" : tl.suffix}`;
                      const w = Math.max(12, txt.length * 7.6 + 2);
                      const cls = `note ${note.level} ${note.flags.length ? "flagged" : ""} ${selectedId === note.id ? "sel" : ""}`;
                      const cmp = compare?.[note.id];
                      return (
                        <g key={`${note.id}-${tie}`} className={cls} data-id={note.id}
                          onClick={(e) => { e.stopPropagation(); onSelect(note); }}
                          onDoubleClick={(e) => { e.stopPropagation(); onSelect(note, { seek: true }); }}>
                          <title>{`${note.name} · string ${note.string} fret ${note.fret} · ${Math.round(note.confidence * 100)}% ${note.flags.length ? "· " + note.flags.join(", ") : ""}`}</title>
                          <rect x={x - w / 2} y={y - 7} width={w} height={14} rx={3} />
                          <text x={x} y={y + 0.5} opacity={tie ? 0.55 : 1}>{txt}</text>
                          {cmp && !tie && <text className="cmp" x={x + w / 2 + 1} y={y - 5}>{`${cmp[0]}:${cmp[1]}`}</text>}
                        </g>
                      );
                    })}
                  </g>
                );
              });
            })}
            {chordMarks.filter((c) => c.y === r.y + 12).map((c) => <text key={c.key} className="chord" x={c.x - 4} y={12}>{c.name}</text>)}
          </g>
        ))}
      </svg>
      <div ref={head} className="playhead" style={{ left: 0, top: 0, display: "none" }} />
    </div>
  );
}

const TabView = memo(TabViewInner);
export default TabView;
