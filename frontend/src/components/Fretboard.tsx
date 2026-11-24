import { useMemo } from "react";
import type { Player } from "../audio/engine";
import { midiName } from "../lib/format";
import { usePlaybackTime } from "../lib/hooks";
import type { Note, Track } from "../lib/types";

interface Props {
  track: Track;
  player: Player;
  selected: Note | null;
  onPick: (string: number, fret: number) => void;
}

const FRETS = 24;
const INLAYS = [3, 5, 7, 9, 15, 17, 19, 21];

/** Interactive fretboard: sounding notes (live), selected note, its alternatives; click to re-place. */
export default function Fretboard({ track, player, selected, onPick }: Props) {
  const t = usePlaybackTime(player, 20);
  const nStr = track.tuning.strings;
  const capo = track.capo || 0;
  const W = 1100;
  const nut = 36;
  const scale = (W - nut - 10) / (1 - Math.pow(2, -FRETS / 12));
  const fx = (f: number) => nut + scale * (1 - Math.pow(2, -f / 12));
  const gap = 20;
  const H = (nStr - 1) * gap + 40;
  const sy = (s: number) => 20 + (s - 1) * gap;
  const mid = (f: number) => (f === 0 ? nut - 14 : (fx(f - 1) + fx(f)) / 2);
  const openPitch = (s: number) => track.tuning.notes[nStr - s] + capo;

  const sounding = useMemo(() => track.notes, [track.notes]);
  const active = sounding.filter((n) => n.string && n.start <= t && n.end > t);

  return (
    <div className="fretboard">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Fretboard">
        <rect x={nut} y={10} width={W - nut - 10} height={H - 20} fill="#2a1d12" rx={3} />
        {INLAYS.map((f) => <circle key={f} cx={mid(f)} cy={H / 2} r={5} fill="#5a4630" />)}
        <circle cx={mid(12)} cy={H / 2 - gap} r={5} fill="#5a4630" />
        <circle cx={mid(12)} cy={H / 2 + gap} r={5} fill="#5a4630" />
        <rect x={nut - 4} y={10} width={5} height={H - 20} fill="#d6cbb5" />
        {Array.from({ length: FRETS }, (_, i) => (
          <g key={i}>
            <line x1={fx(i + 1)} x2={fx(i + 1)} y1={10} y2={H - 10} stroke={i + 1 === capo ? "var(--accent)" : "#9a9a9a"} strokeWidth={i + 1 === capo ? 4 : 1.6} />
            {[1, 3, 5, 7, 9, 12, 15, 17, 19, 21, 24].includes(i + 1) && <text x={mid(i + 1)} y={H - 1} fill="var(--muted)" fontSize={9} textAnchor="middle">{i + 1}</text>}
          </g>
        ))}
        {Array.from({ length: nStr }, (_, k) => (
          <g key={k}>
            <line x1={nut} x2={W - 10} y1={sy(k + 1)} y2={sy(k + 1)} stroke="#c9c2b3" strokeWidth={0.8 + k * 0.28} opacity={0.85} />
            <text x={6} y={sy(k + 1) + 4} fill="var(--text-2)" fontSize={10} fontFamily="var(--mono)">{midiName(openPitch(k + 1)).replace(/-?\d+$/, "")}</text>
          </g>
        ))}
        {/* click targets */}
        {Array.from({ length: nStr }, (_, k) =>
          Array.from({ length: FRETS + 1 }, (_, f) => (
            <rect key={`${k}-${f}`} x={f === 0 ? nut - 26 : fx(f - 1)} y={sy(k + 1) - gap / 2} width={f === 0 ? 24 : fx(f) - fx(f - 1)} height={gap}
              fill="transparent" style={{ cursor: selected ? "pointer" : "default" }}
              onClick={() => selected && onPick(k + 1, f)}>
              <title>{`string ${k + 1}, fret ${f}: ${midiName(openPitch(k + 1) + f)}`}</title>
            </rect>
          )),
        )}
        {selected?.alternatives?.map((a) => (
          <g key={`alt-${a.string}-${a.fret}`} style={{ cursor: "pointer" }} onClick={() => onPick(a.string, a.fret)}>
            <circle cx={mid(a.fret)} cy={sy(a.string)} r={8} fill="none" stroke="var(--cyan)" strokeDasharray="3 2" strokeWidth={1.5} />
            <title>{`Alternative: string ${a.string} fret ${a.fret} (cost +${a.cost_delta.toFixed(2)})`}</title>
          </g>
        ))}
        {active.map((n) => (
          <g key={n.id} pointerEvents="none">
            <circle cx={mid(n.fret!)} cy={sy(n.string!)} r={8.5} fill={n.level === "low" ? "var(--bad)" : n.level === "medium" ? "var(--warn)" : "var(--accent)"} />
            <text x={mid(n.fret!)} y={sy(n.string!) + 3.5} fontSize={9.5} fontWeight={700} textAnchor="middle" fill="#1a1205">{n.fret}</text>
          </g>
        ))}
        {selected?.string && (
          <g pointerEvents="none">
            <circle cx={mid(selected.fret!)} cy={sy(selected.string)} r={10} fill="none" stroke="var(--cyan)" strokeWidth={2.5} />
          </g>
        )}
      </svg>
    </div>
  );
}
