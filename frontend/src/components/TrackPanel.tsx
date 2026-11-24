import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { fmtTime, midiName, pct } from "../lib/format";
import type { Track, Tuning } from "../lib/types";

interface Props {
  id: string;
  track: Track;
  apply: (p: Promise<Track>, msg: string) => Promise<void>;
  compare: string | null;
  setCompare: (p: string | null) => void;
  busy: boolean;
}

export default function TrackPanel({ id, track, apply, compare, setCompare, busy }: Props) {
  const [tunings, setTunings] = useState<Tuning[]>([]);
  const [num, setNum] = useState(track.grid.time_signature[0]);
  const [den, setDen] = useState(track.grid.time_signature[1]);
  const [showRejected, setShowRejected] = useState(false);
  useEffect(() => {
    api.tunings().then(setTunings).catch(() => undefined);
  }, []);
  useEffect(() => {
    setNum(track.grid.time_signature[0]);
    setDen(track.grid.time_signature[1]);
  }, [track.grid.time_signature]);
  const det = track.tuning_detection;
  const set = (body: Record<string, unknown>, msg: string) => apply(api.trackSettings(id, track.id, body), msg);
  return (
    <div className="card col">
      <h3>Track settings</h3>
      <label className="field">Tuning {det && <span className="muted">(detected: {det.best.name}, {pct(det.confidence)})</span>}
        <select value={track.tuning.id} disabled={busy} onChange={(e) => set({ tuning: e.target.value }, "Re-fingered for new tuning")}>
          {track.tuning.id === "custom" && <option value="custom">{track.tuning.name}</option>}
          {tunings.filter((t) => t.instrument === track.instrument).map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </label>
      <div className="sub mono">{track.tuning.notes.map((n) => midiName(n)).join(" ")}</div>
      <div className="row">
        <label className="field">Capo
          <select value={track.capo} disabled={busy} onChange={(e) => set({ capo: Number(e.target.value) }, "Capo updated")}>
            {Array.from({ length: 10 }, (_, i) => <option key={i} value={i}>{i === 0 ? "none" : i}</option>)}
          </select>
        </label>
        <label className="field">Time signature
          <div className="row" style={{ gap: 4 }}>
            <input type="number" min={1} max={16} value={num} onChange={(e) => setNum(Number(e.target.value))} style={{ width: 52 }} />
            <span>/</span>
            <select value={den} onChange={(e) => setDen(Number(e.target.value))}>
              {[2, 4, 8, 16].map((d) => <option key={d}>{d}</option>)}
            </select>
            <button className="sm" disabled={busy} onClick={() => set({ time_signature: [num, den] }, "Time signature changed")}>Apply</button>
          </div>
        </label>
      </div>
      <div className="row">
        <span className="sub">Bar lines</span>
        <button className="sm" disabled={busy} onClick={() => set({ downbeat_offset: (track.downbeat_offset ?? 0) + 1 }, "Downbeat shifted")}>Shift downbeat +1 beat</button>
      </div>
      <div className="sep" />
      <h3>Fingering alternatives</h3>
      <div className="sub">Current: {track.fingering.profile_label} · avg fret {track.playability.mean_fret} · {track.playability.large_jumps} big jumps · {track.fingering.uncertain_groups} uncertain spots</div>
      {Object.entries(track.alternative_fingerings ?? {}).map(([k, v]) => (
        <div key={k} className="row between">
          <div className="col" style={{ gap: 0 }}>
            <strong style={{ fontSize: 13 }}>{v.label}</strong>
            <span className="sub">{v.n_changed} notes differ · avg fret {v.report.mean_fret} · open {pct(v.report.open_string_ratio)}</span>
          </div>
          <div className="row" style={{ gap: 4 }}>
            <button className={`sm ${compare === k ? "active" : ""}`} onClick={() => setCompare(compare === k ? null : k)}>Compare</button>
            <button className="sm" disabled={busy || v.n_changed === 0} onClick={() => set({ profile: k }, `Switched to ${v.label}`)}>Use</button>
          </div>
        </div>
      ))}
      <button disabled={busy} onClick={() => apply(api.reoptimize(id, track.id), "Re-optimised around your locked notes")}>
        Re-optimise fingering around my edits
      </button>
      <div className="sep" />
      <div className="row between">
        <h3>Rejected candidates ({track.rejected?.length ?? 0})</h3>
        <button className="sm ghost" onClick={() => setShowRejected(!showRejected)}>{showRejected ? "Hide" : "Show"}</button>
      </div>
      {showRejected && (
        <div className="col" style={{ maxHeight: 200, overflowY: "auto", gap: 2 }}>
          {track.rejected.slice(0, 200).map((r, i) => (
            <div key={i} className="row between small">
              <span className="mono">{fmtTime(r.start)} {midiName(r.pitch)}</span>
              <span className="muted">{r.reason.replace(/_/g, " ")} · {pct(r.confidence)}</span>
              <button className="sm" disabled={busy} onClick={() => apply(api.restore(id, track.id, i), "Note restored")}>Restore</button>
            </div>
          ))}
        </div>
      )}
      <div className="sub">Engines: {Object.entries(track.engines).map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>
    </div>
  );
}
