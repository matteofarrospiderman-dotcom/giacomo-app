import { useState } from "react";
import { playPitches } from "../audio/pluck";
import { ENGINE_LABELS, FLAG_LABELS, TECHNIQUES, fmtTime, midiName, pct } from "../lib/format";
import type { Note, Technique, Track } from "../lib/types";

interface Props {
  track: Track;
  note: Note | null;
  edit: (body: Record<string, unknown>, refinger?: boolean) => Promise<void>;
  remove: () => Promise<void>;
  busy: boolean;
}

export default function NoteInspector({ track, note, edit, remove, busy }: Props) {
  const [bend, setBend] = useState(2);
  if (!note) {
    return (
      <div className="card">
        <h3 style={{ marginBottom: 8 }}>Note inspector</h3>
        <div className="sub">
          Click a note in the tab to inspect and edit it. Shortcuts: <span className="kbd">Space</span> play ·{" "}
          <span className="kbd">←</span>/<span className="kbd">→</span> prev/next note · <span className="kbd">↑</span>/<span className="kbd">↓</span> move string ·{" "}
          <span className="kbd">0-9</span> type fret · <span className="kbd">+</span>/<span className="kbd">-</span> semitone · <span className="kbd">Del</span> delete ·{" "}
          <span className="kbd">L</span> loop on/off
        </div>
      </div>
    );
  }
  const nStr = track.tuning.strings;
  const capo = track.capo || 0;
  const openPitch = (s: number) => track.tuning.notes[nStr - s] + capo;
  const has = (type: string) => note.techniques.some((t) => t.type === type);
  const toggleTech = (type: string) => {
    let techs: Technique[] = note.techniques.filter((t) => t.type !== type);
    if (!has(type)) {
      const t: Technique = { type, confidence: 1, user: true };
      if (type.includes("bend")) t.semitones = bend;
      if (type === "slide") t.direction = "up";
      techs = [...techs, t];
    }
    void edit({ techniques: techs });
  };
  const stringOptions = Array.from({ length: nStr }, (_, k) => k + 1).filter((s) => {
    const f = note.pitch - openPitch(s);
    return f >= 0 && f <= track.max_fret;
  });
  return (
    <div className="card col">
      <div className="row between">
        <h3>Note {note.name}</h3>
        <span className={`pill ${note.level}`}>{note.level === "user" ? "edited" : `${note.level} · ${pct(note.confidence)}`}</span>
      </div>
      <div className="row">
        <button className="sm" onClick={() => playPitches([note.pitch], { palmMute: has("palm_mute") })}>▶ Hear note</button>
        <button className="sm" onClick={() => {
          const chord = track.notes.filter((n) => n.group === note.group && n.group !== undefined).map((n) => n.pitch);
          playPitches(chord.length ? chord : [note.pitch]);
        }}>▶ Hear chord</button>
        <span className="sub mono">{fmtTime(note.start)} → {fmtTime(note.end)}</span>
      </div>
      <div className="row">
        <label className="field">String
          <select value={note.string ?? ""} disabled={busy} onChange={(e) => edit({ string: Number(e.target.value) })}>
            {note.string == null && <option value="">–</option>}
            {stringOptions.map((s) => <option key={s} value={s}>{s} ({midiName(openPitch(s)).replace(/-?\d+$/, "")}) · fret {note.pitch - openPitch(s)}</option>)}
          </select>
        </label>
        <label className="field">Fret
          <input className="fret-input" type="number" min={0} max={track.max_fret} value={note.fret ?? ""} disabled={busy || !note.string}
            onChange={(e) => e.target.value !== "" && edit({ string: note.string, fret: Number(e.target.value) })} />
        </label>
        <label className="field">Pitch
          <div className="row" style={{ gap: 4 }}>
            <button className="sm" disabled={busy} onClick={() => edit({ pitch: note.pitch - 12 }, true)} title="Octave down">−8va</button>
            <button className="sm" disabled={busy} onClick={() => edit({ pitch: note.pitch - 1 }, true)}>−1</button>
            <button className="sm" disabled={busy} onClick={() => edit({ pitch: note.pitch + 1 }, true)}>+1</button>
            <button className="sm" disabled={busy} onClick={() => edit({ pitch: note.pitch + 12 }, true)} title="Octave up">+8va</button>
          </div>
        </label>
      </div>
      <div className="row">
        <span className="sub">Timing</span>
        <button className="sm" disabled={busy} onClick={() => edit({ start: Math.max(0, note.start - 0.02), end: note.end - 0.02 })}>◀ 20ms</button>
        <button className="sm" disabled={busy} onClick={() => edit({ start: note.start + 0.02, end: note.end + 0.02 })}>20ms ▶</button>
        <button className="sm" disabled={busy || note.end - note.start < 0.08} onClick={() => edit({ end: note.end - 0.05 })}>shorter</button>
        <button className="sm" disabled={busy} onClick={() => edit({ end: note.end + 0.05 })}>longer</button>
      </div>

      {note.alternatives && note.alternatives.length > 0 && (
        <div className="col" style={{ gap: 4 }}>
          <span className="sub">Alternative positions (cost relative to chosen)</span>
          <div className="chips">
            {note.alternatives.map((a) => (
              <button key={`${a.string}-${a.fret}`} className="chip" disabled={busy} onClick={() => edit({ string: a.string, fret: a.fret }, true)}
                title={a.shape ? `Whole shape: ${a.shape.map(([s, f]) => `${s}:${f}`).join(" ")}` : undefined}>
                str {a.string} · fret {a.fret} <span className="muted">+{a.cost_delta.toFixed(2)}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="col" style={{ gap: 4 }}>
        <div className="row between">
          <span className="sub">Techniques</span>
          <label className="sub row" style={{ gap: 4 }}>bend
            <select value={bend} onChange={(e) => setBend(Number(e.target.value))}>
              <option value={1}>½</option><option value={2}>full</option><option value={3}>1½</option>
            </select>
          </label>
        </div>
        <div className="chips">
          {TECHNIQUES.map((t) => {
            const det = note.techniques.find((x) => x.type === t.type);
            return (
              <button key={t.type} className={`chip ${det ? "on" : ""}`} disabled={busy} onClick={() => toggleTech(t.type)}
                title={det?.confidence != null && !det.user ? `detected with ${pct(det.confidence as number)} confidence` : undefined}>
                {t.label}{det && !det.user && det.confidence != null ? ` ${pct(det.confidence as number)}` : ""}
              </button>
            );
          })}
          <button className={`chip ${note.dead ? "on" : ""}`} disabled={busy} onClick={() => edit({ dead: !note.dead })}>Dead (x)</button>
        </div>
      </div>

      <div className="col" style={{ gap: 4 }}>
        <span className="sub">Evidence</span>
        <div className="chips">
          {Object.entries(note.sources).filter(([, v]) => typeof v === "number").map(([k, v]) => (
            <span key={k} className="pill">{ENGINE_LABELS[k] ?? k}: {pct(v as number)}</span>
          ))}
        </div>
        {note.flags.length > 0 && (
          <div className="chips">{note.flags.map((f) => <span key={f} className="pill warn">{FLAG_LABELS[f] ?? f}</span>)}</div>
        )}
      </div>
      <div className="row between">
        <button className="sm" disabled={busy} onClick={() => edit({ locked: !note.locked })}>{note.locked ? "🔒 Locked" : "Lock position"}</button>
        <button className="sm danger" disabled={busy} onClick={() => remove()}>Delete note</button>
      </div>
    </div>
  );
}
