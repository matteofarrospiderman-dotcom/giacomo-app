import { useState } from "react";
import { api } from "../lib/api";
import type { Analysis } from "../lib/types";
import type { TrackSummary } from "./Workspace";

const FORMATS = [
  { fmt: "gp5", label: "Guitar Pro 5 (.gp5)", desc: "Opens in Guitar Pro 5–8, TuxGuitar, MuseScore, Songsterr. Keeps tunings, techniques, ties and all tracks." },
  { fmt: "musicxml", label: "MusicXML", desc: "Standard notation + TAB staves with string/fret for MuseScore, Finale, Sibelius, Dorico." },
  { fmt: "midi", label: "MIDI (performance timing)", desc: "One track per part, tempo map follows the recording, bends as pitch-wheel." },
  { fmt: "midi-q", label: "MIDI (quantised)", desc: "Constant tempo, notes snapped to the rhythmic grid – best for DAWs." },
  { fmt: "midi-tracks", label: "Individual MIDI tracks (.zip)", desc: "One file per instrument, one MIDI channel per string." },
  { fmt: "pdf", label: "PDF tablature", desc: "Printable tab with rhythm, chords and technique marks; uncertain notes in grey." },
  { fmt: "txt", label: "Text tab (.txt)", desc: "Ultimate-Guitar style ASCII tab." },
];

export default function ExportView({ id, analysis, tracks }: { id: string; analysis: Analysis; tracks: TrackSummary[] }) {
  const [sel, setSel] = useState<string[]>(tracks.map((t) => t.id));
  const url = (fmt: string) => (fmt === "midi-q" ? api.exportUrl(id, "midi", sel, true) : api.exportUrl(id, fmt, sel));
  return (
    <div className="grid cols-2">
      <div className="card col">
        <h3>Tracks to include</h3>
        {tracks.map((t) => (
          <label key={t.id} className="check">
            <input type="checkbox" checked={sel.includes(t.id)} onChange={(e) => setSel(e.target.checked ? [...sel, t.id] : sel.filter((x) => x !== t.id))} />
            {t.name} <span className="muted">({t.instrument}, {t.stats.notes} notes)</span>
          </label>
        ))}
        <div className="sep" />
        {FORMATS.map((f) => (
          <div key={f.fmt} className="row between" style={{ padding: "6px 0" }}>
            <div className="col grow" style={{ gap: 0 }}>
              <strong>{f.label}</strong>
              <span className="sub">{f.desc}</span>
            </div>
            <a className={`btn ${f.fmt === "gp5" ? "primary" : ""}`} href={sel.length ? url(f.fmt) : undefined} aria-disabled={!sel.length} download>Download</a>
          </div>
        ))}
        <div className="sub">GPX/GP (Guitar Pro 6–8 native) is a closed format with no reliable open writer; Guitar Pro 7/8 import GP5 losslessly for everything this app produces.</div>
      </div>
      <div className="card col">
        <h3>Audio stems (WAV)</h3>
        <a className="btn primary" href={api.exportUrl(id, "stems")} download style={{ alignSelf: "flex-start" }}>Download all stems (.zip)</a>
        {Object.values(analysis.stems).map((s) => (
          <div key={s.name} className="row between">
            <span>{s.label}</span>
            <div className="row">
              <a className="btn sm" href={api.fileUrl(id, s.wav)} download={`${s.name}.wav`}>WAV</a>
              <a className="btn sm" href={api.fileUrl(id, s.preview)} download>MP3</a>
            </div>
          </div>
        ))}
        <div className="sep" />
        <div className="sub">
          Separation: {analysis.separation.engine} ({analysis.separation.model}). Stems are 44.1 kHz / 16-bit and sum back approximately to the original mix.
        </div>
      </div>
    </div>
  );
}
