import type { Project } from "../lib/types";

const STAGES = [
  { id: "queued", label: "Queued" },
  { id: "decode", label: "Decoding & validating audio" },
  { id: "separate", label: "Separating instruments" },
  { id: "instruments", label: "Identifying instruments & guitar parts" },
  { id: "rhythm", label: "Tempo, beats & time signature" },
  { id: "harmony", label: "Key & chords" },
  { id: "transcribe", label: "Transcribing notes, techniques, tuning & fingering" },
  { id: "tone", label: "Analysing tone & effects" },
  { id: "finalize", label: "Cross-validating & saving" },
];

export default function ProgressPanel({ project, onRetry }: { project: Project; onRetry: () => void }) {
  const st = project.status;
  const cur = STAGES.findIndex((s) => s.id === st.stage);
  const failed = st.state === "failed";
  return (
    <div className="grid cols-2">
      <div className="card col">
        <div className="card-head">
          <h3>{failed ? "Processing failed" : "Analysing your recording"}</h3>
          <span className="mono">{Math.round(st.progress * 100)}%</span>
        </div>
        <div className="bar"><div style={{ width: `${st.progress * 100}%` }} /></div>
        <div className="sub">{st.message}</div>
        <div className="stage-list" style={{ marginTop: 10 }}>
          {STAGES.map((s, i) => (
            <div key={s.id} className={`stage ${i < cur ? "done" : i === cur ? "cur" : ""}`}>
              {i < cur ? <span style={{ color: "var(--ok)" }}>✓</span> : i === cur && !failed ? <span className="spinner" /> : <span className="dot" style={{ background: "var(--panel-3)" }} />}
              {s.label}
            </div>
          ))}
        </div>
        {failed && (
          <>
            <div className="notice bad">{st.message}</div>
            <button className="primary" onClick={onRetry}>Retry</button>
          </>
        )}
      </div>
      <div className="card col">
        <h3>Processing log</h3>
        <div className="col mono small" style={{ maxHeight: 360, overflowY: "auto", color: "var(--text-2)" }}>
          {st.log.slice().reverse().map((l, i) => (
            <div key={i}><span className="muted">{new Date(l.t * 1000).toLocaleTimeString()}</span> [{l.stage}] {l.message}</div>
          ))}
        </div>
        <div className="sub">
          Source separation and transcription run in a background worker. On CPU, neural separation takes roughly 0.5–2× the song length.
        </div>
      </div>
    </div>
  );
}
