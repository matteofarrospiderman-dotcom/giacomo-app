import { useEffect, useRef, useState } from "react";
import { navigate } from "../App";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";
import type { Health, Project, Tuning } from "../lib/types";

const STATE_PILL: Record<string, string> = { done: "ok", failed: "bad", processing: "warn", queued: "warn", uploaded: "warn" };

export default function Dashboard() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [tunings, setTunings] = useState<Tuning[]>([]);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opts, setOpts] = useState({ separator: "auto", tuning: "auto", strings: "", transcribe_bass: true, split_guitars: true });
  const input = useRef<HTMLInputElement>(null);

  const refresh = () => api.projects().then(setProjects).catch(() => undefined);
  useEffect(() => {
    void refresh();
    api.health().then(setHealth).catch(() => setError("Backend not reachable – is the API running on port 8000?"));
    api.tunings().then(setTunings).catch(() => undefined);
    const h = setInterval(refresh, 3000);
    return () => clearInterval(h);
  }, []);

  const upload = async (file: File) => {
    setError(null);
    setUploading(0);
    try {
      const p = await api.upload(file, { ...opts, strings: opts.strings || null }, setUploading);
      navigate(`/project/${p.id}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(null);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) void upload(f);
  };

  const eng = health?.engines;
  return (
    <main className="page">
      <section className="hero">
        <div
          className={`dropzone ${over ? "over" : ""}`}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && input.current?.click()}
        >
          <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.6" aria-hidden>
            <path d="M9 18V5l12-2v13" />
            <circle cx="6" cy="18" r="3" />
            <circle cx="18" cy="16" r="3" />
          </svg>
          <div className="big">{uploading !== null ? `Uploading… ${Math.round(uploading * 100)}%` : "Drop a song or guitar recording"}</div>
          <div className="sub">MP3, WAV, FLAC, OGG, M4A, AIFF… up to {health?.limits.max_upload_mb ?? 250} MB / {Math.round((health?.limits.max_duration_s ?? 900) / 60)} min</div>
          <button className="primary" onClick={(e) => { e.stopPropagation(); input.current?.click(); }} disabled={uploading !== null}>
            Choose file
          </button>
          <input ref={input} type="file" accept="audio/*,video/mp4,video/webm,.flac,.aif,.aiff,.opus" hidden onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </div>
        <div className="card col">
          <h3>Analysis options</h3>
          <label className="field">
            Source separation
            <select value={opts.separator} onChange={(e) => setOpts({ ...opts, separator: e.target.value })}>
              <option value="auto">Automatic (Demucs if available)</option>
              <option value="demucs">Demucs neural separation (best)</option>
              <option value="dsp">DSP approximation (fast, lower quality)</option>
            </select>
          </label>
          <div className="row">
            <label className="field grow">
              Guitar tuning
              <select value={opts.tuning} onChange={(e) => setOpts({ ...opts, tuning: e.target.value })}>
                <option value="auto">Detect automatically</option>
                {tunings.filter((t) => t.instrument === "guitar").map((t) => (
                  <option key={t.id} value={t.id}>{t.name}</option>
                ))}
              </select>
            </label>
            <label className="field">
              Strings
              <select value={opts.strings} onChange={(e) => setOpts({ ...opts, strings: e.target.value })}>
                <option value="">Auto</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
              </select>
            </label>
          </div>
          <label className="check"><input type="checkbox" checked={opts.split_guitars} onChange={(e) => setOpts({ ...opts, split_guitars: e.target.checked })} /> Separate multiple guitars (stereo position + lead/rhythm)</label>
          <label className="check"><input type="checkbox" checked={opts.transcribe_bass} onChange={(e) => setOpts({ ...opts, transcribe_bass: e.target.checked })} /> Also transcribe bass</label>
          <div className="sep" />
          <h3>Engines</h3>
          {health ? (
            <div className="chips">
              <span className={`pill ${eng?.basic_pitch?.available ? "ok" : "bad"}`}>Basic Pitch {eng?.basic_pitch?.available ? "✓" : "✗"}</span>
              <span className={`pill ${eng?.demucs?.installed ? "ok" : "warn"}`}>Demucs {eng?.demucs?.installed ? "✓" : "not installed"}</span>
              <span className={`pill ${eng?.crepe?.installed ? "ok" : "warn"}`}>CREPE {eng?.crepe?.installed ? "✓" : "→ pYIN"}</span>
              <span className={`pill ${health.ffmpeg ? "ok" : "bad"}`}>FFmpeg {health.ffmpeg ? "✓" : "✗"}</span>
              <span className="pill">Device: {health.device}</span>
            </div>
          ) : (
            <div className="sub">Checking backend…</div>
          )}
          {error && <div className="notice bad">{error}</div>}
        </div>
      </section>

      <div className="row between" style={{ marginBottom: 12 }}>
        <h2>Recent projects</h2>
        <span className="sub">{projects.length} project{projects.length === 1 ? "" : "s"}</span>
      </div>
      {projects.length === 0 ? (
        <div className="notice">No projects yet. Upload a recording to generate tabs, stems and a tone analysis.</div>
      ) : (
        <div className="projects">
          {projects.map((p) => (
            <article key={p.id} className="card project-card" onClick={() => navigate(`/project/${p.id}`)}>
              <div className="row between">
                <strong className="grow" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{p.name}</strong>
                <span className={`pill ${STATE_PILL[p.status.state] ?? ""}`}>{p.status.state}</span>
              </div>
              {p.status.state !== "done" && p.status.state !== "failed" && (
                <>
                  <div className="bar"><div style={{ width: `${p.status.progress * 100}%` }} /></div>
                  <div className="sub">{p.status.message}</div>
                </>
              )}
              {p.status.state === "failed" && <div className="sub" style={{ color: "var(--bad)" }}>{p.status.message}</div>}
              {p.summary && (
                <dl className="kv">
                  <dt>Length</dt><dd>{fmtTime(p.summary.duration)}</dd>
                  <dt>Tempo</dt><dd>{p.summary.bpm.toFixed(0)} BPM · {p.summary.time_signature.join("/")}</dd>
                  <dt>Key</dt><dd>{p.summary.key}</dd>
                  <dt>Tuning</dt><dd>{p.summary.tuning ?? "–"}</dd>
                  <dt>Tracks</dt><dd>{p.summary.tracks} · {p.summary.notes} notes · {p.summary.separation}</dd>
                </dl>
              )}
              <div className="row between">
                <span className="sub">{new Date(p.created_at * 1000).toLocaleString()}</span>
                <button
                  className="ghost danger sm"
                  onClick={async (e) => {
                    e.stopPropagation();
                    if (confirm(`Delete "${p.name}"?`)) {
                      await api.deleteProject(p.id).catch((err) => setError(err.message));
                      void refresh();
                    }
                  }}
                >
                  Delete
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </main>
  );
}
