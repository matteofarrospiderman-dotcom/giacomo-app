import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { navigate } from "../App";
import { Player } from "../audio/engine";
import { api } from "../lib/api";
import { fmtTime } from "../lib/format";
import type { Analysis, Project, Track } from "../lib/types";
import AnalysisView from "./AnalysisView";
import EditorView from "./EditorView";
import ExportView from "./ExportView";
import ProgressPanel from "./ProgressPanel";
import ToneView from "./ToneView";
import Transport from "./Transport";

const TABS = [
  { id: "analysis", label: "Analysis" },
  { id: "editor", label: "Tab Editor" },
  { id: "tone", label: "Tone Reconstruction" },
  { id: "export", label: "Export" },
];

export type TrackSummary = { id: string; name: string; instrument: string; role: string; stats: Track["stats"] };

export default function Workspace({ id, tab }: { id: string; tab?: string }) {
  const [project, setProject] = useState<Project | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [tracks, setTracks] = useState<TrackSummary[]>([]);
  const [trackId, setTrackId] = useState<string | null>(null);
  const [toast, setToast] = useState<{ msg: string; bad?: boolean } | null>(null);
  const [notFound, setNotFound] = useState(false);
  const player = useMemo(() => new Player(), []);
  const loadedFor = useRef<string | null>(null);
  const active = tab && TABS.some((t) => t.id === tab) ? tab : "editor";

  useEffect(() => () => player.destroy(), [player]);

  const notify = useCallback((msg: string, bad = false) => {
    setToast({ msg, bad });
    window.setTimeout(() => setToast(null), 3500);
  }, []);

  const loadResults = useCallback(async () => {
    const [a, ts] = await Promise.all([api.analysis(id), api.tracks(id)]);
    setAnalysis(a);
    setTracks(ts);
    setTrackId((cur) => (cur && ts.some((t) => t.id === cur) ? cur : ts.find((t) => t.instrument === "guitar")?.id ?? ts[0]?.id ?? null));
    if (loadedFor.current !== id + a.processing_seconds) {
      loadedFor.current = id + a.processing_seconds;
      const sources = [{ id: "mix", label: "Original mix", url: api.fileUrl(id, a.mix.preview) }];
      for (const s of Object.values(a.stems)) sources.push({ id: s.name, label: s.label, url: api.fileUrl(id, s.preview) });
      const defaults: Record<string, { muted: boolean }> = {};
      for (const s of sources) defaults[s.id] = { muted: s.id !== "mix" };
      player.load(sources, defaults);
      player.beats = a.rhythm.beats;
      player.downbeats = new Set(a.rhythm.downbeats.map((t) => Math.round(t * 1000)));
    }
  }, [id, player]);

  // poll project status while processing; load results once done
  const [pollKey, setPollKey] = useState(0);
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const tick = async () => {
      try {
        const p = await api.project(id);
        if (!alive) return;
        setProject(p);
        if (p.status.state === "done") {
          await loadResults();
          return;
        }
        if (p.status.state === "failed") return;
      } catch (e) {
        if ((e as { status?: number }).status === 404) {
          setNotFound(true);
          return;
        }
      }
      timer = window.setTimeout(tick, 1500);
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id, pollKey, loadResults]);

  const restartPolling = () => {
    setProject((p) => (p ? { ...p, status: { ...p.status, state: "queued", progress: 0, message: "Queued" } } : p));
    setPollKey((k) => k + 1);
  };

  if (notFound) return <main className="page"><div className="notice bad">Project not found. <a href="#/">Back to projects</a></div></main>;
  if (!project) return <main className="page"><span className="spinner" /> Loading…</main>;

  const processing = project.status.state !== "done";
  return (
    <main className="page">
      <div className="ws-head">
        <h2>{project.name}</h2>
        {project.summary && (
          <span className="sub">
            {fmtTime(project.summary.duration)} · {project.summary.bpm.toFixed(0)} BPM · {project.summary.time_signature.join("/")} · {project.summary.key} · {project.summary.tuning}
          </span>
        )}
        <div className="grow" />
        {!processing && (
          <button
            className="sm"
            onClick={async () => {
              if (!confirm("Re-run the full analysis? Manual edits to tracks will be replaced.")) return;
              await api.reprocess(id, {});
              restartPolling();
            }}
          >
            Re-analyse
          </button>
        )}
      </div>
      {processing ? (
        <ProgressPanel project={project} onRetry={async () => { await api.reprocess(id, {}); restartPolling(); }} />
      ) : analysis ? (
        <>
          <Transport player={player} analysis={analysis} />
          <nav className="tabs">
            {TABS.map((t) => (
              <button key={t.id} className={active === t.id ? "on" : ""} onClick={() => navigate(`/project/${id}/${t.id}`)}>
                {t.label}
              </button>
            ))}
          </nav>
          {active === "analysis" && (
            <AnalysisView id={id} analysis={analysis} tracks={tracks} player={player} notify={notify}
              onTranscribe={async (stem, instrument) => {
                try {
                  await api.transcribe(id, stem, instrument);
                  notify(`Transcribing ${stem} as ${instrument}…`);
                  restartPolling();
                } catch (e) {
                  notify((e as Error).message, true);
                }
              }}
              openTrack={(tid) => { setTrackId(tid); navigate(`/project/${id}/editor`); }} />
          )}
          {active === "editor" && (
            <EditorView id={id} analysis={analysis} tracks={tracks} trackId={trackId} setTrackId={setTrackId} player={player} notify={notify}
              onTrackChanged={(t) => setTracks((ts) => ts.map((x) => (x.id === t.id ? { ...x, name: t.name, stats: t.stats } : x)))} />
          )}
          {active === "tone" && <ToneView id={id} analysis={analysis} notify={notify} />}
          {active === "export" && <ExportView id={id} analysis={analysis} tracks={tracks} />}
        </>
      ) : (
        <div><span className="spinner" /> Loading results…</div>
      )}
      {toast && <div className={`toast ${toast.bad ? "bad" : ""}`}>{toast.msg}</div>}
    </main>
  );
}
