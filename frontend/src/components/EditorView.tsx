import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Player } from "../audio/engine";
import { playPitches } from "../audio/pluck";
import { api } from "../lib/api";
import { usePlayerState } from "../lib/hooks";
import type { Analysis, Note, ReviewItem, Track } from "../lib/types";
import Fretboard from "./Fretboard";
import NoteInspector from "./NoteInspector";
import ReviewPanel from "./ReviewPanel";
import TabView from "./TabView";
import TrackPanel from "./TrackPanel";
import type { TrackSummary } from "./Workspace";

interface Props {
  id: string;
  analysis: Analysis;
  tracks: TrackSummary[];
  trackId: string | null;
  setTrackId: (t: string) => void;
  player: Player;
  notify: (m: string, bad?: boolean) => void;
  onTrackChanged: (t: Track) => void;
}

export default function EditorView({ id, analysis, tracks, trackId, setTrackId, player, notify, onTrackChanged }: Props) {
  const [track, setTrack] = useState<Track | null>(null);
  const [selId, setSelId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [follow, setFollow] = useState(true);
  const [compare, setCompare] = useState<string | null>(null);
  const [scrollTo, setScrollTo] = useState<string | null>(null);
  const [autoplay, setAutoplay] = useState(true);
  const fretBuf = useRef<{ s: string; t: number }>({ s: "", t: 0 });
  const st = usePlayerState(player);

  useEffect(() => {
    if (!trackId) return;
    setTrack(null);
    setSelId(null);
    api.track(id, trackId).then(setTrack).catch((e) => notify(e.message, true));
  }, [id, trackId, notify]);

  const sel = useMemo(() => track?.notes.find((n) => n.id === selId) ?? null, [track, selId]);
  const ordered = useMemo(() => (track ? [...track.notes].filter((n) => n.string).sort((a, b) => a.start - b.start || b.pitch - a.pitch) : []), [track]);

  const applyTrack = useCallback(async (p: Promise<Track>, msg?: string) => {
    setBusy(true);
    try {
      const t = await p;
      setTrack(t);
      onTrackChanged(t);
      if (msg) notify(msg);
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setBusy(false);
    }
  }, [notify, onTrackChanged]);

  const edit = useCallback(async (body: Record<string, unknown>, refinger = false) => {
    if (!track || !sel) return;
    await applyTrack(api.editNote(id, track.id, sel.id, body, refinger));
  }, [track, sel, id, applyTrack]);

  const remove = useCallback(async () => {
    if (!track || !sel) return;
    const idx = ordered.findIndex((n) => n.id === sel.id);
    const next = ordered[idx + 1] ?? ordered[idx - 1];
    await applyTrack(api.deleteNote(id, track.id, sel.id), "Note deleted");
    setSelId(next?.id ?? null);
  }, [track, sel, ordered, id, applyTrack]);

  const select = useCallback((n: Note, opts?: { seek?: boolean }) => {
    setSelId(n.id);
    if (opts?.seek) player.seek(n.start);
    else if (autoplay && !player.state.playing) playPitches([n.pitch], { seconds: 1, palmMute: n.techniques.some((t) => t.type === "palm_mute") });
  }, [player, autoplay]);

  const jump = (r: ReviewItem) => {
    const n = track?.notes.find((x) => x.id === r.note_id);
    if (!n) return;
    setSelId(n.id);
    setScrollTo(n.id);
    player.seek(Math.max(0, n.start - 0.5));
  };

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        player.toggle();
        return;
      }
      if (e.key === "l" || e.key === "L") return player.toggleLoop();
      if (e.key === "Escape") return setSelId(null);
      if (!track) return;
      const idx = sel ? ordered.findIndex((n) => n.id === sel.id) : -1;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const n = ordered[Math.max(0, Math.min(ordered.length - 1, idx + (e.key === "ArrowRight" ? 1 : -1)))];
        if (n) {
          select(n);
          setScrollTo(n.id);
        }
        return;
      }
      if (!sel || busy) return;
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        const nStr = track.tuning.strings;
        const capo = track.capo || 0;
        let s = sel.string ?? 1;
        for (let k = 0; k < nStr; k++) {
          s += e.key === "ArrowUp" ? -1 : 1;
          if (s < 1 || s > nStr) break;
          const f = sel.pitch - (track.tuning.notes[nStr - s] + capo);
          if (f >= 0 && f <= track.max_fret) {
            void edit({ string: s });
            break;
          }
        }
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        void remove();
        return;
      }
      if (e.key === "+" || e.key === "=") return void edit({ pitch: sel.pitch + 1 }, true);
      if (e.key === "-") return void edit({ pitch: sel.pitch - 1 }, true);
      if (/^[0-9]$/.test(e.key) && sel.string) {
        const now = performance.now();
        const buf = now - fretBuf.current.t < 700 ? fretBuf.current.s + e.key : e.key;
        fretBuf.current = { s: buf, t: now };
        const f = Number(buf);
        if (f <= track.max_fret) void edit({ string: sel.string, fret: f });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [player, track, sel, ordered, busy, edit, remove, select]);

  if (!tracks.length) return <div className="notice">No transcribed parts yet. Open the Analysis tab to transcribe a stem.</div>;
  const changes = compare && track ? track.alternative_fingerings?.[compare]?.changes ?? null : null;
  const stemSource = st.sources.find((s) => s.id === track?.stem);
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="row">
        <div className="chips">
          {tracks.map((t) => (
            <button key={t.id} className={`chip ${t.id === trackId ? "on" : ""}`} onClick={() => setTrackId(t.id)}>
              {t.name} <span className="muted">· {t.stats.notes}</span>
            </button>
          ))}
        </div>
        <div className="grow" />
        {stemSource && (
          <button className={`sm ${stemSource.solo ? "active" : ""}`} onClick={() => player.setSourceProp(stemSource.id, { solo: !stemSource.solo })}>
            {stemSource.solo ? "Soloing" : "Solo"} {stemSource.label} stem
          </button>
        )}
        <label className="check small"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow playback</label>
        <label className="check small"><input type="checkbox" checked={autoplay} onChange={(e) => setAutoplay(e.target.checked)} /> Hear notes on click</label>
        {busy && <span className="spinner" />}
      </div>
      {!track ? (
        <div><span className="spinner" /> Loading tab…</div>
      ) : (
        <>
          <div className="row small">
            <span className="pill high">{track.stats.levels.high} high</span>
            <span className="pill medium">{track.stats.levels.medium} medium</span>
            <span className="pill low">{track.stats.levels.low} low</span>
            {track.stats.levels.user > 0 && <span className="pill user">{track.stats.levels.user} edited</span>}
            <span className="sub">{track.tuning.name}{track.capo ? ` · capo ${track.capo}` : ""} · {track.grid.time_signature.join("/")} · {Object.entries(track.stats.techniques).map(([k, v]) => `${v} ${k.replace(/_/g, " ")}`).join(", ") || "no techniques detected"}</span>
            {compare && <span className="pill probable">Comparing: {track.alternative_fingerings[compare]?.label} (cyan = alternative string:fret)</span>}
          </div>
          {track.part_split?.adopted && (
            <div className="notice">This stem contained two overlapping guitar parts; they were split into lead and rhythm tracks ({Math.round(track.part_split.confidence * 100)}% confidence).</div>
          )}
          <div className="editor">
            <div className="col" style={{ gap: 14, minWidth: 0 }}>
              <TabView track={track} chords={analysis.chords} player={player} selectedId={selId} onSelect={select} compare={changes} follow={follow} scrollTo={scrollTo} />
              <Fretboard track={track} player={player} selected={sel} onPick={(s, f) => edit({ string: s, fret: f }, true)} />
            </div>
            <div className="side">
              <NoteInspector track={track} note={sel} edit={edit} remove={remove} busy={busy} />
              <ReviewPanel track={track} onJump={jump} />
              <TrackPanel id={id} track={track} apply={applyTrack} compare={compare} setCompare={setCompare} busy={busy} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
