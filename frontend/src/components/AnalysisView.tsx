import type { Player } from "../audio/engine";
import { pct } from "../lib/format";
import { usePlayerState } from "../lib/hooks";
import type { Analysis } from "../lib/types";
import type { TrackSummary } from "./Workspace";
import Waveform from "./Waveform";

function Conf({ v }: { v: number }) {
  const lvl = v >= 0.7 ? "high" : v >= 0.45 ? "medium" : "low";
  return <span className={`pill ${lvl}`}>{lvl} · {pct(v)}</span>;
}

interface Props {
  id: string;
  analysis: Analysis;
  tracks: TrackSummary[];
  player: Player;
  notify: (m: string, bad?: boolean) => void;
  onTranscribe: (stem: string, instrument: string) => void;
  openTrack: (tid: string) => void;
}

export default function AnalysisView({ id, analysis: a, tracks, player, onTranscribe, openTrack }: Props) {
  const st = usePlayerState(player);
  const dur = a.duration;
  const tun = a.tuning;
  return (
    <div className="col" style={{ gap: 16 }}>
      <div className="card">
        <div className="card-head">
          <h3>Waveform · beats · chords</h3>
          <span className="sub">Click to seek · drag to loop a section</span>
        </div>
        <Waveform id={id} peaksPath={a.mix.peaks} duration={dur} player={player} height={110} beats={a.rhythm.beats} downbeats={a.rhythm.downbeats} />
        <div className="chords-lane">
          {a.chords.filter((c) => c.name !== "N").map((c, i) => (
            <div key={i} className="chord-chip" title={`${c.name} · confidence ${pct(c.confidence)}${c.flag ? " · transcription disagrees" : ""}`}
              onClick={() => player.seek(c.start)}
              style={{ left: `${(c.start / dur) * 100}%`, width: `${Math.max(0.4, ((c.end - c.start) / dur) * 100)}%`, opacity: 0.45 + 0.55 * c.confidence, borderColor: c.flag ? "var(--warn)" : undefined }}>
              {c.name}
            </div>
          ))}
        </div>
        {a.chord_validation?.agreement != null && (
          <div className="sub" style={{ marginTop: 8 }}>
            Chords cross-checked against the guitar transcription: {pct(a.chord_validation.agreement)} agreement over {a.chord_validation.chords_checked} chords.
          </div>
        )}
      </div>

      <div className="grid cols-4">
        <div className="card">
          <h3>Tempo</h3>
          <div className="stat">{a.rhythm.bpm.toFixed(1)}<small>BPM</small></div>
          <div className="row"><Conf v={a.rhythm.tempo_confidence} /><span className="sub">stability {pct(a.rhythm.tempo_stability)}</span></div>
          <div className="sub" style={{ marginTop: 6 }}>Alternatives: {a.rhythm.tempo_candidates.slice(1, 4).map((c) => c.bpm.toFixed(0)).join(", ")} · from {a.rhythm.source}</div>
        </div>
        <div className="card">
          <h3>Time signature</h3>
          <div className="stat">{a.rhythm.time_signature.join("/")}</div>
          <div className="row"><Conf v={a.rhythm.meter_confidence} /><span className="sub">{a.rhythm.feel} feel</span></div>
          <div className="sub" style={{ marginTop: 6 }}>Change per track in the editor if wrong.</div>
        </div>
        <div className="card">
          <h3>Key</h3>
          <div className="stat" style={{ fontSize: 22 }}>{a.key.name}</div>
          <Conf v={a.key.confidence} />
          <div className="sub" style={{ marginTop: 6 }}>Also possible: {a.key.alternatives?.slice(0, 2).map((k) => k.name).join(", ")}</div>
        </div>
        <div className="card">
          <h3>Tuning</h3>
          <div className="stat" style={{ fontSize: 18 }}>{tun?.best.name ?? "–"}</div>
          {tun && <Conf v={tun.confidence} />}
          <div className="sub mono" style={{ marginTop: 6 }}>{tun?.best.note_names.join(" ")}</div>
          {tun?.reference?.note && <div className="sub" style={{ marginTop: 4 }}>{tun.reference.note}</div>}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="card">
          <div className="card-head">
            <h3>Instrument separation</h3>
            <span className={`pill ${a.separation.quality === "neural" ? "ok" : "warn"}`}>{a.separation.engine} · {a.separation.model}</span>
          </div>
          {a.separation.notes.map((n, i) => <div key={i} className="notice warn" style={{ marginBottom: 8 }}>{n}</div>)}
          <div className="stem-row">
            <strong>Original mix</strong>
            <SourceControls player={player} id="mix" st={st} />
            <span />
          </div>
          {Object.values(a.stems).map((s) => {
            const d = s.analysis;
            const tr = tracks.filter((t) => a.tracks?.find((x) => x.id === t.id)?.stem === s.name);
            return (
              <div key={s.name} className="stem-row">
                <div className="col" style={{ gap: 2 }}>
                  <strong>{s.label}</strong>
                  {d && (
                    <span className="sub">
                      {d.present ? `${pct(d.energy_share)} of energy · active ${pct(d.active_ratio)}` : "barely present"}
                      {d.guitar_type && (d.guitar_type.confidence >= 0.3
                        ? ` · ${d.guitar_type.type} guitar (${pct(d.guitar_type.confidence)})`
                        : " · electric/acoustic unclear")}
                    </span>
                  )}
                  {d?.content && <span className="sub">likely: {d.content.guesses.map((g) => g.label).join(" / ")}</span>}
                </div>
                <SourceControls player={player} id={s.name} st={st} />
                <div className="row">
                  {tr.map((t) => <button key={t.id} className="sm" onClick={() => openTrack(t.id)}>Open tab</button>)}
                  {!tr.length && s.name !== "drums" && (
                    <>
                      <button className="sm" onClick={() => onTranscribe(s.name, "guitar")}>Tab as guitar</button>
                      <button className="sm ghost" onClick={() => onTranscribe(s.name, "bass")}>as bass</button>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <div className="col" style={{ gap: 16 }}>
          <div className="card">
            <h3 style={{ marginBottom: 10 }}>Transcribed parts</h3>
            <table className="table">
              <thead><tr><th>Part</th><th>Notes</th><th>Confidence</th><th /></tr></thead>
              <tbody>
                {tracks.map((t) => (
                  <tr key={t.id}>
                    <td><strong>{t.name}</strong><div className="sub">{t.instrument} · {t.role}</div></td>
                    <td className="mono">{t.stats.notes}</td>
                    <td>
                      <div className="row" style={{ gap: 4 }}>
                        <span className="pill high">{t.stats.levels.high}</span>
                        <span className="pill medium">{t.stats.levels.medium}</span>
                        <span className="pill low">{t.stats.levels.low}</span>
                      </div>
                    </td>
                    <td><button className="sm" onClick={() => openTrack(t.id)}>Edit</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {tun && (
            <div className="card">
              <h3 style={{ marginBottom: 10 }}>Tuning candidates</h3>
              {tun.ranking.slice(0, 5).map((r) => (
                <div key={r.tuning.id} className="row" style={{ marginBottom: 6 }}>
                  <span style={{ width: 190 }}>{r.tuning.name}</span>
                  <div className="bar grow"><div style={{ width: `${r.probability * 100}%` }} /></div>
                  <span className="mono small" style={{ width: 44, textAlign: "right" }}>{pct(r.probability)}</span>
                </div>
              ))}
              <ul className="sub" style={{ margin: "8px 0 0", paddingLeft: 18 }}>{tun.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </div>
          )}
          {a.guitar_split && (
            <div className="card">
              <h3 style={{ marginBottom: 8 }}>Multiple guitars</h3>
              <PanHistogram hist={a.guitar_split.analysis.pan_histogram} modes={a.guitar_split.analysis.modes} />
              <div className="sub" style={{ marginTop: 6 }}>
                {a.guitar_split.parts.length
                  ? `Separated ${a.guitar_split.parts.length} guitar parts by stereo position.`
                  : a.guitar_split.double_tracked
                    ? "Left and right guitars play the same part (double-tracked) – transcribed once."
                    : `Single stereo position (${a.guitar_split.analysis.reason ?? "no split"}). Lead/rhythm are split by note analysis when needed.`}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SourceControls({ player, id, st }: { player: Player; id: string; st: ReturnType<typeof usePlayerState> }) {
  const s = st.sources.find((x) => x.id === id);
  if (!s) return <span />;
  return (
    <div className="row">
      <div className="ms">
        <button className={`m ${s.muted ? "active" : ""}`} onClick={() => player.setSourceProp(id, { muted: !s.muted })} title="Mute">M</button>
        <button className={`s ${s.solo ? "active" : ""}`} onClick={() => player.setSourceProp(id, { solo: !s.solo })} title="Solo">S</button>
      </div>
      <input type="range" min={0} max={1.5} step={0.01} value={s.volume} onChange={(e) => player.setSourceProp(id, { volume: Number(e.target.value) })} style={{ width: 120 }} aria-label={`${id} volume`} />
    </div>
  );
}

function PanHistogram({ hist, modes }: { hist?: number[]; modes?: { pan: number; share: number }[] }) {
  if (!hist?.length) return null;
  const mx = Math.max(...hist, 1e-6);
  const w = 300;
  const h = 60;
  const pts = hist.map((v, i) => `${(i / (hist.length - 1)) * w},${h - (v / mx) * (h - 4)}`).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h + 14}`} role="img" aria-label="Stereo position of guitar energy">
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="1.5" />
      {modes?.map((m, i) => <line key={i} x1={((m.pan + 1) / 2) * w} x2={((m.pan + 1) / 2) * w} y1={0} y2={h} stroke="var(--cyan)" strokeDasharray="3 3" />)}
      <text x={0} y={h + 12} fill="var(--muted)" fontSize="10">L</text>
      <text x={w / 2 - 3} y={h + 12} fill="var(--muted)" fontSize="10">C</text>
      <text x={w - 8} y={h + 12} fill="var(--muted)" fontSize="10">R</text>
    </svg>
  );
}
