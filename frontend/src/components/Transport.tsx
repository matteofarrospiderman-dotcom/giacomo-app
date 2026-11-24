import type { Player } from "../audio/engine";
import { fmtTime } from "../lib/format";
import { usePlaybackTime, usePlayerState } from "../lib/hooks";
import type { Analysis } from "../lib/types";

const RATES = [0.25, 0.5, 0.6, 0.75, 0.85, 1, 1.25, 1.5];

export default function Transport({ player, analysis }: { player: Player; analysis: Analysis }) {
  const st = usePlayerState(player);
  const t = usePlaybackTime(player, 12);
  const dur = st.duration || analysis.duration;
  const bar = (() => {
    const db = analysis.rhythm.downbeats;
    let k = 0;
    while (k + 1 < db.length && db[k + 1] <= t) k++;
    return db.length && t >= db[0] ? k + 1 : 0;
  })();
  return (
    <div className="transport" role="toolbar" aria-label="Transport">
      <button className="play-btn" onClick={() => player.toggle()} aria-label={st.playing ? "Pause" : "Play"} title="Play / pause (Space)">
        {st.playing ? (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="2" width="3.5" height="12" rx="1" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" /></svg>
        ) : (
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z" /></svg>
        )}
      </button>
      <button className="icon" onClick={() => player.seek(Math.max(0, t - 5))} title="Back 5 s">⟲ 5s</button>
      <span className="time">{fmtTime(t)} <span className="muted">/ {fmtTime(dur)}</span></span>
      <span className="sub mono">bar {bar}</span>
      <div className="row" title="Playback speed – pitch is preserved">
        <span className="sub hide-narrow">Speed</span>
        <select value={st.rate} onChange={(e) => { player.setRate(Number(e.target.value)); e.currentTarget.blur(); }}>
          {RATES.map((r) => <option key={r} value={r}>{Math.round(r * 100)}%</option>)}
        </select>
      </div>
      <button className={st.loopOn ? "active sm" : "sm"} disabled={!st.loop} onClick={() => player.toggleLoop()} title="Toggle loop (L)">
        Loop {st.loop ? `${fmtTime(st.loop[0])}–${fmtTime(st.loop[1])}` : <span className="hide-narrow">(tap a bar number)</span>}
      </button>
      {st.loop && <button className="ghost sm" onClick={() => player.setLoop(null)}>Clear</button>}
      <button className={st.metronome ? "active sm" : "sm"} onClick={() => player.setMetronome(!st.metronome)} title="Click on detected beats">
        <span className="hide-narrow">Metronome</span><span className="show-narrow">Click</span>
      </button>
      <div className="grow" />
      <span className="sub hide-narrow">Listening to: {st.sources.filter((s) => (st.sources.some((x) => x.solo) ? s.solo : !s.muted)).map((s) => s.label).join(", ") || "nothing"}</span>
    </div>
  );
}
