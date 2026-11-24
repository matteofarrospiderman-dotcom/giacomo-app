import { useMemo, useState } from "react";
import { FLAG_LABELS, fmtTime } from "../lib/format";
import type { ReviewItem, Track } from "../lib/types";

const SEV_COLOR = { high: "var(--bad)", medium: "var(--warn)", low: "var(--muted)" };

export default function ReviewPanel({ track, onJump }: { track: Track; onJump: (r: ReviewItem) => void }) {
  const [filter, setFilter] = useState<string | null>(null);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const r of track.review) c[r.flag] = (c[r.flag] ?? 0) + 1;
    return c;
  }, [track.review]);
  const items = track.review.filter((r) => !filter || r.flag === filter);
  return (
    <div className="card col">
      <div className="row between">
        <h3>Review ({track.review.length})</h3>
        <span className="sub">{track.stats.levels.low} low · {track.stats.levels.medium} medium confidence</span>
      </div>
      <div className="chips">
        <button className={`chip ${!filter ? "on" : ""}`} onClick={() => setFilter(null)}>All</button>
        {Object.entries(counts).map(([f, n]) => (
          <button key={f} className={`chip ${filter === f ? "on" : ""}`} onClick={() => setFilter(filter === f ? null : f)}>{FLAG_LABELS[f] ?? f} {n}</button>
        ))}
      </div>
      <div className="review-list">
        {items.slice(0, 400).map((r, i) => (
          <div key={`${r.note_id}-${r.flag}-${i}`} className="review-item" onClick={() => onJump(r)}>
            <span className="mono muted">{fmtTime(r.time)}</span>
            <span><span className="sev" style={{ background: SEV_COLOR[r.severity] }} />{r.message}</span>
          </div>
        ))}
        {!items.length && <div className="sub">Nothing flagged{filter ? " for this filter" : ""}.</div>}
      </div>
    </div>
  );
}
