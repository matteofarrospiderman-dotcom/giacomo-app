import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { pct, tierColor } from "../lib/format";
import type { Analysis, Effect, Recommendation } from "../lib/types";

const TIER_ORDER = ["high", "probable", "possible", "uncertain"] as const;
const TIER_LABEL: Record<string, string> = {
  high: "High confidence",
  probable: "Probable",
  possible: "Possible",
  uncertain: "Uncertain / not detected",
};

function Spectrum({ freqs, db }: { freqs: number[]; db: number[] }) {
  const w = 520;
  const h = 150;
  const fx = (f: number) => (Math.log10(Math.max(40, f)) - Math.log10(40)) / (Math.log10(16000) - Math.log10(40)) * w;
  const mx = Math.max(...db);
  const fy = (v: number) => h - ((v - (mx - 60)) / 60) * h;
  const pts = freqs.map((f, i) => `${fx(f).toFixed(1)},${Math.max(0, Math.min(h, fy(db[i]))).toFixed(1)}`).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${w} ${h + 16}`} role="img" aria-label="Long-term average spectrum of the guitar stem">
      {[100, 250, 500, 1000, 2000, 4000, 8000].map((f) => (
        <g key={f}>
          <line x1={fx(f)} x2={fx(f)} y1={0} y2={h} stroke="rgba(255,255,255,0.06)" />
          <text x={fx(f)} y={h + 12} fill="var(--muted)" fontSize="9" textAnchor="middle">{f >= 1000 ? `${f / 1000}k` : f}</text>
        </g>
      ))}
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
    </svg>
  );
}

function EffectRow({ e }: { e: Effect }) {
  const [open, setOpen] = useState(false);
  const ev: string[] = e.details?.evidence ?? [];
  return (
    <div className="fx-row">
      <div className="col" style={{ gap: 2 }}>
        <strong>{e.effect}</strong>
        <span className="sub">{e.detected ? "detected" : "not detected"}</span>
      </div>
      {e.detected ? (
        <span className={`pill ${e.tier}`} style={{ justifySelf: "start" }}>{e.tier} · {pct(e.confidence)}</span>
      ) : (
        <span className="pill uncertain" style={{ justifySelf: "start" }} title={`Detector score ${pct(e.confidence)}`}>no evidence</span>
      )}
      <div className="col" style={{ gap: 4 }}>
        <span>{e.summary}</span>
        {(ev.length > 0 || e.details?.note) && (
          <button className="ghost sm" style={{ alignSelf: "flex-start" }} onClick={() => setOpen(!open)}>{open ? "Hide evidence" : "Why?"}</button>
        )}
        {open && (
          <ul className="sub" style={{ margin: 0, paddingLeft: 18 }}>
            {ev.map((x, i) => <li key={i}>{x}</li>)}
            {e.details?.note && <li>{e.details.note}</li>}
          </ul>
        )}
      </div>
    </div>
  );
}

export default function ToneView({ id, analysis, notify }: { id: string; analysis: Analysis; notify: (m: string, bad?: boolean) => void }) {
  const stems = Object.keys(analysis.tone ?? {});
  const [stem, setStem] = useState(stems[0] ?? "");
  const [gear, setGear] = useState<{ amps: Record<string, { label: string; kind: string }>; pedals: Record<string, { label: string; role: string }> } | null>(null);
  const [amp, setAmp] = useState(() => localStorage.getItem("tf.amp") ?? "generic");
  const [pedals, setPedals] = useState<string[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("tf.pedals") ?? "[]");
    } catch {
      return [];
    }
  });
  const [guitar, setGuitar] = useState(() => localStorage.getItem("tf.guitar") ?? "");
  const [rec, setRec] = useState<Recommendation | null>(null);
  const tone = analysis.tone?.[stem];
  const reqSeq = useRef(0);

  useEffect(() => {
    api.gear().then(setGear).catch(() => undefined);
  }, []);
  useEffect(() => {
    localStorage.setItem("tf.amp", amp);
    localStorage.setItem("tf.pedals", JSON.stringify(pedals));
    localStorage.setItem("tf.guitar", guitar);
    if (!stem) return;
    const seq = ++reqSeq.current;
    api.recommend(id, stem, amp, pedals, guitar || undefined)
      .then((r) => { if (seq === reqSeq.current) setRec(r); }) // drop out-of-order answers
      .catch((e) => notify(e.message, true));
  }, [id, stem, amp, pedals, guitar, notify]);

  const grouped = useMemo(() => {
    const g: Record<string, Effect[]> = {};
    for (const e of tone?.effects ?? []) {
      const tier = e.detected ? e.tier : "uncertain";
      (g[tier] ??= []).push(e);
    }
    return g;
  }, [tone]);

  if (!stems.length) return <div className="notice">No guitar stem was analysed for tone.</div>;
  const chips = (
    <div className="row">
      <span className="sub">Guitar part</span>
      <div className="chips">
        {stems.map((s) => <button key={s} className={`chip ${s === stem ? "on" : ""}`} onClick={() => setStem(s)}>{analysis.stems[s]?.label ?? s}</button>)}
      </div>
    </div>
  );
  if (!tone?.available) return <div className="col" style={{ gap: 16 }}>{chips}<div className="notice">Tone analysis unavailable for this part.</div></div>;
  return (
    <div className="col" style={{ gap: 16 }}>
      {chips}
      <div className="grid cols-3">
        <div className="card">
          <h3>Amp character</h3>
          <div className="stat" style={{ fontSize: 18, marginTop: 6 }}>{tone.amp_character.style}</div>
          <div className="sub">Gain ≈ {tone.amp_character.gain}/10 · {tone.amp_character.cabinet}</div>
        </div>
        <div className="card">
          <h3>EQ character</h3>
          <div className="chips" style={{ marginTop: 8 }}>{tone.eq.character.map((c) => <span key={c} className="pill">{c}</span>)}</div>
          <div className="sub" style={{ marginTop: 8 }}>
            Suggested amp EQ: {Object.entries(tone.amp_character.eq).map(([k, v]) => `${k} ${v}`).join(" · ")}
          </div>
        </div>
        <div className="card">
          <h3>Spectrum</h3>
          <Spectrum freqs={tone.eq.spectrum.freqs} db={tone.eq.spectrum.db} />
        </div>
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Original sound analysis</h3>
          <span className="sub">Grouped by confidence – never a claim about the exact gear</span>
        </div>
        {TIER_ORDER.filter((t) => grouped[t]?.length).map((t) => (
          <div key={t} style={{ marginBottom: 10 }}>
            <div className="row" style={{ margin: "6px 0" }}><span className="dot" style={{ background: tierColor[t] }} /><strong>{TIER_LABEL[t]}</strong></div>
            {grouped[t].map((e) => <EffectRow key={e.key} e={e} />)}
          </div>
        ))}
        <div className="notice" style={{ marginTop: 8 }}>{tone.disclaimer}</div>
      </div>

      <div className="card">
        <div className="card-head">
          <h3>Suggested reconstruction</h3>
        </div>
        <div className="grid cols-3" style={{ marginBottom: 14 }}>
          <label className="field">Your amp / modeler
            <select value={amp} onChange={(e) => setAmp(e.target.value)}>
              {gear && Object.entries(gear.amps).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
            </select>
          </label>
          <label className="field">Pickups
            <select value={guitar} onChange={(e) => setGuitar(e.target.value)}>
              <option value="">Unknown / same as recording</option>
              <option value="single_coil">Single coils</option>
              <option value="humbucker">Humbuckers</option>
            </select>
          </label>
          <div className="field">Your pedals
            <div className="chips">
              {gear && Object.entries(gear.pedals).map(([k, v]) => (
                <button key={k} className={`chip ${pedals.includes(k) ? "on" : ""}`} onClick={() => setPedals(pedals.includes(k) ? pedals.filter((p) => p !== k) : [...pedals, k])}>
                  {v.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        {rec && (
          <>
            <div className="chain">
              <div className="chain-block" style={{ minWidth: 90 }}><strong>Guitar</strong></div>
              {rec.chain.map((b, i) => (
                <div key={i} style={{ display: "contents" }}>
                  <span className="chain-arrow">→</span>
                  <div className={`chain-block ${b.available ? "" : "unavailable"}`}>
                    <div className="row between"><strong style={{ fontSize: 13 }}>{b.block}</strong><span className={`pill ${b.tier}`}>{b.tier}</span></div>
                    {b.character && <div className="sub">{b.character}</div>}
                    <div className="knobs">
                      {Object.entries(b.settings).map(([k, v]) => (
                        <span key={k} style={{ display: "contents" }}><span>{k.replace(/_/g, " ")}</span><span className="mono">{String(v)}</span></span>
                      ))}
                    </div>
                    {!b.available && <div className="sub" style={{ marginTop: 6, color: "var(--warn)" }}>not in your gear list</div>}
                  </div>
                </div>
              ))}
              <span className="chain-arrow">→</span>
              <div className="chain-block" style={{ minWidth: 90 }}><strong>Output</strong></div>
            </div>
            {rec.notes.map((n, i) => <div key={i} className="sub" style={{ marginTop: 6 }}>• {n}</div>)}
          </>
        )}
      </div>
    </div>
  );
}
