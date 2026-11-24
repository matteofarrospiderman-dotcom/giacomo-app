/* TabForge Pocket – touch UI. Depends on window.tf (TF.js) and window.TabEngine. */
(function () {
  "use strict";
  const E = window.TabEngine;
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  // ------------------------------------------------------------------ preferences
  const store = {
    get(k, d) { try { const v = localStorage.getItem("tfp." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("tfp." + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  const S = {
    doc: null, title: "", songId: null, y: null, blob: null,
    sel: null, loop: null, loopOn: false, loopAnchor: null,
    metro: false, follow: store.get("follow", true), rate: store.get("rate", 1), showFb: store.get("fb", true),
    model: null, history: [], layout: null, playing: false, wake: null,
  };

  // ------------------------------------------------------------------ audio
  const audio = new Audio();
  audio.preload = "auto";
  audio.preservesPitch = true;
  audio.mozPreservesPitch = true;
  audio.webkitPreservesPitch = true;
  audio.playsInline = true;
  let objUrl = null;
  let actx = null;
  function ctx() {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state !== "running") actx.resume().catch(() => {});
    return actx;
  }
  try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch { /* not supported */ }

  function setAudio(blob) {
    if (objUrl) URL.revokeObjectURL(objUrl);
    objUrl = URL.createObjectURL(blob);
    audio.src = objUrl;
    audio.playbackRate = S.rate;
    // some sandboxes refuse blob: media; fall back to a data: URI
    audio.onerror = () => {
      audio.onerror = null;
      const fr = new FileReader();
      fr.onload = () => { audio.src = fr.result; audio.playbackRate = S.rate; };
      fr.readAsDataURL(blob);
    };
  }

  function click(accent, when = 0) {
    const c = ctx();
    const t = c.currentTime + when;
    const o = c.createOscillator();
    const g = c.createGain();
    o.frequency.value = accent ? 1760 : 1180;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(accent ? 0.55 : 0.32, t + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
    o.connect(g).connect(c.destination);
    o.start(t);
    o.stop(t + 0.06);
  }

  const pluckCache = new Map();
  function pluck(pitches) {
    const c = ctx();
    const sr = c.sampleRate;
    const master = c.createGain();
    master.gain.value = Math.min(0.9, 1.1 / Math.sqrt(pitches.length));
    master.connect(c.destination);
    [...pitches].sort((a, b) => a - b).forEach((p, i) => {
      let buf = pluckCache.get(p);
      if (!buf) {
        const n = Math.floor(sr * 1.3);
        buf = c.createBuffer(1, n, sr);
        const out = buf.getChannelData(0);
        const period = sr / (440 * Math.pow(2, (p - 69) / 12));
        const len = Math.max(2, Math.floor(period));
        const frac = period - len;
        const line = new Float32Array(len);
        let prev = 0;
        for (let k = 0; k < len; k++) { prev = 0.55 * (Math.random() * 2 - 1) + 0.45 * prev; line[k] = prev; }
        let idx = 0, last = 0;
        for (let k = 0; k < n; k++) {
          const a = line[idx], b = line[(idx + 1) % len];
          const s = a * (1 - frac) + b * frac;
          line[idx] = 0.996 * 0.5 * (s + last);
          last = s;
          idx = (idx + 1) % len;
          out[k] = s * Math.min(1, k / (sr * 0.002)) * (k > n - sr * 0.05 ? (n - k) / (sr * 0.05) : 1) * 0.6;
        }
        pluckCache.set(p, buf);
      }
      const src = c.createBufferSource();
      src.buffer = buf;
      src.connect(master);
      src.start(c.currentTime + i * 0.02);
    });
  }

  async function wakeLock(on) {
    try {
      if (on && !S.wake && navigator.wakeLock) S.wake = await navigator.wakeLock.request("screen");
      if (!on && S.wake) { await S.wake.release(); S.wake = null; }
    } catch { S.wake = null; }
  }
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && S.playing) wakeLock(true); });

  async function play() {
    if (!S.doc) return;
    ctx();
    if (S.metro && audio.currentTime < 0.05 + (S.doc.measures[0]?.start || 0) || (S.metro && S.countInEveryStart)) await countIn();
    try {
      await audio.play();
      S.playing = true;
      wakeLock(true);
    } catch (e) {
      S.playing = false;
      toast("Tap play again to start audio", true);
    }
    syncPlayIcon();
  }
  function pause() {
    audio.pause();
    S.playing = false;
    wakeLock(false);
    syncPlayIcon();
  }
  function syncPlayIcon() {
    // SVG elements have no .hidden property: toggle the attribute itself
    $("ico-play").toggleAttribute("hidden", S.playing);
    $("ico-pause").toggleAttribute("hidden", !S.playing);
    $("btn-play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
  }
  function seek(t) {
    audio.currentTime = clamp(t, 0, (audio.duration || S.doc?.duration || 0) - 0.05);
    lastBeat = beatIndex(audio.currentTime);
    frame(true);
  }
  audio.addEventListener("ended", () => {
    if (S.loopOn && S.loop) { seek(S.doc.measures[S.loop[0]].start); play(); }
    else pause();
  });
  audio.addEventListener("pause", () => { if (S.playing && !audio.ended) { S.playing = false; syncPlayIcon(); wakeLock(false); } });

  async function countIn() {
    const per = 60 / S.doc.bpm / S.rate;
    const n = S.doc.beatsPerBar;
    const layer = document.createElement("div");
    layer.className = "count-in";
    document.body.appendChild(layer);
    for (let k = 0; k < n; k++) {
      click(k === 0);
      layer.textContent = String(k + 1);
      await new Promise((r) => setTimeout(r, per * 1000));
    }
    layer.remove();
  }

  try {
    if ("mediaSession" in navigator) {
      navigator.mediaSession.setActionHandler("play", play);
      navigator.mediaSession.setActionHandler("pause", pause);
      navigator.mediaSession.setActionHandler("seekbackward", () => seek(audio.currentTime - 5));
      navigator.mediaSession.setActionHandler("seekforward", () => seek(audio.currentTime + 5));
    }
  } catch { /* optional */ }

  // ------------------------------------------------------------------ persistence (IndexedDB)
  const DB = {
    db: null,
    async open() {
      if (this.db) return this.db;
      this.db = await new Promise((res, rej) => {
        const r = indexedDB.open("tabforge-pocket", 1);
        r.onupgradeneeded = () => r.result.createObjectStore("songs", { keyPath: "id" });
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
      return this.db;
    },
    async tx(mode, fn) {
      try {
        const db = await this.open();
        return await new Promise((res, rej) => {
          const t = db.transaction("songs", mode);
          const out = fn(t.objectStore("songs"));
          t.oncomplete = () => res(out && out.result !== undefined ? out.result : out);
          t.onerror = () => rej(t.error);
        });
      } catch { return null; }
    },
    put(rec) { return this.tx("readwrite", (s) => s.put(rec)); },
    del(id) { return this.tx("readwrite", (s) => s.delete(id)); },
    async get(id) { const r = await this.tx("readonly", (s) => s.get(id)); return r || null; },
    async list() {
      const all = await this.tx("readonly", (s) => s.getAll());
      return (all || []).map(({ id, title, savedAt, duration }) => ({ id, title, savedAt, duration })).sort((a, b) => b.savedAt - a.savedAt);
    },
  };
  let saveTimer = 0;
  function saveSong() {
    if (!S.songId || !S.blob) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => DB.put({ id: S.songId, title: S.title, savedAt: Date.now(), duration: S.doc.duration, blob: S.blob, doc: S.doc }), 400);
  }

  // ------------------------------------------------------------------ model / transcription
  async function ensureModel(onStep) {
    if (S.model) return S.model;
    onStep && onStep("Loading the note model");
    // free GPU textures as soon as a tensor is disposed instead of pooling them: the pool grows
    // to hundreds of MB, which phones (especially in-app browsers) do not survive
    try { tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0); } catch { /* older TF.js */ }
    try { await tf.setBackend("webgl"); } catch { /* fall back below */ }
    await tf.ready();
    if (tf.getBackend() !== "webgl") await tf.setBackend("cpu");
    const m = await (await fetch("model.json")).json();
    const bin = Uint8Array.from(atob(m.weightsBase64), (c) => c.charCodeAt(0));
    S.model = await tf.loadGraphModel(tf.io.fromMemory({ modelTopology: m.modelTopology, weightSpecs: m.weightSpecs, weightData: bin.buffer, format: m.format, signature: m.signature }));
    return S.model;
  }

  // Blob.arrayBuffer() is missing on older iOS; FileReader works everywhere
  function readArrayBuffer(blob) {
    if (blob.arrayBuffer) return blob.arrayBuffer();
    return new Promise((res, rej) => {
      const fr = new FileReader();
      fr.onload = () => res(fr.result);
      fr.onerror = () => rej(fr.error || new Error("Could not read the file"));
      fr.readAsArrayBuffer(blob);
    });
  }

  async function decodeTo22k(arrayBuffer) {
    // decodeAudioData resamples to the context's rate, so decoding in a 22.05 kHz context gives
    // the model's input directly: no full-quality copy, no second rendering pass
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const tmp = new Ctx(1, 1, E.SR);
    const buf = await new Promise((res, rej) => {
      const p = tmp.decodeAudioData(arrayBuffer, res, rej);
      if (p && p.then) p.then(res, rej);
    });
    if (buf.sampleRate !== E.SR) {
      // very old WebKit ignores the context rate: resample once
      const off = new Ctx(1, Math.ceil(buf.duration * E.SR), E.SR);
      const src = off.createBufferSource();
      src.buffer = buf;
      src.connect(off.destination);
      src.start();
      return (await off.startRendering()).getChannelData(0).slice();
    }
    const n = buf.length;
    const out = new Float32Array(n);
    const ch = buf.numberOfChannels;
    for (let c = 0; c < ch; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < n; i++) out[i] += d[i] / ch;
    }
    return out;
  }

  // ------------------------------------------------------------------ crash breadcrumbs & errors
  // If the phone kills the page (out of memory), no code runs; the breadcrumb tells the next
  // visit what was happening so the person gets an explanation instead of a blank error.
  function breadcrumb(info) { store.set("lastRun", info ? { ...info, at: Date.now() } : null); }

  // self-contained (no helpers declared later), so it also works for errors during start-up
  function showProblem(title, detail) {
    const el = document.querySelector(".problem") || document.createElement("div");
    el.className = "problem";
    el.setAttribute("role", "alert");
    el.textContent = "";
    const t = document.createElement("strong");
    t.textContent = title;
    const p = document.createElement("p");
    p.textContent = detail;
    const b = document.createElement("button");
    b.className = "btn";
    b.textContent = "Dismiss";
    b.addEventListener("click", () => el.remove());
    el.append(t, p, b);
    document.body.appendChild(el);
  }

  window.addEventListener("error", (e) => {
    e.preventDefault();
    showProblem("Something failed inside TabForge Pocket", `${e.message || "Unknown error"}. Screenshot this message if it keeps happening.`);
  });
  window.addEventListener("unhandledrejection", (e) => {
    e.preventDefault();
    const msg = (e.reason && (e.reason.message || String(e.reason))) || "Unknown error";
    showProblem("Something failed inside TabForge Pocket", `${msg}. Screenshot this message if it keeps happening.`);
  });

  let cancelled = false;
  const MAX_SECONDS = 600;
  async function transcribeBlob(blob, title) {
    cancelled = false;
    const steps = ["Decoding audio", "Loading the note model", "Listening for notes", "Choosing strings and frets"];
    const ui = progress(steps);
    const sizeMB = Math.round((blob.size / 1048576) * 10) / 10;
    try {
      ui.step(0);
      breadcrumb({ step: "decoding the audio", title, sizeMB });
      let y = await decodeTo22k(await readArrayBuffer(blob));
      const dur = y.length / E.SR;
      if (dur < 1) throw new Error("The file is too short.");
      if (dur > MAX_SECONDS) throw new Error(`This song is ${Math.round(dur / 60)} minutes long. Pocket handles up to ${MAX_SECONDS / 60} minutes; trim it first.`);
      ui.step(1);
      breadcrumb({ step: "loading the note model", title, sizeMB, dur });
      const model = await ensureModel();
      if (cancelled) return;
      ui.step(2, `${tf.getBackend() === "webgl" ? "GPU" : "CPU"} · ${fmt(dur, true)} of audio`);
      breadcrumb({ step: "listening for notes", title, sizeMB, dur, backend: tf.getBackend() });
      const post = await E.runModel(tf, model, y, (p) => { ui.meter(p); if (cancelled) throw new Error("cancelled"); });
      ui.step(3);
      breadcrumb({ step: "choosing strings and frets", title, sizeMB, dur });
      await new Promise((r) => setTimeout(r, 30));
      const doc = E.transcribe(post, dur, { tuningId: "auto", minConfidence: store.get("minConf", 0.45) }, y);
      y = null; // the samples are no longer needed; let them be collected
      E.refinger(doc, store.get("prior", null));
      setSong(doc, title, blob, "s" + Date.now());
      saveSong();
      breadcrumb(null);
      ui.close();
      toast(`${visibleNotes().length} notes in ${doc.measures.length} bars`);
    } catch (e) {
      breadcrumb(null);
      ui.close();
      if (e.message === "cancelled") return;
      const msg = e && e.message ? e.message : String(e);
      if (/decode|EncodingError|Unable to decode|NotSupported/i.test(msg + (e && e.name))) {
        showProblem("This file could not be read", "Try an MP3, M4A (AAC) or WAV file. Songs bought from Apple Music are protected and cannot be opened.");
      } else {
        showProblem("Transcription stopped", msg);
      }
    }
  }

  // ------------------------------------------------------------------ song state
  function setSong(doc, title, blob, id) {
    pause();
    S.doc = doc;
    S.title = title;
    S.blob = blob;
    S.songId = id;
    S.sel = null;
    S.loop = null;
    S.loopOn = false;
    S.history = [];
    setAudio(blob);
    $("song-title").textContent = title;
    if ("mediaSession" in navigator) try { navigator.mediaSession.metadata = new MediaMetadata({ title, artist: "TabForge Pocket" }); } catch { /* optional */ }
    renderAll();
    $("tab-wrap").scrollTop = 0;
  }

  const visibleNotes = () => (S.doc ? S.doc.notes.filter((n) => !n.deleted && !n.hidden && n.string) : []);
  const level = (n) => (n.edited ? "user" : n.confidence >= 0.7 ? "high" : n.confidence >= 0.45 ? "medium" : "low");

  function snapshot() {
    S.history.push(JSON.stringify({ notes: S.doc.notes, tuning: S.doc.tuning, capo: S.doc.capo, profile: S.doc.profile, minConfidence: S.doc.minConfidence, beats: S.doc.beats, bpm: S.doc.bpm, beatsPerBar: S.doc.beatsPerBar, downbeatPhase: S.doc.downbeatPhase }));
    if (S.history.length > 50) S.history.shift();
  }
  function undo() {
    const h = S.history.pop();
    if (!h) return toast("Nothing to undo");
    Object.assign(S.doc, JSON.parse(h));
    E.refinger(S.doc, store.get("prior", null));
    renderAll();
    saveSong();
    toast("Undone");
  }

  function learnMove(n) {
    const prior = store.get("prior", {}) || {};
    prior[n.pitch] = prior[n.pitch] || {};
    prior[n.pitch][n.string] = (prior[n.pitch][n.string] || 0) + 1;
    store.set("prior", prior);
  }

  function edit(fn, refinger = true) {
    snapshot();
    fn();
    if (refinger) E.refinger(S.doc, store.get("prior", null));
    renderAll();
    saveSong();
  }

  // ------------------------------------------------------------------ chips
  function renderChips() {
    const d = S.doc;
    const el = $("chips");
    el.innerHTML = "";
    if (!d) return;
    const vis = visibleNotes();
    const lv = { high: 0, medium: 0, low: 0, user: 0 };
    vis.forEach((n) => lv[level(n)]++);
    const tot = Math.max(1, vis.length);
    const chip = (html, onClick, title) => {
      const b = document.createElement("button");
      b.className = "chip";
      b.innerHTML = html;
      if (title) b.title = title;
      b.addEventListener("click", onClick);
      el.appendChild(b);
    };
    chip(`<b>${Math.round(d.bpm)}</b> BPM · ${d.beatsPerBar}/4`, () => openMenu("rhythm"));
    chip(`<b>${d.tuning.name.replace(/\s*\(.*\)/, "")}</b>${d.capo ? ` · capo ${d.capo}` : ""}`, () => openMenu("tab"));
    chip(`<b>${vis.length}</b> notes <span class="conf-bar" aria-hidden="true"><i style="width:${(100 * lv.high) / tot}%;background:var(--ok)"></i><i style="width:${(100 * lv.medium) / tot}%;background:var(--warn)"></i><i style="width:${(100 * (lv.low + lv.user)) / tot}%;background:var(--sel)"></i></span>`, () => openMenu("tab"), "High / medium confidence");
    const hidden = d.notes.filter((n) => n.hidden && !n.deleted).length;
    if (hidden) chip(`${hidden} faint notes hidden`, () => openMenu("tab"));
    if (S.history.length) chip("Undo", undo);
  }

  // ------------------------------------------------------------------ tab layout & render
  const SG = 18;
  const TOP = 34;
  const BOT = 12;
  const PAD = 12;
  const LEFT = 26;

  function el(tag, attrs = {}, text) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  }

  function layout() {
    const d = S.doc;
    const W = $("tab-wrap").clientWidth;
    const usable = Math.max(280, W - LEFT - 8);
    const nStr = d.tuning.notes.length;
    const h = TOP + (nStr - 1) * SG + BOT + 8;
    const vis = visibleNotes();
    const rows = [];
    let cur = [];
    let curW = 0;
    const flush = () => {
      if (!cur.length) return;
      const scale = cur.length > 1 || curW > usable * 0.6 ? usable / curW : Math.min(usable / curW, 1.5);
      let x = LEFT;
      const y = rows.length ? rows[rows.length - 1].y + h : 0;
      rows.push({ y, ms: cur.map((m) => { const r = { idx: m.idx, x, w: m.need * scale }; x += r.w; return r; }) });
      cur = [];
      curW = 0;
    };
    d.measures.forEach((m, i) => {
      const onsets = new Set(vis.filter((n) => n.start >= m.start - 0.03 && n.start < m.end - 0.03).map((n) => Math.round(n.start * 20))).size;
      const need = Math.max(130, 28 + onsets * 23);
      if (curW + need > usable && cur.length) flush();
      cur.push({ idx: i, need });
      curW += need;
    });
    flush();
    const meas = new Map();
    rows.forEach((r, ri) => r.ms.forEach((m) => meas.set(m.idx, { row: ri, x: m.x, w: m.w })));
    return { rows, meas, h, height: rows.length * h + 16, nStr, width: W };
  }

  function noteX(m, geo, t) {
    const pos = E.beatPos(m.beats, t);
    return geo.x + PAD + (pos / S.doc.beatsPerBar) * (geo.w - 2 * PAD);
  }

  function techLabels(n) {
    const above = [];
    let prefix = "", suffix = "";
    for (const t of n.techniques || []) {
      if (t.type === "hammer_on") prefix = "h";
      else if (t.type === "pull_off") prefix = "p";
      else if (t.type === "bend") above.push(t.semitones === 2 ? "↑full" : t.semitones === 1 ? "↑½" : `↑${t.semitones / 2}`);
      else if (t.type === "bend_release") above.push("↑↓");
      else if (t.type === "prebend_release") above.push("pb↓");
      else if (t.type === "vibrato") suffix = "~";
      else if (t.type === "palm_mute") above.push("PM");
      else if (t.type === "let_ring") above.push("LR");
    }
    return { above, prefix, suffix };
  }

  function renderTab() {
    const d = S.doc;
    const svg = $("tab");
    svg.innerHTML = "";
    if (!d) return;
    const L = (S.layout = layout());
    svg.setAttribute("viewBox", `0 0 ${L.width} ${L.height}`);
    svg.setAttribute("height", L.height);
    const labels = [...d.tuning.notes].reverse().map((p) => E.midiName(p + d.capo).replace(/-?\d+$/, ""));
    const vis = visibleNotes();
    L.rows.forEach((r) => {
      const g = el("g", { transform: `translate(0 ${r.y})` });
      const last = r.ms[r.ms.length - 1];
      for (let s = 0; s < L.nStr; s++) {
        g.appendChild(el("text", { class: "slabel", x: 6, y: TOP + s * SG + 4 }, labels[s]));
        g.appendChild(el("line", { class: "str", x1: LEFT, x2: last.x + last.w, y1: TOP + s * SG, y2: TOP + s * SG }));
      }
      r.ms.forEach((m) => {
        const ms = d.measures[m.idx];
        if (S.loop && m.idx >= S.loop[0] && m.idx <= S.loop[1]) g.appendChild(el("rect", { class: "loopband", x: m.x, y: TOP - 12, width: m.w, height: (L.nStr - 1) * SG + 24 }));
        g.appendChild(el("line", { class: "bar", x1: m.x, x2: m.x, y1: TOP, y2: TOP + (L.nStr - 1) * SG }));
        g.appendChild(el("text", { class: "mnum", x: m.x + 3, y: 12 }, String(m.idx + 1)));
        g.appendChild(el("rect", { class: "mnum-hit", x: m.x, y: 0, width: Math.min(44, m.w), height: 24, "data-measure": m.idx }));
        const inBar = vis.filter((n) => n.start >= ms.start - 0.03 && n.start < ms.end - 0.03);
        const aboveAt = new Map();
        for (const n of inBar) {
          const x = noteX(ms, m, n.start);
          const y = TOP + (n.string - 1) * SG;
          const tl = techLabels(n);
          const txt = `${tl.prefix}${n.fret}${tl.suffix}`;
          const w = Math.max(14, txt.length * 8.2 + 3);
          const cls = `note ${level(n)}${S.sel === n.id ? " sel" : ""}`;
          const ng = el("g", { class: cls, "data-id": n.id });
          ng.appendChild(el("rect", { class: "hit", x: x - 16, y: y - SG / 2, width: 32, height: SG }));
          ng.appendChild(el("rect", { class: "bg", x: x - w / 2, y: y - 8, width: w, height: 16, rx: 4 }));
          ng.appendChild(el("text", { x, y: y + 0.5 }, txt));
          g.appendChild(ng);
          const key = Math.round(x);
          for (const a of tl.above) {
            const k = aboveAt.get(key) || 0;
            aboveAt.set(key, k + 1);
            g.appendChild(el("text", { class: "tech", x: x - 8, y: TOP - 9 - k * 10 }, a));
          }
        }
      });
      g.appendChild(el("line", { class: "bar", x1: last.x + last.w, x2: last.x + last.w, y1: TOP, y2: TOP + (L.nStr - 1) * SG }));
      svg.appendChild(g);
    });
    cacheNoteEls();
  }

  let noteEls = new Map();
  let lastOn = new Set();
  function cacheNoteEls() {
    noteEls = new Map();
    $("tab").querySelectorAll("[data-id]").forEach((e) => noteEls.set(e.getAttribute("data-id"), e));
    lastOn = new Set();
  }

  function timeToPos(t) {
    const d = S.doc;
    const L = S.layout;
    if (!d || !L || !d.measures.length) return null;
    let i = 0;
    while (i + 1 < d.measures.length && d.measures[i + 1].start <= t) i++;
    const geo = L.meas.get(i);
    if (!geo) return null;
    return { x: noteX(d.measures[i], geo, t), row: geo.row, idx: i };
  }

  function posToTime(x, y) {
    const L = S.layout;
    const ri = Math.floor(y / L.h);
    const row = L.rows[ri];
    if (!row) return null;
    const m = row.ms.find((mm) => x >= mm.x && x < mm.x + mm.w) || (x < row.ms[0].x ? row.ms[0] : null);
    if (!m) return null;
    const ms = S.doc.measures[m.idx];
    const frac = clamp((x - m.x - PAD) / Math.max(1, m.w - 2 * PAD), 0, 1) * S.doc.beatsPerBar;
    const b = Math.floor(Math.min(frac, S.doc.beatsPerBar - 1e-6));
    return ms.beats[b] + (frac - b) * (ms.beats[b + 1] - ms.beats[b]);
  }

  // ------------------------------------------------------------------ fretboard
  function renderFretboard(active = []) {
    const svg = $("fb");
    const fbWrap = $("fretboard");
    fbWrap.hidden = !S.showFb || !S.doc;
    if (fbWrap.hidden) return;
    const d = S.doc;
    const nStr = d.tuning.notes.length;
    const W = 900;
    const frets = window.innerWidth < 700 ? 15 : 22;
    const nut = 34;
    const gap = nStr > 6 ? 16 : 20;
    const H = (nStr - 1) * gap + 32;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const scale = (W - nut - 8) / (1 - Math.pow(2, -frets / 12));
    const fx = (f) => nut + scale * (1 - Math.pow(2, -f / 12));
    const mid = (f) => (f === 0 ? nut - 14 : (fx(f - 1) + fx(f)) / 2);
    const sy = (s) => 16 + (s - 1) * gap;
    svg.innerHTML = "";
    [3, 5, 7, 9, 15, 17, 19, 21].filter((f) => f <= frets).forEach((f) => svg.appendChild(el("circle", { cx: mid(f), cy: H / 2, r: 5, fill: "var(--fret)", opacity: 0.35 })));
    if (frets >= 12) [-1, 1].forEach((k) => svg.appendChild(el("circle", { cx: mid(12), cy: H / 2 + k * gap, r: 5, fill: "var(--fret)", opacity: 0.35 })));
    svg.appendChild(el("rect", { x: nut - 4, y: 8, width: 5, height: H - 16, fill: "var(--fg-2)" }));
    for (let f = 1; f <= frets; f++) {
      svg.appendChild(el("line", { x1: fx(f), x2: fx(f), y1: 8, y2: H - 8, stroke: f === d.capo ? "var(--accent)" : "var(--fret)", "stroke-width": f === d.capo ? 4 : 1.5 }));
      if ([3, 5, 7, 9, 12, 15, 17, 19, 21].includes(f)) svg.appendChild(el("text", { x: mid(f), y: H - 0.5, "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" }, String(f)));
    }
    for (let s = 1; s <= nStr; s++) svg.appendChild(el("line", { x1: nut, x2: W - 8, y1: sy(s), y2: sy(s), stroke: "var(--string)", "stroke-width": 1 + (s - 1) * 0.3 }));
    // stored frets are relative to the capo
    const abs = (n) => n.fret + d.capo;
    const sel = S.sel && d.notes.find((n) => n.id === S.sel);
    if (sel && sel.alternatives) for (const a of sel.alternatives) {
      if (a.fret + d.capo > frets) continue;
      svg.appendChild(el("circle", { cx: mid(a.fret + d.capo), cy: sy(a.string), r: 8, fill: "none", stroke: "var(--sel)", "stroke-dasharray": "3 2", "stroke-width": 1.5 }));
    }
    for (const n of active) {
      if (abs(n) > frets) continue;
      const fill = level(n) === "low" ? "var(--bad)" : level(n) === "medium" ? "var(--warn)" : "var(--accent)";
      svg.appendChild(el("circle", { cx: mid(abs(n)), cy: sy(n.string), r: 9, fill }));
      svg.appendChild(el("text", { x: mid(abs(n)), y: sy(n.string) + 4, "text-anchor": "middle", "font-size": 10.5, "font-weight": 700, fill: "var(--accent-ink)" }, String(n.fret)));
    }
    if (sel && sel.string && abs(sel) <= frets) svg.appendChild(el("circle", { cx: mid(abs(sel)), cy: sy(sel.string), r: 11, fill: "none", stroke: "var(--sel)", "stroke-width": 2.5 }));
    svg.dataset.frets = frets;
  }

  // ------------------------------------------------------------------ per-frame updates
  let lastBeat = -1;
  let lastFbKey = "";
  let lastRow = -1;
  function beatIndex(t) {
    const b = S.doc ? S.doc.beats : [];
    let lo = 0, hi = b.length - 1;
    if (hi < 0 || t < b[0]) return -1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (b[mid] <= t) lo = mid; else hi = mid - 1; }
    return lo;
  }

  function frame(force = false) {
    const d = S.doc;
    if (!d) return;
    const t = audio.currentTime;
    // loop
    if (S.playing && S.loopOn && S.loop) {
      const end = d.measures[S.loop[1]].end;
      if (t >= end - 0.01) { seek(d.measures[S.loop[0]].start); return; }
    }
    // metronome
    if (S.playing && S.metro) {
      const bi = beatIndex(t);
      if (bi > lastBeat && bi >= 0) {
        const bt = d.beats[bi];
        click(d.measures.some((m) => Math.abs(m.start - bt) < 0.01));
      }
      lastBeat = bi;
    }
    // playhead
    const p = timeToPos(t);
    const ph = $("playhead");
    if (p) {
      const L = S.layout;
      const row = L.rows[p.row];
      const wrap = $("tab-wrap");
      const scale = wrap.clientWidth / L.width;
      ph.style.display = "block";
      ph.style.transform = `translate(${p.x * scale}px, ${(row.y + TOP - 16) * scale}px)`;
      ph.style.height = `${((L.nStr - 1) * SG + 32) * scale}px`;
      if (S.follow && (S.playing || force) && p.row !== lastRow) {
        const top = row.y * scale;
        if (top < wrap.scrollTop || top + L.h * scale > wrap.scrollTop + wrap.clientHeight) wrap.scrollTo({ top: Math.max(0, top - 8), behavior: force ? "auto" : "smooth" });
      }
      lastRow = p.row;
      $("bar-label").textContent = `bar ${p.idx + 1}${S.loopOn && S.loop ? ` · loop ${S.loop[0] + 1}–${S.loop[1] + 1}` : ""}`;
    }
    // active notes
    const on = new Set();
    const active = [];
    for (const n of visibleNotes()) if (n.start <= t && n.end > t) { on.add(n.id); active.push(n); }
    for (const id of lastOn) if (!on.has(id)) noteEls.get(id)?.classList.remove("on");
    for (const id of on) if (!lastOn.has(id)) noteEls.get(id)?.classList.add("on");
    lastOn = on;
    const key = [...on].sort().join(",") + "|" + S.sel;
    if (key !== lastFbKey || force) { lastFbKey = key; renderFretboard(active); }
    const dur = audio.duration || d.duration;
    $("time").innerHTML = `${fmt(t)} <small>/ ${fmt(dur, true)}</small>`;
  }
  const fmt = (t, short) => {
    if (!isFinite(t)) t = 0;
    const m = Math.floor(t / 60);
    const s = t - m * 60;
    return short ? `${m}:${String(Math.floor(s)).padStart(2, "0")}` : `${m}:${s.toFixed(1).padStart(4, "0")}`;
  };
  (function loop() {
    if (S.playing || document.visibilityState === "visible") frame();
    requestAnimationFrame(loop);
  })();

  // ------------------------------------------------------------------ interactions
  function svgPoint(evt) {
    const svg = $("tab");
    const r = svg.getBoundingClientRect();
    const scale = S.layout.width / r.width;
    return { x: (evt.clientX - r.left) * scale, y: (evt.clientY - r.top) * scale };
  }
  // taps are handled on "click": browsers suppress it when the finger scrolled, and acting on
  // pointerup would let the follow-up synthetic click land on a freshly opened sheet
  $("tab").addEventListener("click", (e) => {
    if (!S.doc) return;
    const target = e.target;
    const noteEl = target.closest && target.closest("[data-id]");
    if (noteEl) return selectNote(noteEl.getAttribute("data-id"));
    if (target.dataset && target.dataset.measure != null) return tapMeasure(Number(target.dataset.measure));
    const p = svgPoint(e);
    const t = posToTime(p.x, p.y);
    if (t != null) seek(t);
  });

  function tapMeasure(i) {
    // first tap loops one bar; a second tap on another bar extends the loop to it
    if (S.loopAnchor != null && S.loop && S.loopAnchor !== i && Date.now() - S.loopAnchorAt < 8000) {
      S.loop = [Math.min(S.loopAnchor, i), Math.max(S.loopAnchor, i)];
      S.loopAnchor = null;
    } else {
      S.loop = [i, i];
      S.loopAnchor = i;
      S.loopAnchorAt = Date.now();
    }
    S.loopOn = true;
    seek(S.doc.measures[S.loop[0]].start);
    syncTransport();
    renderTab();
    toast(S.loop[0] === S.loop[1] ? `Looping bar ${i + 1} – tap another bar number to extend` : `Looping bars ${S.loop[0] + 1}–${S.loop[1] + 1}`);
  }

  function selectNote(id) {
    S.sel = id;
    const n = S.doc.notes.find((x) => x.id === id);
    renderTab();
    frame(true);
    if (!S.playing) pluck([n.pitch]);
    openNoteSheet(n);
  }

  // transport
  $("btn-play").addEventListener("click", () => (S.playing ? pause() : play()));
  $("btn-prev").addEventListener("click", () => {
    const p = timeToPos(audio.currentTime);
    if (!p) return seek(0);
    const m = S.doc.measures[p.idx];
    seek(audio.currentTime - m.start < 0.4 && p.idx > 0 ? S.doc.measures[p.idx - 1].start : m.start);
  });
  $("btn-next").addEventListener("click", () => {
    const p = timeToPos(audio.currentTime);
    if (p && p.idx + 1 < S.doc.measures.length) seek(S.doc.measures[p.idx + 1].start);
  });
  $("btn-loop").addEventListener("click", () => {
    if (!S.loop) {
      const p = timeToPos(audio.currentTime);
      S.loop = [p ? p.idx : 0, p ? p.idx : 0];
      toast("Looping this bar – tap bar numbers in the tab to change the loop");
    } else S.loopOn = !S.loopOn;
    if (!S.loopOn && S.loop && !$("btn-loop").classList.contains("on")) S.loopOn = true;
    syncTransport();
    renderTab();
  });
  $("btn-metro").addEventListener("click", () => { S.metro = !S.metro; ctx(); syncTransport(); toast(S.metro ? "Click on: one bar count-in when starting from the top" : "Click off"); });
  $("btn-follow").addEventListener("click", () => { S.follow = !S.follow; store.set("follow", S.follow); syncTransport(); });
  $("btn-fb").addEventListener("click", () => { S.showFb = !S.showFb; store.set("fb", S.showFb); syncTransport(); frame(true); });
  const RATES = [0.5, 0.6, 0.75, 0.9, 1];
  function syncTransport() {
    const seg = $("speed");
    seg.innerHTML = "";
    RATES.forEach((r) => {
      const b = document.createElement("button");
      b.textContent = `${Math.round(r * 100)}%`;
      b.className = Math.abs(S.rate - r) < 0.001 ? "on" : "";
      b.setAttribute("aria-pressed", String(b.className === "on"));
      b.addEventListener("click", () => { S.rate = r; audio.playbackRate = r; store.set("rate", r); syncTransport(); });
      seg.appendChild(b);
    });
    $("btn-loop").classList.toggle("on", !!(S.loop && S.loopOn));
    $("btn-loop").textContent = S.loop ? `Loop ${S.loop[0] + 1}${S.loop[1] !== S.loop[0] ? "–" + (S.loop[1] + 1) : ""}` : "Loop";
    $("btn-metro").classList.toggle("on", S.metro);
    $("btn-follow").classList.toggle("on", S.follow);
    $("btn-fb").classList.toggle("on", S.showFb);
  }
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.code === "Space") { e.preventDefault(); S.playing ? pause() : play(); }
    if (e.key === "Escape") closeSheet();
  });

  // ------------------------------------------------------------------ sheets
  function closeSheet() { $("layer").innerHTML = ""; }
  function openSheet(build) {
    closeSheet();
    const bd = document.createElement("div");
    bd.className = "sheet-backdrop";
    const opened = Date.now();
    bd.addEventListener("click", () => { if (Date.now() - opened > 350) closeSheet(); });
    const sh = document.createElement("div");
    sh.className = "sheet";
    sh.setAttribute("role", "dialog");
    sh.innerHTML = '<div class="grab"></div>';
    build(sh);
    $("layer").append(bd, sh);
    // swipe down to dismiss
    let y0 = null;
    sh.addEventListener("touchstart", (e) => { y0 = sh.scrollTop <= 0 ? e.touches[0].clientY : null; }, { passive: true });
    sh.addEventListener("touchend", (e) => { if (y0 != null && e.changedTouches[0].clientY - y0 > 90) closeSheet(); y0 = null; });
    return sh;
  }
  function btn(label, cls, onClick) {
    const b = document.createElement("button");
    b.className = "btn " + (cls || "");
    b.innerHTML = label;
    b.addEventListener("click", onClick);
    return b;
  }
  const h = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  const TECHS = [
    ["bend", "Bend"], ["bend_release", "Bend & release"], ["vibrato", "Vibrato"], ["hammer_on", "Hammer-on"],
    ["pull_off", "Pull-off"], ["palm_mute", "Palm mute"], ["let_ring", "Let ring"],
  ];

  function openNoteSheet(n) {
    const d = S.doc;
    openSheet((sh) => {
      const lv = level(n);
      sh.appendChild(h("h2", null, `${E.midiName(n.pitch)} · string ${n.string}, fret ${n.fret}`));
      const meta = h("div", "row");
      meta.appendChild(h("span", `pill ${lv}`, lv === "user" ? "edited by you" : `${lv} confidence · ${Math.round(n.confidence * 100)}%`));
      if ((n.flags || []).includes("possible_harmonic")) meta.appendChild(h("span", "pill medium", "may be an overtone"));
      if (n.uncertainString) meta.appendChild(h("span", "pill medium", "other positions are almost as likely"));
      sh.appendChild(meta);
      const r1 = h("div", "row");
      r1.style.marginTop = "12px";
      r1.append(
        btn("▶ Hear", "", () => pluck([n.pitch])),
        btn("▶ Play from here", "primary", () => { closeSheet(); seek(Math.max(0, n.start - 0.6)); play(); }),
        btn("Hear chord", "", () => pluck(visibleNotes().filter((m) => Math.abs(m.start - n.start) < 0.04).map((m) => m.pitch))),
      );
      sh.appendChild(r1);
      if (n.alternatives && n.alternatives.length) {
        sh.appendChild(h("h3", null, "Play it elsewhere on the neck"));
        const r = h("div", "row");
        n.alternatives.forEach((a) => r.appendChild(btn(`String ${a.string} · fret ${a.fret}`, "", () => {
          edit(() => { n.string = a.string; n.fret = a.fret; n.locked = true; n.edited = true; learnMove(n); });
          openNoteSheet(n);
        })));
        sh.appendChild(r);
      }
      sh.appendChild(h("h3", null, "Fix the pitch"));
      const rp = h("div", "row");
      [[-12, "Octave down"], [-1, "−1 fret"], [1, "+1 fret"], [12, "Octave up"]].forEach(([dv, label]) => rp.appendChild(btn(label, "", () => {
        edit(() => { n.pitch += dv; n.edited = true; n.locked = false; n.string = n.fret = null; });
        const m = S.doc.notes.find((x) => x.id === n.id);
        pluck([m.pitch]);
        if (m.string) openNoteSheet(m); else { closeSheet(); toast("That pitch is outside this tuning's range", true); }
      })));
      sh.appendChild(rp);
      sh.appendChild(h("h3", null, "Technique"));
      const rt = h("div", "row");
      TECHS.forEach(([type, label]) => {
        const has = (n.techniques || []).some((t) => t.type === type);
        rt.appendChild(btn(label, has ? "on" : "", () => {
          edit(() => {
            n.techniques = (n.techniques || []).filter((t) => t.type !== type);
            if (!has) {
              const t = { type, user: true };
              if (type.startsWith("bend")) t.semitones = 2;
              if (type === "hammer_on" || type === "pull_off") {
                const prev = visibleNotes().filter((m) => m.start < n.start - 0.02 && m.string === n.string).pop();
                t.fromPitch = prev ? prev.pitch : n.pitch + (type === "hammer_on" ? -2 : 2);
              }
              n.techniques.push(t);
            }
            n.edited = true;
          }, false);
          openNoteSheet(n);
        }));
      });
      sh.appendChild(rt);
      const r3 = h("div", "row");
      r3.style.marginTop = "16px";
      r3.append(
        btn("Delete note", "danger", () => { edit(() => { n.deleted = true; }); closeSheet(); toast("Note deleted – Undo is in the top bar"); }),
        btn(n.locked ? "Unlock position" : "Lock position", "", () => { edit(() => { n.locked = !n.locked; }); openNoteSheet(n); }),
      );
      sh.appendChild(r3);
      sh.appendChild(h("p", "hint", "Tip: tap a bar number in the tab to loop that bar. Changing a position re-fingers the notes around it."));
      void d;
    });
  }

  async function openMenu(section) {
    const d = S.doc;
    const songs = await DB.list();
    openSheet((sh) => {
      sh.appendChild(h("h2", null, "Songs, settings and export"));
      // songs
      sh.appendChild(h("h3", null, "Songs"));
      const r = h("div", "row");
      r.append(btn("Open a song file", "primary", () => { closeSheet(); $("file").click(); }), btn("Load the demo", "", () => { closeSheet(); loadDemo(); }));
      sh.appendChild(r);
      sh.appendChild(h("p", "hint", "Works best on a solo guitar recording or an isolated guitar track. Full band mixes give extra notes from other instruments."));
      if (songs.length) {
        const list = h("div", "list");
        songs.forEach((s) => {
          const it = h("div", "list-item");
          it.appendChild(h("div", "grow", `<div class="t">${escapeHtml(s.title)}</div><div class="s">${fmt(s.duration, true)} · ${new Date(s.savedAt).toLocaleDateString()}</div>`));
          it.appendChild(btn("Open", "", async () => { closeSheet(); const rec = await DB.get(s.id); if (rec) { E.refinger(rec.doc, store.get("prior", null)); setSong(rec.doc, rec.title, rec.blob, rec.id); } }));
          it.appendChild(btn("✕", "danger", async () => { await DB.del(s.id); openMenu(section); }));
          list.appendChild(it);
        });
        sh.appendChild(list);
      }
      if (!d) return;
      // tab settings
      sh.appendChild(h("h3", null, "Tab"));
      const g = h("div", "grid2");
      const tun = h("label", "field", "Tuning");
      const sel = document.createElement("select");
      sel.id = "set-tuning";
      E.TUNINGS.forEach((t) => {
        const o = document.createElement("option");
        o.value = t.id;
        const det = d.tuningInfo && d.tuningInfo.id === t.id ? " (detected)" : "";
        o.textContent = t.name + det;
        o.selected = t.id === d.tuning.id;
        sel.appendChild(o);
      });
      sel.addEventListener("change", () => edit(() => { d.tuning = E.TUNINGS.find((t) => t.id === sel.value); d.notes.forEach((n) => { n.locked = false; }); }));
      tun.appendChild(sel);
      const capo = h("label", "field", "Capo");
      const cs = document.createElement("select");
      cs.id = "set-capo";
      for (let i = 0; i <= 9; i++) { const o = document.createElement("option"); o.value = i; o.textContent = i ? `Fret ${i}` : "None"; o.selected = i === d.capo; cs.appendChild(o); }
      cs.addEventListener("change", () => edit(() => { d.capo = Number(cs.value); d.notes.forEach((n) => { n.locked = false; }); }));
      capo.appendChild(cs);
      const prof = h("label", "field", "Fingering style");
      const ps = document.createElement("select");
      ps.id = "set-profile";
      [["balanced", "Most likely"], ["open_position", "Open position"], ["minimal_movement", "Least hand movement"]].forEach(([v, l]) => { const o = document.createElement("option"); o.value = v; o.textContent = l; o.selected = v === d.profile; ps.appendChild(o); });
      ps.addEventListener("change", () => edit(() => { d.profile = ps.value; }));
      prof.appendChild(ps);
      g.append(tun, capo, prof);
      sh.appendChild(g);
      const sens = h("label", "field");
      sens.style.marginTop = "10px";
      const sv = Math.round((1 - (d.minConfidence - 0.2) / 0.5) * 100);
      sens.innerHTML = `<span>Show faint notes: <b id="sens-val">${sv}%</b></span>`;
      const range = document.createElement("input");
      range.type = "range";
      range.id = "set-sens";
      range.min = 0; range.max = 100; range.value = sv;
      range.addEventListener("change", () => {
        const v = Number(range.value) / 100;
        edit(() => { d.minConfidence = 0.7 - 0.5 * v; });
        store.set("minConf", d.minConfidence);
        $("sens-val").textContent = `${range.value}%`;
      });
      sens.appendChild(range);
      sens.appendChild(h("span", "hint", "Lower shows only confident notes. Higher reveals quiet or uncertain notes (in red)."));
      sh.appendChild(sens);
      // rhythm
      const rh = h("h3", null, "Rhythm");
      rh.id = "sec-rhythm";
      sh.appendChild(rh);
      const rr = h("div", "row");
      rr.append(
        btn(`Tempo ÷2 (${Math.round(d.bpm / 2)})`, "", () => edit(() => E.regrid(d, 0.5))),
        btn(`Tempo ×2 (${Math.round(d.bpm * 2)})`, "", () => edit(() => E.regrid(d, 2))),
        btn("Move bar lines by one beat", "", () => edit(() => { d.downbeatPhase = (d.downbeatPhase + 1) % d.beatsPerBar; })),
      );
      sh.appendChild(rr);
      const rb = h("div", "row");
      rb.style.marginTop = "8px";
      [2, 3, 4, 6].forEach((b) => rb.appendChild(btn(`${b} beats per bar`, b === d.beatsPerBar ? "on" : "", () => { edit(() => { d.beatsPerBar = b; d.downbeatPhase = 0; }); openMenu("rhythm"); })));
      sh.appendChild(rb);
      // export
      sh.appendChild(h("h3", null, "Export"));
      const pre = h("pre", "tabtext");
      pre.textContent = E.toText(d, S.title);
      const rx = h("div", "row");
      rx.appendChild(btn("Copy tab as text", "primary", async () => {
        try { await navigator.clipboard.writeText(pre.textContent); toast("Tab copied"); } catch {
          const range2 = document.createRange(); range2.selectNodeContents(pre); const s2 = getSelection(); s2.removeAllRanges(); s2.addRange(range2);
          toast("Text selected – use your phone's Copy");
        }
      }));
      if (S.history.length) rx.appendChild(btn("Undo last change", "", () => { undo(); openMenu(section); }));
      sh.appendChild(rx);
      sh.appendChild(pre);
      // about
      sh.appendChild(h("h3", null, "How it works"));
      sh.appendChild(h("p", null, "Everything runs on your phone. The audio never leaves the device. Notes come from Spotify's Basic Pitch network (Apache-2.0) running in TensorFlow.js. A pitch tracker refines single notes. A fretting-hand model picks the most playable string and fret for each note."));
      sh.appendChild(h("p", null, "Colours show confidence: white is sure, amber is probable, red is a guess, and blue means you edited it. This pocket version has no instrument separation and no tone analysis. The desktop TabForge app adds both, plus Guitar Pro, MusicXML and MIDI export."));
      if (section === "rhythm") setTimeout(() => rh.scrollIntoView({ block: "start" }), 50);
    });
  }

  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  // ------------------------------------------------------------------ progress & toast
  function progress(steps) {
    const wrap = h("div", "progress");
    wrap.setAttribute("role", "status");
    const card = h("div", "progress-card");
    card.appendChild(h("h2", null, "Transcribing on your phone"));
    const list = h("div", "steps");
    const items = steps.map((s) => { const e = h("div", null, "○ " + s); list.appendChild(e); return e; });
    const meter = h("div", "meter", "<i></i>");
    const note = h("div", "hint", "Keep this screen open. A 3-minute song takes about a minute on a recent phone.");
    card.append(list, meter, note, btn("Cancel", "", () => { cancelled = true; wrap.remove(); }));
    wrap.appendChild(card);
    document.body.appendChild(wrap);
    return {
      step(i, extra) {
        items.forEach((e, k) => { e.className = k < i ? "done" : k === i ? "cur" : ""; e.textContent = (k < i ? "✓ " : k === i ? "● " : "○ ") + steps[k] + (k === i && extra ? ` (${extra})` : ""); });
        meter.firstChild.style.width = `${(100 * i) / steps.length}%`;
      },
      meter(p) { meter.firstChild.style.width = `${(100 * (2 + p)) / steps.length}%`; },
      close() { wrap.remove(); },
    };
  }
  let toastTimer = 0;
  function toast(msg, bad) {
    document.querySelectorAll(".toast").forEach((e) => e.remove());
    const t = h("div", "toast" + (bad ? " bad" : ""), escapeHtml(msg));
    t.setAttribute("role", "status");
    document.body.appendChild(t);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.remove(), 3200);
  }

  // ------------------------------------------------------------------ boot
  function renderAll() {
    renderChips();
    renderTab();
    syncTransport();
    lastFbKey = "";
    frame(true);
  }
  $("btn-open").addEventListener("click", () => $("file").click());
  $("btn-menu").addEventListener("click", () => openMenu());
  $("file").addEventListener("change", () => {
    const f = $("file").files && $("file").files[0];
    $("file").value = "";
    if (f) { ctx(); transcribeBlob(f, f.name.replace(/\.[^.]+$/, "")); }
  });
  let resizeTimer = 0;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (S.doc) { renderTab(); lastFbKey = ""; frame(true); } }, 150); });

  async function loadDemo() {
    try {
      const [docRes, audioRes] = await Promise.all([fetch("demo.json"), fetch("demo.mp3")]);
      if (!docRes.ok || !audioRes.ok) throw new Error("demo missing");
      const doc = await docRes.json();
      doc.minConfidence = store.get("minConf", doc.minConfidence ?? 0.45);
      E.refinger(doc, store.get("prior", null));
      setSong(doc, "Demo: E minor arpeggios & lead", await audioRes.blob(), null);
    } catch {
      $("song-title").textContent = "Open a song to begin";
      syncTransport();
    }
  }
  syncTransport();
  loadDemo();
  const last = store.get("lastRun", null);
  if (last && Date.now() - last.at < 24 * 3600 * 1000) {
    breadcrumb(null);
    showProblem("The last transcription did not finish",
      `The page stopped while ${last.step} for "${last.title}" (${last.sizeMB} MB${last.dur ? `, ${fmt(last.dur, true)}` : ""}). ` +
      "This usually means the phone ran out of memory for this page. It now uses much less memory, so try again. " +
      "If it happens again, open the link in Safari instead of the Claude app, or try a shorter clip.");
  }
  // expose for automated tests
  window.__pocket = { S, transcribeBlob, frame };
})();
