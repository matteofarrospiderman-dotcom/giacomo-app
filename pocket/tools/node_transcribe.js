// Run the Pocket engine in Node on a mono 22.05 kHz float32 file (CPU TF.js), write the
// tab document as JSON and, if a truth file is given, print accuracy metrics.
//
// usage: NODE_PATH=<dir with @tensorflow/tfjs> node node_transcribe.js <model_dir> <audio.f32> <out.json> [truth.json]
const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");
const E = require("../src/engine.js");

async function loadModel(dir) {
  const mj = JSON.parse(fs.readFileSync(path.join(dir, "model.json")));
  const bin = fs.readFileSync(path.join(dir, "group1-shard1of1.bin"));
  const weightSpecs = mj.weightsManifest.flatMap((g) => g.weights);
  const weightData = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  return tf.loadGraphModel(tf.io.fromMemory({ modelTopology: mj.modelTopology, weightSpecs, weightData, format: mj.format, signature: mj.signature }));
}

function f1(est, ref, tol = 0.06) {
  const used = new Set();
  let tp = 0;
  const pairs = [];
  for (const r of ref) {
    let best = null;
    est.forEach((e, i) => {
      if (used.has(i) || e.pitch !== r.pitch) return;
      const d = Math.abs(e.start - r.start);
      if (d <= tol && (!best || d < best[0])) best = [d, i];
    });
    if (best) { used.add(best[1]); tp++; pairs.push([r, est[best[1]]]); }
  }
  const p = est.length ? tp / est.length : 0;
  const rc = ref.length ? tp / ref.length : 0;
  return { precision: +p.toFixed(3), recall: +rc.toFixed(3), f1: +(p + rc ? (2 * p * rc) / (p + rc) : 0).toFixed(3), pairs };
}

(async () => {
  const [modelDir, audio, out, truthPath] = process.argv.slice(2);
  await tf.setBackend("cpu");
  const model = await loadModel(modelDir);
  const raw = fs.readFileSync(audio);
  const y = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  const t0 = Date.now();
  const cache = audio + ".post.json";
  let post;
  if (fs.existsSync(cache) && process.env.USE_CACHE) {
    const c = JSON.parse(fs.readFileSync(cache));
    post = { T: c.T, note: Float32Array.from(c.note), onset: Float32Array.from(c.onset), contour: Float32Array.from(c.contour) };
  } else {
    post = await E.runModel(tf, model, y);
    fs.writeFileSync(cache, JSON.stringify({ T: post.T, note: Array.from(post.note), onset: Array.from(post.onset), contour: Array.from(post.contour) }));
  }
  const t1 = Date.now();
  if (process.env.DEBUG_VIB) {
    const raw = E.decodeNotes(post, {});
    for (const n of raw.filter((n) => Math.abs(n.start - Number(process.env.DEBUG_VIB)) < 0.08)) {
      console.log("DBG", n.start.toFixed(2), n.pitch, Array.from(n.contour).filter((_, i) => i % 2 === 0).map((v) => v.toFixed(2)).join(" "));
    }
  }
  const doc = E.transcribe(post, y.length / E.SR, { tuningId: "auto" }, y);
  const t2 = Date.now();
  fs.writeFileSync(out, JSON.stringify(doc));
  console.log(`model ${t1 - t0} ms, engine ${t2 - t1} ms, notes ${doc.notes.length}, bpm ${doc.bpm.toFixed(2)}, tuning ${doc.tuning.id}, measures ${doc.measures.length}, phase ${doc.downbeatPhase}`);
  if (truthPath) {
    const truth = JSON.parse(fs.readFileSync(truthPath));
    const ref = truth.notes || [...(truth.rhythm_guitar || []), ...(truth.lead_guitar || [])];
    const m = f1(doc.notes, ref);
    const withPos = m.pairs.filter(([r]) => r.string);
    const same = withPos.filter(([r, e]) => r.string === e.string && r.fret === e.fret).length;
    const tech = {};
    for (const [r, e] of m.pairs) if (r.technique) tech[r.technique] = (tech[r.technique] || []).concat([(e.techniques || []).map((t) => t.type).join("+") || "-"]);
    console.log(JSON.stringify({ precision: m.precision, recall: m.recall, f1: m.f1, fingering_identical: withPos.length ? +(same / withPos.length).toFixed(3) : null, truth_bpm: truth.bpm, techniques_on_truth_notes: tech }));
  }
})();
