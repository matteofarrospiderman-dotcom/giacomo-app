"""Note-level transcription metrics (onset tolerance 50 ms, exact pitch), mir_eval-style."""


def note_f1(est: list[dict], ref: list[dict], onset_tol: float = 0.05) -> dict:
    used = set()
    tp = 0
    for r in ref:
        best = None
        for i, e in enumerate(est):
            if i in used or e["pitch"] != r["pitch"]:
                continue
            d = abs(e["start"] - r["start"])
            if d <= onset_tol and (best is None or d < best[0]):
                best = (d, i)
        if best:
            used.add(best[1])
            tp += 1
    p = tp / len(est) if est else 0.0
    rc = tp / len(ref) if ref else 0.0
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    return {"precision": round(p, 3), "recall": round(rc, 3), "f1": round(f, 3), "tp": tp, "n_est": len(est), "n_ref": len(ref)}
