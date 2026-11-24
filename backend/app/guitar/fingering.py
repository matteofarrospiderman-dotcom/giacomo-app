"""Guitar-aware string/fret assignment.

The problem: every pitch can usually be played in 2-5 places on the neck. Choosing
positions note-by-note gives unplayable tabs (huge jumps, impossible stretches, chords
spread over eight frets). We solve it as a shortest-path problem:

* Notes whose onsets fall within 35 ms form a *group* (chord, double-stop or a single
  note).
* For every group we enumerate physically possible *fingerings*: one distinct string
  per note, fret range 0..max_fret, fretted span within reach, at most four fretting
  fingers (a barre counts as one finger). Each has an **intrinsic cost**: stretch,
  fret height, finger count, muted inner strings in strummed chords, a bonus for
  common open-position shapes, a learned prior from the user's own corrections,
  and a penalty for bending an open string.
* Between consecutive groups a **transition cost** models the fretting hand as a
  four-fret window: notes inside the current window are free, moving the window costs
  proportionally to the distance and inversely to the available time; open strings
  never force a move. It also scores string continuity in single-note lines, rewards
  keeping hammer-ons/pull-offs/slides on one string, penalises cutting off a note that
  is still ringing, and keeps repeated chords in the same shape.
* Viterbi dynamic programming finds the globally cheapest path. The fretting-hand
  window is carried along the best path, so open-string groups inherit the hand
  position of what came before.

For every group we also return the cheapest *local* alternatives (with cost deltas).
When the best and second-best are nearly tied the string assignment is flagged as
uncertain, and the UI offers the alternatives with one click.

Different weight *profiles* give whole alternative transcriptions
(balanced / open-position / minimal-movement), which the editor can compare.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tunings import Tuning

GROUP_TOL = 0.035

PROFILES = {
    "balanced": {"height": 0.05, "shift": 1.0, "shift_fixed": 0.6, "span": 1.2, "open_bonus": 0.25, "string_jump": 0.25,
                 "gap": 0.7, "legato": 0.8, "sustain": 1.0, "repeat": 0.4, "high_fret": 0.15},
    "open_position": {"height": 0.22, "shift": 1.0, "shift_fixed": 0.6, "span": 1.2, "open_bonus": 0.6,
                      "string_jump": 0.2, "gap": 0.7, "legato": 0.8, "sustain": 1.0, "repeat": 0.4, "high_fret": 0.3},
    "minimal_movement": {"height": 0.02, "shift": 2.6, "shift_fixed": 1.4, "span": 1.0, "open_bonus": 0.1,
                         "string_jump": 0.15, "gap": 0.6, "legato": 0.8, "sustain": 0.8, "repeat": 0.6, "high_fret": 0.08},
}
PROFILE_LABELS = {
    "balanced": "Balanced (most likely)",
    "open_position": "Open / low position",
    "minimal_movement": "Minimal hand movement",
}


@dataclass
class Cand:
    pos: tuple[tuple[int, int], ...]  # (string, fret) per note, same order as group notes
    cost: float
    minf: int  # lowest fretted fret, -1 if all open
    maxf: int
    fingers: int
    feasible: bool


def positions_for_pitch(pitch: int, tuning: Tuning, max_fret: int) -> list[tuple[int, int]]:
    out = []
    for s in range(1, tuning.strings + 1):
        f = pitch - tuning.open_midi(s)
        if 0 <= f <= max_fret:
            out.append((s, f))
    return out


def _finger_count(pos: tuple[tuple[int, int], ...]) -> tuple[int, bool]:
    fretted = [(s, f) for s, f in pos if f > 0]
    if not fretted:
        return 0, False
    minf = min(f for _, f in fretted)
    at_min = [s for s, f in fretted if f == minf]
    barre = False
    if len(at_min) >= 2:
        lo, hi = min(at_min), max(at_min)
        # a barre is possible if no string between lo..hi is played open
        barre = all(not (lo < s < hi and f == 0) for s, f in pos)
    n = len(fretted) - (len(at_min) - 1 if barre else 0)
    return n, barre


# common open-position chord shapes (string 6..1 frets, None = muted); recognising
# them gives a bonus because real players reach for these first
OPEN_SHAPES = [
    (None, 3, 2, 0, 1, 0), (3, 2, 0, 0, 0, 3), (3, 2, 0, 0, 3, 3), (None, None, 0, 2, 3, 2), (0, 2, 2, 1, 0, 0),
    (None, 0, 2, 2, 2, 0), (0, 2, 2, 0, 0, 0), (None, 0, 2, 2, 1, 0), (None, None, 0, 2, 3, 1), (0, 2, 0, 1, 0, 0),
    (None, 0, 2, 0, 2, 0), (None, 3, 2, 0, 1, 0), (None, 0, 2, 2, 0, 0), (0, 0, 2, 2, 0, 0),
]


def _shape_bonus(pos: tuple[tuple[int, int], ...], n_strings: int) -> float:
    if n_strings != 6 or len(pos) < 3:
        return 0.0
    frets = {s: f for s, f in pos}
    for shape in OPEN_SHAPES:
        ok = True
        for i, f in enumerate(shape):
            s = 6 - i
            if f is None:
                if s in frets:
                    ok = False
                    break
            elif frets.get(s) not in (f, None):
                ok = False
                break
        if ok and sum(1 for s in frets if shape[6 - s] is not None) == len(frets):
            return 0.6
    return 0.0


class FingeringOptimizer:
    def __init__(self, tuning: Tuning, max_fret: int = 24, profile: str = "balanced", priors: dict | None = None,
                 top_k: int = 24):
        self.t = tuning
        self.max_fret = max_fret
        self.w = PROFILES.get(profile, PROFILES["balanced"])
        self.priors = priors or {}
        self.top_k = top_k

    # ------------------------------------------------------------------ intrinsic
    def _prior_cost(self, pitch: int, string: int, fret: int) -> float:
        sp = self.priors.get("string_given_pitch", {}).get(str(pitch))
        c = 0.0
        if sp:
            total = sum(sp.values()) + 2.0
            p = (sp.get(str(string), 0) + 2.0 / self.t.strings) / total
            c += -0.5 * math.log(p * self.t.strings)  # 0 for uniform, <0 for preferred strings
        zone = self.priors.get("fret_zone")
        if zone:
            total = sum(zone.values()) + 5
            z = str(min(fret // 5, 4))
            c += -0.25 * math.log(((zone.get(z, 0) + 1) / total) * 5)
        return c

    def _intrinsic(self, pos: tuple[tuple[int, int], ...], notes: list[dict]) -> Cand:
        w = self.w
        frets = [f for _, f in pos]
        fretted = [f for f in frets if f > 0]
        minf = min(fretted) if fretted else -1
        maxf = max(fretted) if fretted else -1
        span = (maxf - minf) if fretted else 0
        max_span = 6 if minf >= 9 else 5
        fingers, barre = _finger_count(pos)
        feasible = span <= max_span and fingers <= 4
        cost = 0.0
        if span > 3:
            cost += w["span"] * (span - 3) ** 2
        for (s, f), n in zip(pos, notes):
            cost += w["height"] * f
            if f > 17:
                cost += w["high_fret"] * (f - 17)
            if f == 0:
                cost -= w["open_bonus"]
                if any(t["type"] in ("bend", "bend_release", "vibrato", "prebend_release") for t in n.get("techniques", [])):
                    cost += 3.0  # you cannot bend or vibrato an open string
            cost += self._prior_cost(n["pitch"], s, f)
        if len(pos) >= 3:
            strings = sorted(s for s, _ in pos)
            gaps = (strings[-1] - strings[0] + 1) - len(strings)
            cost += w["gap"] * gaps
            cost += 0.15 * max(0, fingers - 3)
            if barre:
                cost += 0.2
            cost -= _shape_bonus(pos, self.t.strings)
        elif len(pos) == 2:
            (s1, _), (s2, _) = pos
            cost += 0.1 * max(0, abs(s1 - s2) - 2)
        if not feasible:
            cost += 25.0 + 4.0 * max(0, fingers - 4) + 3.0 * max(0, span - max_span)
        return Cand(pos=pos, cost=cost, minf=minf, maxf=maxf, fingers=fingers, feasible=feasible)

    def candidates(self, notes: list[dict]) -> list[Cand]:
        options = []
        for n in notes:
            if n.get("locked") and n.get("string") and n.get("fret") is not None:
                options.append([(int(n["string"]), int(n["fret"]))])
                continue
            opts = positions_for_pitch(n["pitch"], self.t, self.max_fret)
            if not opts:
                opts = []  # unplayable pitch (out of range) -> handled by caller
            options.append(opts)
        if any(not o for o in options):
            return []
        cands: list[Cand] = []
        if len(notes) > self.t.strings:
            return []
        # DFS with distinct strings
        order = sorted(range(len(notes)), key=lambda i: len(options[i]))
        chosen: dict[int, tuple[int, int]] = {}
        used: set[int] = set()
        limit = 4000

        def dfs(k: int) -> None:
            nonlocal limit
            if limit <= 0:
                return
            if k == len(order):
                pos = tuple(chosen[i] for i in range(len(notes)))
                cands.append(self._intrinsic(pos, notes))
                limit -= 1
                return
            i = order[k]
            for s, f in options[i]:
                if s in used:
                    continue
                fr = [ff for ss, ff in chosen.values() if ff > 0] + ([f] if f > 0 else [])
                if fr and max(fr) - min(fr) > 7:
                    continue
                chosen[i] = (s, f)
                used.add(s)
                dfs(k + 1)
                used.discard(s)
                del chosen[i]

        dfs(0)
        cands.sort(key=lambda c: c.cost)
        return cands[: self.top_k]

    # ------------------------------------------------------------------ transitions
    @staticmethod
    def _reach(c: Cand) -> tuple[float, float] | None:
        """Index-finger positions h from which every fretted note of c lies in [h, h+3]."""
        if c.minf < 0:
            return None
        lo = max(1.0, float(c.maxf - 3))
        hi = float(c.minf)
        if lo > hi:  # stretch chord: index must sit on the lowest fret
            lo = hi = float(c.minf)
        return lo, hi

    def _window(self, h: tuple[float, float] | None, c: Cand) -> tuple[tuple[float, float] | None, float]:
        """The hand is modelled lazily as an *interval* of feasible index-finger positions.

        Notes that fit the current interval narrow it without moving the hand; otherwise the
        hand shifts by the distance between the intervals and a fresh interval starts.
        Open strings never constrain the hand.
        """
        r = self._reach(c)
        if r is None:
            return h, 0.0
        if h is None:
            return r, 0.0
        lo, hi = max(h[0], r[0]), min(h[1], r[1])
        if lo <= hi:
            return (lo, hi), 0.0
        dist = r[0] - h[1] if r[0] > h[1] else h[0] - r[1]
        return r, float(dist)

    def _transition(self, a: Cand, a_notes: list[dict], h_a, b: Cand, b_notes: list[dict], dt: float):
        w = self.w
        nh, shift = self._window(h_a, b)
        cost = 0.0
        if shift > 0:
            speed = 0.6 + 0.12 / max(dt, 0.03)
            cost += w["shift_fixed"] + w["shift"] * shift * speed
        if len(a.pos) == 1 and len(b.pos) == 1:
            ds = abs(a.pos[0][0] - b.pos[0][0])
            cost += w["string_jump"] * max(0, ds - 1)
        # legato needs the same string
        a_by_pitch = {n["pitch"]: p for n, p in zip(a_notes, a.pos)}
        for n, (s, f) in zip(b_notes, b.pos):
            for t in n.get("techniques", []):
                if t["type"] in ("hammer_on", "pull_off"):
                    src = a_by_pitch.get(t.get("from_pitch"))
                    if src is not None:
                        cost += -w["legato"] if src[0] == s else 0.4
        for n, (s, f) in zip(a_notes, a.pos):
            for t in n.get("techniques", []):
                if t["type"] == "slide":
                    for m, (s2, _) in zip(b_notes, b.pos):
                        if m["pitch"] == t.get("to_pitch"):
                            cost += -w["legato"] * 0.8 if s2 == s else 0.5
        # don't cut off a note that is still ringing
        b_start = min(n["start"] for n in b_notes)
        for n, (s, f) in zip(a_notes, a.pos):
            if n["end"] - b_start > 0.08:
                for m, (s2, f2) in zip(b_notes, b.pos):
                    if s2 == s and m["pitch"] != n["pitch"]:
                        cost += w["sustain"]
        # repeated identical chord -> same shape
        if len(a.pos) >= 2 and sorted(n["pitch"] for n in a_notes) == sorted(n["pitch"] for n in b_notes):
            if sorted(a.pos) != sorted(b.pos):
                cost += w["repeat"]
        return cost, nh

    # ------------------------------------------------------------------ main
    def optimize(self, notes: list[dict]) -> dict:
        active = [n for n in notes if not n.get("dead")]
        groups = group_notes(active)
        unplayable: list[dict] = []
        cand_lists: list[list[Cand]] = []
        group_notes_l: list[list[dict]] = []
        for g in groups:
            g = sorted(g, key=lambda n: -n["pitch"])
            cands = self.candidates(g)
            dropped = []
            # too many / impossible notes: drop the least confident until playable
            while (not cands or not cands[0].feasible) and len(g) > 1:
                worst = min(g, key=lambda n: (n.get("confidence", 0.5), -abs(n["pitch"] - 60)))
                g = [n for n in g if n is not worst]
                dropped.append(worst)
                cands = self.candidates(g)
            if not cands:
                dropped.extend(g)
                g = []
            for d in dropped:
                d.setdefault("flags", [])
                if "impossible_fingering" not in d["flags"]:
                    d["flags"].append("impossible_fingering")
                unplayable.append(d)
            if g:
                cand_lists.append(cands)
                group_notes_l.append(g)

        G = len(cand_lists)
        if G == 0:
            return {"groups": 0, "unplayable": unplayable, "total_cost": 0.0}
        INF = 1e18
        dp = [np.full(len(c), INF) for c in cand_lists]
        back = [np.zeros(len(c), dtype=int) for c in cand_lists]
        hand: list[list] = [[None] * len(c) for c in cand_lists]
        for k, c in enumerate(cand_lists[0]):
            dp[0][k] = c.cost
            hand[0][k] = self._reach(c)
        for g in range(1, G):
            dt = min(n["start"] for n in group_notes_l[g]) - min(n["start"] for n in group_notes_l[g - 1])
            for k, cb in enumerate(cand_lists[g]):
                best, arg, bh = INF, 0, None
                for j, ca in enumerate(cand_lists[g - 1]):
                    if dp[g - 1][j] >= INF:
                        continue
                    tc, nh = self._transition(ca, group_notes_l[g - 1], hand[g - 1][j], cb, group_notes_l[g], dt)
                    v = dp[g - 1][j] + tc
                    if v < best:
                        best, arg, bh = v, j, nh
                dp[g][k] = best + cb.cost
                back[g][k] = arg
                hand[g][k] = bh
        path = [int(np.argmin(dp[-1]))]
        for g in range(G - 1, 0, -1):
            path.append(int(back[g][path[-1]]))
        path.reverse()

        # assign + local alternatives
        uncertain = 0
        for g, k in enumerate(path):
            cands = cand_lists[g]
            chosen = cands[k]
            notes_g = group_notes_l[g]
            prev = (cand_lists[g - 1][path[g - 1]], group_notes_l[g - 1], hand[g - 1][path[g - 1]]) if g > 0 else None
            nxt = (cand_lists[g + 1][path[g + 1]], group_notes_l[g + 1]) if g + 1 < G else None

            def local(c: Cand) -> float:
                v = c.cost
                h = None
                if prev:
                    dt = min(n["start"] for n in notes_g) - min(n["start"] for n in prev[1])
                    tc, h = self._transition(prev[0], prev[1], prev[2], c, notes_g, dt)
                    v += tc
                else:
                    h = self._reach(c)
                if nxt:
                    dt = min(n["start"] for n in nxt[1]) - min(n["start"] for n in notes_g)
                    tc, _ = self._transition(c, notes_g, h, nxt[0], nxt[1], dt)
                    v += tc
                return v

            base = local(chosen)
            alts = sorted(((local(c) - base, c) for i, c in enumerate(cands) if i != k), key=lambda x: x[0])[:4]
            margin = alts[0][0] if alts else 99.0
            for idx, n in enumerate(notes_g):
                n["string"], n["fret"] = chosen.pos[idx]
                n["group"] = g
                n["alternatives"] = []
                seen = {(n["string"], n["fret"])}
                for delta, c in alts:
                    p = c.pos[idx]
                    if p not in seen:
                        seen.add(p)
                        n["alternatives"].append({"string": p[0], "fret": p[1], "cost_delta": round(float(delta), 3),
                                                  "shape": [list(x) for x in c.pos] if len(c.pos) > 1 else None})
                flags = n.setdefault("flags", [])
                if "impossible_fingering" in flags and chosen.feasible:
                    flags.remove("impossible_fingering")
                if not chosen.feasible and "impossible_fingering" not in flags:
                    flags.append("impossible_fingering")
                if margin < 0.35 and n["alternatives"] and not n.get("locked"):
                    if "uncertain_string" not in flags:
                        flags.append("uncertain_string")
                    uncertain += 1
                elif "uncertain_string" in flags:
                    flags.remove("uncertain_string")
        return {"groups": G, "unplayable": unplayable, "total_cost": round(float(dp[-1][path[-1]]), 3),
                "uncertain_groups": uncertain}


def group_notes(notes: list[dict], tol: float = GROUP_TOL) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for n in sorted(notes, key=lambda n: (n["start"], -n["pitch"])):
        if groups and n["start"] - groups[-1][0]["start"] <= tol and n["pitch"] not in {m["pitch"] for m in groups[-1]}:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def finalize_techniques(notes: list[dict]) -> None:
    """Confirm string-dependent techniques once strings are known."""
    by_start = sorted([n for n in notes if not n.get("dead")], key=lambda n: n["start"])
    for i, n in enumerate(by_start):
        kept = []
        for t in n.get("techniques", []):
            if t["type"] in ("hammer_on", "pull_off"):
                prev = [m for m in by_start[max(0, i - 6): i] if m["pitch"] == t.get("from_pitch")]
                if prev and prev[-1].get("string") == n.get("string"):
                    t = {k: v for k, v in t.items() if k != "provisional"}
                    t["from_fret"] = prev[-1].get("fret")
                    kept.append(t)
                # else: the notes are on different strings, so it was not legato
                continue
            if t["type"] == "slide":
                nxt = [m for m in by_start[i + 1: i + 6] if m["pitch"] == t.get("to_pitch")]
                if not nxt or nxt[0].get("string") != n.get("string"):
                    continue
                t["to_fret"] = nxt[0].get("fret")
            if t["type"] in ("bend", "bend_release", "vibrato", "prebend_release") and n.get("fret") == 0:
                continue
            kept.append(t)
        n["techniques"] = kept
    # let ring: a note that keeps sounding across >= 2 later attacks on other strings
    for i, n in enumerate(by_start):
        later = [m for m in by_start[i + 1: i + 8] if m["start"] < n["end"] - 0.03 and m.get("string") != n.get("string")]
        has = any(t["type"] == "let_ring" for t in n.get("techniques", []))
        if len({m["start"] for m in later}) >= 2 and not has:
            n.setdefault("techniques", []).append({"type": "let_ring", "confidence": 0.6})


def playability_report(notes: list[dict], tuning: Tuning) -> dict:
    """Summary metrics used by tuning detection and shown in the UI."""
    frets = [n["fret"] for n in notes if n.get("fret") is not None and not n.get("dead")]
    groups = group_notes([n for n in notes if n.get("fret") is not None and not n.get("dead")])
    jumps = []
    prev = None
    for g in groups:
        fr = [n["fret"] for n in g if n["fret"] > 0]
        if fr:
            pos = min(fr)
            if prev is not None:
                jumps.append(abs(pos - prev))
            prev = pos
    return {
        "mean_fret": round(float(np.mean(frets)), 2) if frets else 0.0,
        "open_string_ratio": round(float(np.mean([f == 0 for f in frets])), 3) if frets else 0.0,
        "mean_position_jump": round(float(np.mean(jumps)), 2) if jumps else 0.0,
        "large_jumps": int(sum(1 for j in jumps if j > 5)),
        "impossible": int(sum(1 for n in notes if "impossible_fingering" in n.get("flags", []))),
    }


def all_shapes_for_chord(pitches: list[int], tuning: Tuning, max_fret: int = 15, limit: int = 8) -> list[list[tuple[int, int]]]:
    """Enumerate playable voicings for a pitch set (used by the editor's chord helper)."""
    opt = FingeringOptimizer(tuning, max_fret=max_fret)
    notes = [{"pitch": p} for p in sorted(pitches, reverse=True)]
    cands = [c for c in opt.candidates(notes) if c.feasible]
    return [list(c.pos) for c in cands[:limit]]

