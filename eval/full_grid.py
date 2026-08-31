#!/usr/bin/env python3
"""Every prompt on all 5,168 annotated instances, with confidence intervals.

The point of this run is statistical power, so the table leads with intervals
rather than point estimates. Earlier single-image tables compared 0.803 against
0.835 on 104 positives and the deployment table compared 0.667 against 0.468 on
13 — neither could separate anything, but the point estimates did not say so.

    python full_grid.py
"""
import json
import os
import random
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
O = OUTDIR
sys.path.insert(0, JOBS)

ARMS = [("eyes", "eyes / gaze"), ("svfd", "svfd"),
        ("metav4", "metav4 (deliverable)"), ("combined", "combined"),
        ("plain", "plain"), ("trueonly", "trueonly"),
        ("features", "eyes+nose+mouth"), ("subattr", "sub-attribute schema"),
        # PADQ answers ONE attribute per call, so it arrives as two files. It
        # also sees the scene plus a box rather than a crop, so its rows are
        # marked: a comparison against the arms above mixes prompt with
        # representation.
        ("padqE", "PADQ exposed_v3 *"), ("padqW", "PADQ watched_v2 *")]


def load(tag):
    p = os.path.join(O, f"full_{tag}.json")
    if not os.path.exists(p):
        return None
    import momentary_targeted_eval as M
    rows = [r for r in json.load(open(p)) if isinstance(r.get("subattr"), dict)]
    out = {}
    for a in ("exposed", "watched"):
        pairs = [(M.derive(r["subattr"], a), bool(r[f"gt_{a}"])) for r in rows]
        pairs = [(bool(p_), g) for p_, g in pairs if p_ is not None]
        out[a] = pairs or None
    out["n_raw"] = len(rows)
    return out


def f1_of(pairs):
    tp = sum(1 for p, g in pairs if p and g)
    fp = sum(1 for p, g in pairs if p and not g)
    fn = sum(1 for p, g in pairs if (not p) and g)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def stats(pairs, boot=2000):
    tp = sum(1 for p, g in pairs if p and g)
    fp = sum(1 for p, g in pairs if p and not g)
    fn = sum(1 for p, g in pairs if (not p) and g)
    tn = len(pairs) - tp - fp - fn
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    rng = random.Random(0)
    vals = []
    for _ in range(boot):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        vals.append(f1_of(s))
    vals.sort()
    return {"f1": f1_of(pairs), "lo": vals[int(.025 * boot)],
            "hi": vals[int(.975 * boot)], "bacc": (R + tnr) / 2,
            "p": P, "r": R, "pred": (tp + fp) / len(pairs),
            "gt": (tp + fn) / len(pairs), "n": len(pairs), "pos": tp + fn}


def main():
    got = [(lab, load(tag)) for tag, lab in ARMS]
    got = [(lab, d) for lab, d in got if d]
    if not got:
        print("no full_* runs yet")
        return 1

    for attr in ("exposed", "watched"):
        rows = [(lab, stats(d[attr])) for lab, d in got if d.get(attr)]
        if not rows:
            continue
        rows.sort(key=lambda x: -x[1]["f1"])
        print(f"\n{attr.upper()} — all annotated instances, 4 sessions")
        print("=" * 84)
        print(f"{'prompt':22s} {'F1':>6s} {'95% CI':>16s} {'bAcc':>6s} "
              f"{'P':>6s} {'R':>6s} {'pred+':>7s} {'n':>6s}")
        for lab, s in rows:
            print(f"{lab:22s} {s['f1']:6.3f} [{s['lo']:.3f},{s['hi']:.3f}]"
                  f" {s['bacc']:6.3f} {s['p']:6.3f} {s['r']:6.3f} "
                  f"{s['pred']:6.1%} {s['n']:6d}")
        s0 = rows[0][1]
        print(f"  true rate {s0['gt']:.1%}  ({s0['pos']} positives)")
        # who is actually separated from the leader
        best_lab, best = rows[0]
        tied = [lab for lab, s in rows[1:] if s["hi"] >= best["lo"]]
        if tied:
            print(f"  {best_lab} leads, but its CI overlaps: "
                  + ", ".join(tied[:4]) + ("…" if len(tied) > 4 else ""))
        else:
            print(f"  {best_lab} is separated from every other arm.")
    print("\n* PADQ rows see the full scene with the target named by bounding box,")
    print("  not a crop, and answer one attribute per call. Differences against the")
    print("  other arms are prompt AND representation together.")
    print("\nCIs are paired bootstrap over instances. Overlapping intervals mean")
    print("this data cannot rank those arms, however different the point")
    print("estimates look.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
