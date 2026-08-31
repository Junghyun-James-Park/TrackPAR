#!/usr/bin/env python3
"""Every prompt at K=1, beside the same prompt at K=8.

The pairing is the point. Both columns score the same 426 trusted-region frames
through the same representation and the same parser, so the only thing that
differs within a row is whether the model saw one frame or eight.

    python k1_grid.py
"""
import json
import os
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
O = OUTDIR
sys.path.insert(0, JOBS)

# k1 tag -> (label, key in the deployment grid)
ARMS = [("plain", "plain", "plain"),
        ("svfd", "svfd", "svfd"),
        ("trueonly", "trueonly", "trueonly"),
        ("meta", "metav4  (deliverable)", "metav4  (deliverable)"),
        ("eyes", "eyes / gaze", "eyes / gaze"),
        ("features", "eyes+nose+mouth", "eyes+nose+mouth"),
        ("combined", "combined", "combined")]


def score(path):
    if not os.path.exists(path):
        return None
    rows = json.load(open(path))
    out = {}
    for a in ("exposed", "watched"):
        pairs = [(r[a], r[f"gt_{a}"]) for r in rows
                 if r.get(a) is not None and r.get(f"gt_{a}") is not None]
        if not pairs:
            out[a] = None
            continue
        p = [bool(x) for x, _ in pairs]
        g = [bool(y) for _, y in pairs]
        tp = sum(1 for x, y in zip(p, g) if x and y)
        fp = sum(1 for x, y in zip(p, g) if x and not y)
        fn = sum(1 for x, y in zip(p, g) if (not x) and y)
        tn = len(p) - tp - fp - fn
        rc = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        out[a] = {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                  "bacc": (rc + tnr) / 2, "pred": sum(p) / len(p),
                  "gt": sum(g) / len(g), "n": len(p)}
    return out


def main():
    grid = os.path.join(O, "momentary_grid.json")
    k8 = json.load(open(grid)).get("deploy", {}) if os.path.exists(grid) else {}

    rows = []
    for tag, lab, k8key in ARMS:
        k1 = score(os.path.join(O, f"k1_control_{tag}.json"))
        if k1:
            rows.append((lab, k1, k8.get(k8key)))
    if not rows:
        print("no K=1 runs yet")
        return 1

    print("\nSAME 426 FRAMES, K=1 AGAINST K=8")
    print("=" * 86)
    print(f"{'prompt':24s} {'exp K=1':>8s} {'exp K=8':>8s} {'Δ':>7s}   "
          f"{'wat K=1':>8s} {'wat K=8':>8s} {'Δ':>7s}")
    f = lambda d, a: (d[a]["f1"] if d and d.get(a) else None)
    cell = lambda v: f"{v:.3f}" if v is not None else "—"

    def delta(a, b):
        return f"{a - b:+.3f}" if (a is not None and b is not None) else "—"

    for lab, k1, k8v in rows:
        e1, e8 = f(k1, "exposed"), f(k8v, "exposed")
        w1, w8 = f(k1, "watched"), f(k8v, "watched")
        print(f"{lab:24s} {cell(e1):>8s} {cell(e8):>8s} {delta(e1, e8):>7s}   "
              f"{cell(w1):>8s} {cell(w8):>8s} {delta(w1, w8):>7s}")

    print("\ncalibration — predicted positive rate, K=1 against K=8")
    print(f"{'prompt':24s} {'exp K=1':>8s} {'exp K=8':>8s}   "
          f"{'wat K=1':>8s} {'wat K=8':>8s}")
    pc = lambda d, a: (f"{d[a]['pred']:.1%}" if d and d.get(a) else "—")
    for lab, k1, k8v in rows:
        print(f"{lab:24s} {pc(k1,'exposed'):>8s} {pc(k8v,'exposed'):>8s}   "
              f"{pc(k1,'watched'):>8s} {pc(k8v,'watched'):>8s}")
    ref = rows[0][1]
    if ref.get("exposed"):
        print(f"{'':24s} true exposed {ref['exposed']['gt']:.1%}"
              + (f", true watched {ref['watched']['gt']:.1%}"
                 if ref.get("watched") else ""))

    best_e = max((f(k1, "exposed") or 0, lab) for lab, k1, _ in rows)
    best_w = max((f(k1, "watched") or 0, lab) for lab, k1, _ in rows)
    print(f"\nbest at K=1 — exposed: {best_e[1]} ({best_e[0]:.3f}), "
          f"watched: {best_w[1]} ({best_w[0]:.3f})")
    print("Gaps under ~0.011 are inside the measured run-to-run noise floor "
          "(see\nmomentary_deploy_grid.noise_floor) and are not a ranking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
