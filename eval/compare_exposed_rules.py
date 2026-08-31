#!/usr/bin/env python3
"""Compare several `exposed` rules against one set of stored observations.

The combined prompt reports eyes / nose / mouth / gaze per frame and leaves every
decision to us. That means the choice of rule is a post-hoc analysis rather than
another GPU run: the same answers can be scored under "eyes alone", "all three
features", "any two", and so on.

Two things to keep honest about that. A rule picked because it scores best on
this data is fitted, not measured, so the spread is reported and the sessions are
shown separately rather than pooled. And the rule that ships should be the one
that also holds on the low-prevalence session, since that is the harder half of
the corpus.

    python compare_exposed_rules.py
"""
import json
import os
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
O = OUTDIR

RULES = {
    "eyes resolvable (both|one)":
        lambda f: f["eyes"] in ("both", "one"),
    "both eyes":
        lambda f: f["eyes"] == "both",
    "eyes AND nose AND mouth":
        lambda f: f["eyes"] in ("both", "one") and f["nose"] and f["mouth"],
    "eyes AND (nose OR mouth)":
        lambda f: f["eyes"] in ("both", "one") and (f["nose"] or f["mouth"]),
    "any two of eyes/nose/mouth":
        lambda f: (int(f["eyes"] in ("both", "one")) + int(f["nose"])
                   + int(f["mouth"])) >= 2,
    "all three, eyes counted as both":
        lambda f: f["eyes"] == "both" and f["nose"] and f["mouth"],
}


def load(tag):
    p = os.path.join(O, f"pm_single_{tag}.json")
    if not os.path.exists(p):
        return None
    rows = []
    for r in json.load(open(p)):
        o = r.get("subattr")
        if not isinstance(o, dict):
            continue
        fr = o.get("frames")
        f0 = fr[0] if isinstance(fr, list) and fr and isinstance(fr[0], dict) else o
        if "eyes" not in f0 or "gaze" not in f0:
            continue
        rows.append(({"eyes": str(f0.get("eyes", "")).strip().lower(),
                      "nose": str(f0.get("nose", "")).strip().lower() in ("yes", "true", "1"),
                      "mouth": str(f0.get("mouth", "")).strip().lower() in ("yes", "true", "1"),
                      "gaze": str(f0.get("gaze", "")).strip().lower()},
                     bool(r["gt_exposed"]), bool(r["gt_watched"])))
    return rows


def f1(pred, gt):
    tp = sum(1 for a, b in zip(pred, gt) if a and b)
    fp = sum(1 for a, b in zip(pred, gt) if a and not b)
    fn = sum(1 for a, b in zip(pred, gt) if (not a) and b)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main():
    sets = [(lab, load(tag)) for lab, tag in
            (("09.20 (high prevalence)", "combined_w"),
             ("09.32 (low prevalence)", "combined_f3"))]
    sets = [(lab, rows) for lab, rows in sets if rows]
    if not sets:
        print("no combined-prompt runs yet")
        return 1

    print("\nexposed — one set of observations, several rules")
    print("=" * 74)
    hdr = f"{'rule':32s}" + "".join(f"{lab.split()[0]:>14s}" for lab, _ in sets)
    print(hdr + f"{'worst':>8s}")
    scored = []
    for name, fn_ in RULES.items():
        vals, preds = [], []
        for _, rows in sets:
            pred = [fn_(f) for f, _, _ in rows]
            gt = [e for _, e, _ in rows]
            vals.append(f1(pred, gt))
            preds.append(sum(pred) / len(pred))
        scored.append((name, vals))
        print(f"{name:32s}" + "".join(f"{v:14.3f}" for v in vals)
              + f"{min(vals):8.3f}")

    print("\npredicted-positive rate against the truth")
    for lab, rows in sets:
        gt = sum(1 for _, e, _ in rows if e) / len(rows)
        print(f"  {lab}: true {gt:.1%}, n={len(rows)}")

    best_mean = max(scored, key=lambda x: sum(x[1]) / len(x[1]))
    best_worst = max(scored, key=lambda x: min(x[1]))
    print(f"\nbest on average : {best_mean[0]}")
    print(f"best worst case : {best_worst[0]}")
    if best_mean[0] != best_worst[0]:
        print("They disagree, so the choice is between average accuracy and")
        print("behaviour on the harder session rather than being read off a total.")

    # watched has one rule worth stating, and it comes free from the same answers
    print("\nwatched — eyes resolvable AND gaze up")
    for lab, rows in sets:
        pred = [f["eyes"] in ("both", "one") and f["gaze"] == "up"
                for f, _, _ in rows]
        gt = [w for _, _, w in rows]
        print(f"  {lab:26s} F1 {f1(pred, gt):.3f}   "
              f"predicted+ {sum(pred)/len(pred):.1%}   true+ {sum(gt)/len(gt):.1%}")
    print("\nA rule chosen because it wins here is fitted, not measured. The")
    print("sessions are kept separate for that reason.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
