#!/usr/bin/env python3
"""Assemble the label file, taking each attribute from wherever it measured best.

Nothing here runs a model. It joins finished runs by track id.

The reason this exists rather than one unified pass: the four attributes stopped
agreeing on a single setting. Measured on all 5,168 annotated instances,

    exposed   eyes 0.689 [0.670,0.708]   PADQ 0.664 [0.646,0.682]  (overlap)
    watched   svfd 0.740 [0.689,0.784]   PADQ 0.585 [0.536,0.628]  (separated)

and on the 426 deployment frames the same prompts prefer different K: eyes holds
up at K=8 (0.728 against 0.714 at K=1) while svfd's watched collapses from 0.667
to 0.000. So exposed and watched want different prompts AND different K.

That is a per-attribute split, which this project tried once before and dropped.
It was dropped then because the split rested on a 385-track development set and
did not survive full scale. This one is measured on every annotated instance with
confidence intervals, which is the evidence that was missing.

    python merge_deliverable.py --dry-run     # show what it would use
    python merge_deliverable.py --out out/labels_v2.json
"""
import argparse
import glob
import json
import os
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
O = OUTDIR
sys.path.insert(0, JOBS)

# attribute -> (label, how to find it). Each source names the run it comes from
# so a label can be traced back to a measurement.
SOURCES = {
    "gender": ("c_w35 adapter, K=4 multi-image",
               ["e15_identity_*.json"], "record"),
    "age": ("integer prompt, K=4 multi-image, full corpus",
            ["age_all_tracks.json"], "age"),
    "exposed": ("eyes prompt, K=8 deployment path",
                ["pm_eyes_base9b_full_mask_K8_evenly_meta_sh*.json"], "frames"),
    "watched": ("svfd prompt, K=1",
                ["k1_control_svfd.json", "pm_svfd_base9b_full_mask_K8_evenly_svfd_sh*.json"],
                "frames"),
}


def find(pats):
    for p in pats:
        fs = sorted(glob.glob(os.path.join(O, p)))
        if fs:
            return fs
    return []


def load_frames(files):
    """exp20-shaped runs: one record per track holding a frames[] list."""
    out = {}
    for f in files:
        d = json.load(open(f))
        rows = d.get("records", d) if isinstance(d, dict) else d
        for r in rows:
            tid = r.get("tid")
            if not tid:
                continue
            fr = r.get("frames")
            if isinstance(fr, list):
                out.setdefault(tid, {}).update(
                    {e.get("fnum"): e for e in fr if isinstance(e, dict)})
            elif r.get("fnum") is not None:      # k1_control shape: one row per frame
                out.setdefault(tid, {})[r["fnum"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(O, "labels_v2.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print(f"{'attribute':10s} {'source':44s} {'files':>5s}  status")
    print("=" * 78)
    found = {}
    for attr, (lab, pats, kind) in SOURCES.items():
        fs = find(pats)
        found[attr] = (lab, fs, kind)
        print(f"{attr:10s} {lab:44s} {len(fs):5d}  "
              f"{'ok' if fs else 'MISSING — run it first'}")
    missing = [k for k, (_, fs, _) in found.items() if not fs]
    if a.dry_run:
        if missing:
            print(f"\nnot assemblable yet: {', '.join(missing)}")
        return 0
    if missing:
        print(f"\nrefusing to write a partial label file; missing {missing}")
        return 1

    ident_files = found["gender"][1]
    ident = json.load(open(ident_files[0]))
    recs = ident.get("records", ident)
    ages = {r["tid"]: r.get("pred") for r in
            json.load(open(found["age"][1][0])).get("records", [])}
    exp = load_frames(found["exposed"][1])
    wat = load_frames(found["watched"][1])

    out, n_frames, no_exp, no_wat = [], 0, 0, 0
    for r in recs:
        tid = r.get("tid")
        if not tid:
            continue
        s = r.get("subattr") or {}
        fe, fw = exp.get(tid, {}), wat.get(tid, {})
        if not fe:
            no_exp += 1
        if not fw:
            no_wat += 1
        frames = []
        for fn in sorted(set(fe) | set(fw)):
            frames.append({"fnum": fn,
                           "exposed": (fe.get(fn) or {}).get("exposed"),
                           "watched": (fw.get(fn) or {}).get("watched")})
        n_frames += len(frames)
        out.append({"tid": tid, "gender": s.get("gender"), "age": ages.get(tid),
                    "frames": frames,
                    "source": {k: v[0] for k, v in found.items()}})

    json.dump(out, open(a.out, "w"), indent=1)
    have = lambda k: sum(1 for r in out if r.get(k) is not None)
    print(f"\n{len(out)} tracks -> {a.out}")
    print(f"  gender set on {have('gender')}")
    print(f"  age    set on {have('age')}")
    print(f"  {n_frames} frame rows; {no_exp} tracks with no exposed source, "
          f"{no_wat} with no watched source")
    print("\nexposed and watched come from DIFFERENT prompts at DIFFERENT K. That is")
    print("deliberate and measured, but it means the two columns are not two")
    print("readings of one model and should not be reported as if they were.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
