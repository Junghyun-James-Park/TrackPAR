#!/usr/bin/env python3
"""Join the finished passes into one label file.

Nothing here runs a model.

The output shape carries the pipeline's central decision, so it is worth stating
rather than inferring from the JSON:

    gender, age        properties of the PERSON  -> one value per track
    exposed, watched   properties of the FRAME   -> one value per frame

Measured on tracks holding at least one positive, exposed changes between frames
71.8% of the time and watched 85.7%. One value per track for those two is not a
coarse answer, it is the wrong kind of answer.

exposed and watched also come from DIFFERENT prompts at DIFFERENT K. That is
measured, not accidental — see docs/RESULTS.md — but it means the two columns are
not two readings of one model and should not be reported as if they were.

    python pipeline/merge_labels.py --dry-run
    python pipeline/merge_labels.py
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(ROOT, "out"))


def load_records(path):
    if not path or not os.path.exists(path):
        return {}
    obj = json.load(open(path))
    rows = obj.get("records", obj) if isinstance(obj, dict) else obj
    return {r["tid"]: r for r in rows if r.get("tid")}


def load_frames(pattern):
    """Accepts both shapes the pipeline produces.

    exp20 writes one record per track holding a frames[] list; the per-frame
    runner writes one record per frame. Both are keyed back to (tid, fnum).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, pattern))):
        for r in json.load(open(f)):
            tid = r.get("tid")
            if not tid:
                continue
            fr = r.get("frames")
            if isinstance(fr, list):
                for e in fr:
                    if isinstance(e, dict) and e.get("fnum") is not None:
                        out.setdefault(tid, {})[e["fnum"]] = e
            elif r.get("fnum") is not None:
                out.setdefault(tid, {})[r["fnum"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default=os.path.join(OUTDIR, "identity.json"))
    ap.add_argument("--age", default=os.path.join(OUTDIR, "age.json"))
    ap.add_argument("--exposed", default="momentary_exposed*.json",
                    help="glob relative to $TRACKPAR_OUT")
    ap.add_argument("--watched", default="momentary_watched*.json")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "labels.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    src = {
        "gender": (a.gender, os.path.exists(a.gender)),
        "age": (a.age, os.path.exists(a.age)),
        "exposed": (a.exposed, bool(glob.glob(os.path.join(OUTDIR, a.exposed)))),
        "watched": (a.watched, bool(glob.glob(os.path.join(OUTDIR, a.watched)))),
    }
    print(f"{'attribute':10s} {'source':46s} status")
    print("=" * 70)
    for k, (p, ok) in src.items():
        print(f"{k:10s} {os.path.basename(p):46s} {'ok' if ok else 'MISSING'}")
    missing = [k for k, (_, ok) in src.items() if not ok]
    if a.dry_run:
        print(f"\nnot assemblable yet: {', '.join(missing)}" if missing
              else "\nall four sources present")
        return 0
    if missing:
        # A partial file is how the deliverable shipped with age: null for weeks.
        print(f"\nrefusing to write a partial label file; missing {missing}")
        return 1

    ident = load_records(a.gender)
    ages = load_records(a.age)
    exp = load_frames(a.exposed)
    wat = load_frames(a.watched)

    out, n_frames = [], 0
    for tid in sorted(set(ident) | set(ages) | set(exp) | set(wat)):
        s = (ident.get(tid) or {}).get("subattr") or {}
        fe, fw = exp.get(tid, {}), wat.get(tid, {})
        frames = [{"fnum": fn,
                   "exposed": (fe.get(fn) or {}).get("exposed"),
                   "watched": (fw.get(fn) or {}).get("watched")}
                  for fn in sorted(set(fe) | set(fw))]
        n_frames += len(frames)
        out.append({
            "tid": tid,
            "gender": s.get("gender"),
            "age": (ages.get(tid) or {}).get("pred"),
            "frames": frames,
            "source": {
                "gender": f"identity adapter, K={os.environ.get('IDENTITY_K', 4)}",
                "age": "integer prompt, multi-image",
                "exposed": f"{os.path.basename(os.environ.get('EXPOSED_PROMPT', '?'))}"
                           f" @ K={os.environ.get('EXPOSED_K', '?')}",
                "watched": f"{os.path.basename(os.environ.get('WATCHED_PROMPT', '?'))}"
                           f" @ K={os.environ.get('WATCHED_K', '?')}",
            },
        })

    json.dump(out, open(a.out, "w"), indent=1)
    have = lambda k: sum(1 for r in out if r.get(k) is not None)
    print(f"\n{len(out)} tracks -> {a.out}")
    print(f"  gender set on {have('gender')}")
    print(f"  age    set on {have('age')}")
    print(f"  {n_frames} frame rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
