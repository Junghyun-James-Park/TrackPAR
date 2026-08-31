#!/usr/bin/env python3
"""Multi-image inference on the held-out tracks — the test MARS actually needs.

The gap this closes
-------------------
MARS was added for exactly one reason: its labels are per-TRACKLET, so it is the
only corpus that can teach MULTI-IMAGE reading, which is how the pipeline
actually runs (one VLM call per person-track). It was then trained on 4 frames
per tracklet and **evaluated on a single crop**, like every other arm. That
scores it on the modality it was not built for and leaves its hypothesis
untested.

There is a second reason to run this. Earlier project work found that
single-image fine-tuning BREAKS multi-image ability — valid multi-image output
collapsed to 16-25% — and exp14 fixed it by fine-tuning on multi-image. Every
UPAR arm here (stage1, u12, D*, S*) is single-image fine-tuned. So this run
doubles as a check on whether those arms are usable at all in deployment.

Same prompt for every arm, so the comparison is about the model, not the schema.
MARS's own training prompt is a reduced one; asking it the standard prompt is
off-distribution for it, which if anything works against the MARS hypothesis.

    python multiimg_eval.py --arm base
    python multiimg_eval.py --arm mars --adapter /path/to/adapter
"""
import argparse
import json
import math
import os
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
sys.path.insert(0, JOBS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--holdout", default=f"{OUTDIR}/holdout_tids.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/nvme0n1p1/pjh/.cache/huggingface")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "4")

    import tvlm_pseudo_subattr as tv

    # Empty --holdout means "every track", which is what an end-to-end labelling
    # run needs: it produces the deliverable, and scoring happens afterwards on
    # whatever subset carries GT.
    keep = set(json.load(open(a.holdout))) if a.holdout else None
    # gt_gender is required only when scoring; a labelling run must cover every
    # track, including the 1,165 with no GT match.
    tracks = [t for t in json.load(open(tv.FRAGMENTS)) if t.get("frames")
              and (keep is None or t["tid"] in keep)
              and (keep is None or t.get("gt_gender"))]
    print(f"{len(tracks)} tracks"
          f"{' (held-out subset)' if keep is not None else ' (ALL, labelling run)'}",
          flush=True)

    gen = tv.TransformersChat(a.model_id, adapter=a.adapter, quantize=False)

    res, bad = [], 0
    for n, t in enumerate(tracks, 1):
        fl = t["frames"]
        k = min(a.frames, len(fl))
        step = (len(fl) - 1) / max(k - 1, 1)
        sel = [fl[round(i * step)] for i in range(k)]
        try:
            ims = [tv.crop_pil(f) for f in sel]
        except Exception:
            bad += 1
            continue
        try:
            txt = gen.ask(ims, tv.PROMPT)
        except Exception as e:
            print(f"  fail {t['tid']}: {e}", flush=True)
            bad += 1
            continue
        # exposed/watched come from the frame the box is largest in, which is the
        # same frame the single-crop protocol scored. Recording them costs nothing
        # (the GT lookup is local) and stops the best models from going unmeasured
        # on two of the four deployment attributes.
        fr = tv.largest_box_frame(fl)
        ge, gw = tv.gt_momentary(fr["image"], fr["box"])
        res.append({"tid": t["tid"], "n_img": len(ims),
                    "gt_gender": t.get("gt_gender"), "gt_age": t.get("gt_age"),
                    "gt_exposed": ge, "gt_watched": gw,
                    "subattr": tv.parse_json(txt),
                    "raw": None if tv.parse_json(txt) else txt})
        if n % 50 == 0:
            ok = sum(1 for r in res if r["subattr"])
            print(f"  {n}/{len(tracks)}  valid {ok}/{len(res)}", flush=True)

    ok = [r for r in res if r["subattr"]]
    valid = len(ok) / len(res) if res else 0.0
    g_ok = g_n = fem = fem_n = 0
    for r in ok:
        s = r["subattr"]
        if s.get("gender") in ("male", "female"):
            p = s["gender"] == "female"
            fem_n += 1
            fem += p
            if r.get("gt_gender"):        # tracks without a GT match still get a label
                g_n += 1
                g_ok += p == str(r["gt_gender"]).upper().startswith("F")
    acc = g_ok / g_n if g_n else float("nan")
    fem_rate = fem / fem_n if fem_n else float("nan")

    print(f"\n===== multi-image (K<={a.frames}) — arm {a.arm} =====")
    print(f"valid multi-image output {len(ok)}/{len(res)} ({valid:.1%})")
    print("  [exp14 found single-image FT collapses this to 16-25%; "
          "multi-image FT restored it to 100%]")
    print(f"gender {acc:.4f} +/-{1.96*math.sqrt(acc*(1-acc)/g_n):.4f} (n={g_n})"
          f"  predicted female {fem_rate:.3f} vs GT 0.567")
    print("  [single-crop reference on the same 349: base 0.8501, u12 0.8682, "
          "MARS 0.8424]")

    # exposed / watched, derived from the same sub-attribute output
    mom = {}
    for i, name in ((0, "exposed"), (1, "watched")):
        pairs = [(tv.derive_momentary(r["subattr"])[i], r[f"gt_{name}"])
                 for r in ok if r.get(f"gt_{name}") is not None]
        pairs = [(p, g) for p, g in pairs if p is not None]
        if not pairs:
            print(f"{name:8s} no GT on these tracks")
            continue
        p, g = zip(*pairs)
        tp = sum(1 for x, y in zip(p, g) if x and y)
        fp = sum(1 for x, y in zip(p, g) if x and not y)
        fn = sum(1 for x, y in zip(p, g) if (not x) and y)
        tn = len(p) - tp - fp - fn
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        mom[name] = {"f1": f1, "p": pr, "r": rc, "bacc": (rc + tnr) / 2,
                     "n": len(p), "gt_rate": sum(g) / len(g),
                     "pred_rate": sum(p) / len(p)}
        print(f"{name:8s} F1 {f1:.3f}  bAcc {(rc+tnr)/2:.3f}  "
              f"pred+ {sum(p)/len(p):.1%} vs gt {sum(g)/len(g):.1%}  (n={len(p)})")
    print("  [bAcc is the comparable column: F1 moves with the base rate]")

    out = a.out or f"{OUTDIR}/multiimg_{a.arm}.json"
    json.dump({"arm": a.arm, "adapter": a.adapter, "frames": a.frames,
               "n": len(res), "valid_rate": valid, "gender": acc,
               "n_gender": g_n, "pred_female": fem_rate if fem_n else None,
               "momentary": mom, "records": res}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
