#!/usr/bin/env python3
"""Is the K=8 packing what kills momentary, or is it the prompt?

The svfd prompt scores watched at bAcc 0.921 shown one image at a time and
exactly 0.500 — chance — on the deployment path, where eight frames go into one
call and eight answers come back. Same prompt, same base model, same per-frame
representation. Its predicted-positive rate goes from 14.2% to 0.0%: asked about
eight frames at once, it answers "no" to all of them.

Two explanations survive that observation and they lead to opposite decisions:

  K       the multi-frame call is the problem, and the fix is to stop batching
  people  the two paths score different frames, and the gap is population

This script removes the second one. It scores the SAME (track, frame) pairs the
deployment run scored, restricted to the same trusted region, built through
exp20's own `build_images` and read back through exp20's own `parse_unified` —
so the only thing that differs from the deployment run is that K is 1.

    python momentary_k1_control.py --prompt svfd
    python momentary_k1_control.py --prompt meta --prompt_file out/metav4_best_prompt.txt

What it found, on those 426 frames:

                    exposed K=1 / K=8      watched K=1 / K=8
  svfd                  0.719 / 0.500          0.667 / 0.000
  metav4                0.647 / 0.730          0.415 / 0.125

So K is not a single knob that helps. Watched is rescued by K=1 for both prompts;
exposed moves in OPPOSITE directions, because each prompt is calibrated at a
different K — at K=8 svfd predicts exposed on 20.7% of frames against a true
42.3% while metav4 predicts 44.6%, and at K=1 that reverses. Any claim about K
has to name the attribute and the prompt.
"""
import argparse
import glob
import json
import os
import sys
import time

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
sys.path.insert(0, JOBS)
O = OUTDIR

import exp20_unified_infer as X          # noqa: E402
import momentary_deploy_grid as G        # noqa: E402


def scored_frames(pattern):
    """The exact (tid, fnum) pairs the deployment run produced an answer for.

    Taken from the stored run rather than recomputed, so a difference in frame
    selection cannot leak in and be mistaken for an effect of K.
    """
    want = []
    for f in sorted(glob.glob(os.path.join(O, pattern))):
        for r in json.load(open(f)):
            for fr in r.get("frames") or []:
                if fr.get("exposed") is not None or fr.get("watched") is not None:
                    want.append((r["session"], r["tid"], fr["fnum"]))
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="svfd",
                    choices=["plain", "padq", "svfd", "trueonly", "meta"])
    ap.add_argument("--prompt_file", default=None)
    ap.add_argument("--rep", default="full_mask")
    ap.add_argument("--model", default="base9b")
    ap.add_argument("--from-run", default=None,
                    help="glob of the deployment run whose frames to reuse; "
                         "defaults to the matching arm")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all-tracks", action="store_true",
                    help="label every track instead of scoring against a stored "
                         "K=8 run. Frames come from exp20's own select_frames at "
                         "K=8, so the frame set matches what the deployment path "
                         "would have chosen, but all 2,438 tracks are covered "
                         "rather than the 1,273 that carry GT. Used to PRODUCE "
                         "the deliverable; scoring still only happens where GT "
                         "exists.")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.all_tracks:
        # O is this file's own out/ directory, which is where phase1 writes.
        frags = json.load(open(os.path.join(O, "phase1_fragments.json")))
        frags = [f for f in frags if f.get("frames")]
        frags = frags[a.shard_idx::a.n_shards]
        want = []
        for f in frags:
            # Select from "frames", NOT "all_frames". all_frames holds every
            # tracked box (62,155 corpus-wide) but only a box — no image path.
            # "frames" holds the ones whose image actually resolved (9,541), and
            # a crop needs the image. Selecting from all_frames and letting the
            # image lookup drop the rest labelled only 6,478 frames, 68% of what
            # is croppable; selecting here covers all of them.
            for fr in X.select_frames(f["frames"], 8, "evenly"):
                want.append((f["session"], f["tid"], fr["fnum"]))
        print(f"{len(frags)} tracks in this shard -> {len(want)} frames "
              f"(all tracks, not just GT-matched)", flush=True)
    else:
        want = None

    pat = a.from_run or (
        f"exp20_base9b_full_mask_K8_evenly"
        f"{'' if a.prompt == 'plain' else '_' + a.prompt}_sh*.json")
    if want is None:
        want = scored_frames(pat)
    if not want:
        print(f"no deployment run matched {pat} — nothing to control against")
        return 1

    fm, tv = G.frame_index()
    # Same trusted-region restriction the deployment table uses. Without it the
    # unlabelled sessions count as false alarms and every number halves.
    if a.all_tracks:
        # A labelling run keeps every frame. The trusted-region restriction below
        # exists for SCORING; applying it here would silently drop most of the
        # corpus from the deliverable.
        inside = [(s, t, n) for s, t, n in want if (t, n) in fm]
    else:
        inside = [(s, t, n) for s, t, n in want
                  if (t, n) in fm and G.trusted(fm[(t, n)][0])]
    # Being inside the region is not enough: gt_momentary returns None when the
    # track box does not match an annotated instance, and those frames are
    # unscorable. The deployment table drops them at scoring time; dropping them
    # here too halves the GPU cost for an identical result.
    if a.all_tracks:
        todo = inside
    else:
        todo = []
        for s, t, n in inside:
            img, box = fm[(t, n)]
            ge, gw = tv.gt_momentary(img, box)
            if ge is not None or gw is not None:
                todo.append((s, t, n))
    print(f"{len(want)} frames answered by the K=8 run, {len(inside)} inside the "
          f"trusted region, {len(todo)} of those carrying GT", flush=True)
    if a.limit:
        todo = todo[: a.limit]

    img_of = X.load_img_index()
    job1_idx = X.E.load_job1_index() if a.rep == "full_mask" else {}
    proc, model = X.M.load_model(a.model)
    model.eval()
    device = next(model.parameters()).device

    # K=1 everywhere it appears: the view paragraph, the prompt body, and the
    # "exactly N entries" instruction all have to agree or the model is asked for
    # a number of answers nobody wants.
    if a.prompt == "meta":
        raw_p = open(a.prompt_file).read()
        prompt = X._view(a.rep, 1) + "\n\n" + raw_p.replace("{K}", "1")
    else:
        prompt = X.build_prompt(a.rep, 1, a.prompt)

    from PIL import Image
    res, t0 = [], time.time()
    for i, (session, tid, fnum) in enumerate(todo, 1):
        rel = img_of.get((session, fnum))
        if not rel:
            continue
        img, box = fm[(tid, fnum)]
        im = Image.open(os.path.join(X.E.IMG_ROOT, rel)).convert("RGB")
        imgs = X.build_images(a.rep, [{"fnum": fnum, "box": box}],
                              {fnum: im}, job1_idx, session)
        raw, _, _ = X.gen_with_tokens(model, proc, imgs, prompt, device, 256)
        pr = X.parse_unified(raw, 1, a.prompt)
        f0 = (pr or {}).get("frames") or [{}]
        ge, gw = tv.gt_momentary(img, box)
        res.append({"session": session, "tid": tid, "fnum": fnum,
                    "exposed": f0[0].get("exposed"),
                    "watched": f0[0].get("watched"),
                    "gt_exposed": ge, "gt_watched": gw,
                    "raw_head": None if pr else raw[:160]})
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}  ({(time.time()-t0)/i:.1f}s/frame)",
                  flush=True)

    out = a.out or os.path.join(O, f"k1_control_{a.prompt}.json")
    json.dump(res, open(out, "w"), indent=1)

    print(f"\n===== K=1 control — prompt {a.prompt}, same frames as K=8 =====")
    print(f"parsed {sum(1 for r in res if r['raw_head'] is None)}/{len(res)}")
    for attr in ("exposed", "watched"):
        pairs = [(r[attr], r[f"gt_{attr}"]) for r in res
                 if r[attr] is not None and r[f"gt_{attr}"] is not None]
        if not pairs:
            print(f"  {attr}: no usable answers")
            continue
        p = [bool(x) for x, _ in pairs]
        g = [bool(y) for _, y in pairs]
        tp = sum(1 for x, y in zip(p, g) if x and y)
        fp = sum(1 for x, y in zip(p, g) if x and not y)
        fn = sum(1 for x, y in zip(p, g) if (not x) and y)
        tn = len(p) - tp - fp - fn
        rc = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        print(f"  {attr:8s} F1 {f1:.3f}  bAcc {(rc+tnr)/2:.3f}  "
              f"pred+ {sum(p)/len(p):.1%} vs true {sum(g)/len(g):.1%}  n={len(p)}")
    print("\nSame frames, same trusted region, same representation and parser as")
    print("the K=8 row. Only K differs, so a gap here is the packing.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
