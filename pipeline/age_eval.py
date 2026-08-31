#!/usr/bin/env python3
"""Age as MAE on an integer prediction, not as 3-bucket accuracy.

Why the metric had to change
----------------------------
Every arm was reporting age through a 3-value field (young/adult/old). On the
held-out 349, five different arms scored **0.9341 to four decimals** — which is
exactly the always-"adult" rate, because 93.4% of the GT is adult. The metric was
measuring prevalence, not the model. base-9B's lower 0.6275 is not worse
perception; it is the only arm that predicted anything other than adult.

MAE on an integer age has no such degenerate point, and it is the metric the rest
of this project already uses, so the numbers connect to the existing work.

Comparability warning, stated up front: exp14's MAE 5.37 was **multi-image**
(K=2-4 mask-crops) on a 62-track held-out set. This script is **single crop** on
349 tracks. Those are different conditions and the numbers must not be tabled
together — the base-9B row below is the baseline this script's arms are compared
against.

    python age_eval.py --arm base
    python age_eval.py --arm u12 --adapter /path/to/adapter
"""
import argparse
import json
import os
import re
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
sys.path.insert(0, JOBS)

PROMPT = """These are one or more cropped views of the SAME person from an elevated retail CCTV camera.

Estimate the person's age in years as a single integer. Commit to a number even
when the view is poor — a rough estimate is more useful than a refusal.

Output exactly one line of JSON and nothing else:
{"age": <integer years>}"""
PREFILL = '{"age": '


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--frames", type=int, default=1,
                    help="1 = single largest-box crop (the historical setting). "
                         ">1 sends K evenly spaced frames, which is how the "
                         "pipeline actually runs and where exp14 found age MAE "
                         "8.69 -> 5.37. Single-crop age turned out to be no "
                         "better than a constant guess, so this axis matters.")
    ap.add_argument("--min-age", type=float, default=0.0,
                    help="drop tracks whose GT age is <= this. Default 0 removes "
                         "the missing-value placeholders (see below).")
    ap.add_argument("--keep-age-zero", action="store_true",
                    help="keep gt_age==0 rows, reproducing the inflated MAEs "
                         "published before 2026-08-22")
    ap.add_argument("--holdout", default=f"{OUTDIR}/holdout_tids.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/nvme0n1p1/pjh/.cache/huggingface")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "4")

    import torch
    import tvlm_pseudo_subattr as tv
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # Empty --holdout means "every track", which is what a labelling run needs.
    # Scoring then happens on whatever subset carries GT, and the tracks without
    # it still get a prediction. Without this the deliverable ships age: null on
    # all 2,438 tracks, because the only way to run this script was to ask it for
    # a held-out score.
    keep = set(json.load(open(a.holdout))) if a.holdout else None
    tracks = [t for t in json.load(open(tv.FRAGMENTS))
              if t.get("frames")
              and (keep is None
                   or (t["tid"] in keep and t.get("gt_age") is not None))]

    # gt_age == 0 is a MISSING-VALUE placeholder, not an infant: 11 of the 349
    # held-out tracks carry it and there are no ages 1-9 anywhere. Scoring them
    # added 30-45 years of error each, and it hit the good models hardest
    # (c_w35 +1.30 yr, combo +1.17, base only +0.59), so it was quietly
    # compressing the gap between arms. Excluded by default; --keep-age-zero
    # reproduces the older, inflated numbers.
    def usable(t):
        return t.get("gt_age") is not None and float(t["gt_age"]) > a.min_age

    dropped = [t for t in tracks
               if t.get("gt_age") is not None and float(t["gt_age"]) <= a.min_age]
    if not a.keep_age_zero:
        # Drop the placeholder-aged tracks either way. On a labelling run a track
        # with no GT at all is kept and simply goes unscored.
        tracks = [t for t in tracks
                  if usable(t) or (keep is None and t.get("gt_age") is None)]
    n_gt = sum(1 for t in tracks if usable(t))
    print(f"{len(tracks)} tracks"
          f"{' (held-out subset)' if keep is not None else ' (ALL, labelling run)'}"
          f", {n_gt} with usable integer GT age "
          f"(excluded {len(dropped)} with gt_age <= {a.min_age})", flush=True)

    proc = AutoProcessor.from_pretrained(a.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()
    if a.adapter:
        # The vision tower and merger are trained too; PeftModel alone drops them.
        from rap2_eval import attach_adapter
        model = attach_adapter(model, a.adapter)
    print(f"model ready ({a.arm})", flush=True)

    # batch=1 deliberately: batched left-padded generation was verified to change
    # completions on this exact holdout, and every existing Lotte number was
    # measured one at a time.
    errs, preds, bad, recs = [], [], 0, []
    for n, t in enumerate(tracks, 1):
        fl = t["frames"]
        if a.frames <= 1:
            sel = [tv.largest_box_frame(fl)]
        else:
            k = min(a.frames, len(fl))
            step = (len(fl) - 1) / max(k - 1, 1)
            sel = [fl[round(i * step)] for i in range(k)]
        try:
            ims = [tv.crop_pil(f) for f in sel]
        except Exception:
            bad += 1
            continue
        txt = None
        try:
            msg = [{"role": "user", "content":
                    [{"type": "image", "image": im} for im in ims]
                    + [{"type": "text", "text": PROMPT}]}]
            s = proc.apply_chat_template(msg, tokenize=False,
                                         add_generation_prompt=True) + PREFILL
            inp = proc(text=[s], images=ims, return_tensors="pt").to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=24, do_sample=False)
            txt = PREFILL + proc.decode(o[0][inp["input_ids"].shape[1]:],
                                        skip_special_tokens=True)
        except Exception as e:
            print(f"  fail {t['tid']}: {e}", flush=True)
            bad += 1
            continue
        m = re.search(r"-?\d+", txt)
        if not m:
            bad += 1
            recs.append({"tid": t["tid"], "gt": t.get("gt_age"), "pred": None,
                         "raw": txt})
            continue
        p = int(m.group())
        preds.append(p)
        if usable(t):
            errs.append(abs(p - float(t["gt_age"])))
        recs.append({"tid": t["tid"], "gt": t.get("gt_age"), "pred": p})
        if n % 50 == 0:
            sofar = f"{sum(errs)/len(errs):.2f}" if errs else "n/a (no GT yet)"
            print(f"  {n}/{len(tracks)}  MAE so far {sofar}", flush=True)

    import statistics
    gts = [float(t["gt_age"]) for t in tracks if usable(t)]
    mae = w10 = w5 = const = None
    if errs:
        mae = sum(errs) / len(errs)
        w10 = sum(1 for e in errs if e <= 10) / len(errs)
        w5 = sum(1 for e in errs if e <= 5) / len(errs)
        const = min(sum(abs(g - c) for g in gts) / len(gts)
                    for c in range(15, 70))      # best single-number guess
    print(f"\n===== age (integer) — arm {a.arm} =====")
    print(f"predicted     {len(preds)}/{len(tracks)}  (unparsed {bad})")
    if errs:
        print(f"scored on     {len(errs)} tracks carrying GT age")
        print(f"MAE           {mae:.2f} years")
        print(f"within 5 / 10 {w5:.3f} / {w10:.3f}")
        print(f"pred median   {statistics.median(preds)}   "
              f"GT median {statistics.median(gts)}")
        print(f"best constant guess would score MAE {const:.2f} — anything at or "
              f"above that has learned nothing")
    else:
        # A labelling run over the whole corpus is the expected way to land here.
        print("no GT age among the selected tracks, so nothing was scored — "
              "the predictions are still written")
        print(f"pred median   {statistics.median(preds) if preds else 'n/a'}")

    out = a.out or f"{OUTDIR}/age_eval_{a.arm}.json"
    json.dump({"arm": a.arm, "adapter": a.adapter,
               "n": len(preds), "n_scored": len(errs),
               "unparsed": bad, "mae": mae, "within5": w5, "within10": w10,
               "constant_baseline_mae": const,
               "excluded_age_le": None if a.keep_age_zero else a.min_age,
               "n_excluded": len(dropped), "records": recs},
              open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
