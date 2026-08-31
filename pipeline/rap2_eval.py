#!/usr/bin/env python3
"""
Track 1.2 — score every Phase-4 arm on RAP v2 body attributes.

The gap this closes: arms A/B/C/D1/D2 were compared on Lotte, where the only GT
is gender. The 40 BODY attributes the public-PAR warm-up exists to teach were
never evaluated on held-out data, so "public-PAR warm-up is unhelpful" was an
extrapolation from a single attribute. RAP v2's test split is unseen by all arms
(UPAR train = PA100k + Market1501 + PETA only).

    python rap2_eval.py --arm base                                   # arm A
    python rap2_eval.py --arm stage1 --adapter output/upar_subattr_9b_...
    python rap2_eval.py --arm clippar --backend clippar              # arm B
    python rap2_eval.py --arm d1 --adapter output/subattr5bsplit_tvlm_9b_...
    python rap2_eval.py --arm d2 --adapter output/subattr5bsplit_tpar_9b_...

mA is averaged over `scored_attributes` from the pool file (36 of 40; four RAP
attributes sit under 30 positives and would make bAcc a coin flip on 11 rows).
The always-negative control is computed alongside and MUST come out at exactly
0.500 — that is gate G2's sanity check on the metric implementation itself.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
sys.path.insert(0, JOBS)
from upar_build_subattr_trainset import PROMPT  # noqa: E402
from rap2_mapping import subattr_to_40binary  # noqa: E402

PREFILL = '{"gender": "'


def attach_adapter(model, adapter, with_non_lora=True):
    """Load a LoRA run the way it was SAVED, not just its LoRA tensors.

    These runs train with `--freeze_vision_tower False --freeze_merger False`, so
    the trainer writes a second file, `non_lora_state_dict.bin`, holding the
    vision tower and merger — 456 M parameters. `PeftModel.from_pretrained`
    alone silently discards all of them, which evaluates a model that was never
    trained: base vision tower + fine-tuned LoRA.

    The ordering and the key surgery below mirror `src/utils.py:68-75`, which is
    the repo's own reference loader. non-LoRA weights must go in BEFORE the LoRA
    wrapper, because PeftModel renames the module tree.

    `with_non_lora=False` reproduces the older, incomplete behaviour so the two
    can be compared directly rather than argued about.
    """
    if with_non_lora:
        p = os.path.join(adapter, "non_lora_state_dict.bin")
        if os.path.exists(p):
            sd = torch.load(p, map_location="cpu", weights_only=False)
            # An EMPTY file is legitimate and must not trip the guard below. When
            # both the vision tower and the merger are frozen there is nothing
            # outside the adapter to save, and the trainer still writes the file
            # (1,333 bytes, zero tensors). That is the 27B QLoRA arm's normal
            # state, not a broken checkpoint. The guard exists for the opposite
            # case: a file full of tensors, none of which land.
            if not sd:
                print(f"non_lora: {p} holds 0 tensors — nothing outside the "
                      f"adapter was trained, so there is nothing to load",
                      flush=True)
                from peft import PeftModel
                return PeftModel.from_pretrained(model, adapter).eval()
            sd = {(k[11:] if k.startswith("base_model.") else k): v
                  for k, v in sd.items()}
            if any(k.startswith("model.model.") for k in sd):
                sd = {(k[6:] if k.startswith("model.") else k): v
                      for k, v in sd.items()}
            sd = {k: v.to(model.dtype) for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            hit = len(sd) - len(unexpected)
            print(f"non_lora: loaded {hit}/{len(sd)} tensors "
                  f"({len(unexpected)} unexpected)", flush=True)
            if hit == 0:
                raise RuntimeError(
                    f"non_lora_state_dict.bin matched NOTHING in the model — "
                    f"key surgery is wrong, refusing to report a silent no-op")
        else:
            print(f"non_lora: no file at {p} (nothing to load)", flush=True)

    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter).eval()


def first_json(text):
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def mean_accuracy(pred, gt, header, scored):
    """Per-attribute (TPR+TNR)/2. `scored` restricts the mean; the full table is
    still returned so an excluded or collapsed attribute stays visible."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    rows = []
    for j, h in enumerate(header):
        p, g = pred[:, j], gt[:, j]
        npos, nneg = int(g.sum()), int((1 - g).sum())
        tpr = float(p[g == 1].mean()) if npos else float("nan")
        tnr = float(1 - p[g == 0].mean()) if nneg else float("nan")
        rows.append({"attr": h, "tpr": tpr, "tnr": tnr,
                     "bacc": (tpr + tnr) / 2, "n_pos": npos,
                     "scored": h in scored})
    use = [r["bacc"] for r in rows if r["scored"] and r["bacc"] == r["bacc"]]
    return (sum(use) / len(use) if use else float("nan")), rows


# ------------------------------------------------------------------ backends

def run_vlm(pool, header, args):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model_id)
    proc.tokenizer.padding_side = "left"          # generation must start aligned
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()
    if args.adapter:
        model = attach_adapter(model, args.adapter,
                               with_non_lora=not args.lora_only)
    print(f"model ready ({args.arm})", flush=True)

    txt = PROMPT.replace("<image>\n", "")
    preds, bad, recs, t0 = [], 0, [], time.time()
    for s in range(0, len(pool), args.batch):
        chunk = pool[s: s + args.batch]
        imgs = [Image.open(c["image"]).convert("RGB") for c in chunk]
        texts = [proc.apply_chat_template(
                    [{"role": "user", "content": [
                        {"type": "image", "image": im},
                        {"type": "text", "text": txt}]}],
                    tokenize=False, add_generation_prompt=True) + PREFILL
                 for im in imgs]
        inputs = proc(text=texts, images=imgs, return_tensors="pt",
                      padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        gens = [PREFILL + g for g in proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]
        for c, gen in zip(chunk, gens):
            obj = first_json(gen)
            if obj is None:
                bad += 1
                obj = {}          # a malformed answer is a failure, not a skip
            pb = subattr_to_40binary(obj, header)
            preds.append([pb[h] for h in header])
            recs.append({"name": c["name"], "pred": obj})
        el = time.time() - t0
        print(f"  {len(preds)}/{len(pool)}  parse-fail {bad}  "
              f"{el/len(preds):.2f}s/img  ETA {(len(pool)-len(preds))*el/len(preds)/60:.0f}m",
              flush=True)
    return preds, bad, recs


def run_clippar(pool, header, args):
    """arm B — frozen CLIP ViT-L/14 + the linear 40-attribute head."""
    from clippar_train import load_clippar

    proc, enc, head, ck_header = load_clippar(args.clippar_ckpt)
    assert ck_header == header, "CLIP-PAR head was trained on a different header"

    preds, t0 = [], time.time()
    for s in range(0, len(pool), 64):
        chunk = pool[s: s + 64]
        imgs = [Image.open(c["image"]).convert("RGB") for c in chunk]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].half().cuda()
        with torch.no_grad():
            f = enc(pixel_values=px).image_embeds.float()
            # The head was trained on L2-NORMALISED embeddings (clippar_train.embed
            # divides by the norm). Feeding raw embeds here silently scales every
            # logit by ~10x and saturates the sigmoid.
            f = f / f.norm(dim=-1, keepdim=True)
            p = (torch.sigmoid(head(f)) > 0.5).int().cpu().numpy()
        preds.extend(p.tolist())
        print(f"  {len(preds)}/{len(pool)}  {(time.time()-t0)/len(preds):.3f}s/img",
              flush=True)
    # No parsing stage exists for a classifier, so parse-failure is 0 by
    # construction; G2's >=95% parse bar is vacuously met for this arm.
    return preds, 0, [{"name": c["name"]} for c in pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="label used in the output filename, e.g. base/stage1/d1")
    ap.add_argument("--backend", choices=["vlm", "clippar"], default="vlm")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--clippar-ckpt", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--pool", default=f"{OUTDIR}/rap2_eval_pool.json")
    ap.add_argument("--limit", type=int, default=800,
                    help="prefix of the pool; a prefix is itself a valid sample, "
                         "so runs at different limits stay comparable")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lora-only", action="store_true",
                    help="skip non_lora_state_dict.bin — reproduces the older, "
                         "incomplete loading so the two can be compared")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mnt/nvme0n1p1/pjh/.cache/huggingface")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "6")
    torch.set_num_threads(6)

    blob = json.load(open(args.pool))
    header, scored = blob["header"], set(blob["scored_attributes"])
    pool = blob["samples"][: args.limit]
    gts = [[s["binary"][h] for h in header] for s in pool]
    print(f"RAP v2 test pool: {len(pool)} images, {len(scored)}/40 attrs scored",
          flush=True)

    preds, bad, recs = (run_clippar if args.backend == "clippar" else run_vlm)(
        pool, header, args)

    mA, table = mean_accuracy(preds, gts, header, scored)
    neg, _ = mean_accuracy([[0] * 40] * len(gts), gts, header, scored)
    parse_rate = (len(pool) - bad) / len(pool)

    print(f"\n===== RAP v2 body attributes — arm {args.arm} =====")
    print(f"parse-valid  {len(pool)-bad}/{len(pool)} ({100*parse_rate:.1f}%)")
    print(f"mA (36 attr) {mA:.4f}")
    print(f"always-neg   {neg:.4f}   (must be exactly 0.5000)")
    print(f"\n{'attribute':32s} {'TPR':>6s} {'TNR':>6s} {'bAcc':>6s} {'n_pos':>6s}")
    for r in table:
        mark = "" if r["scored"] else "  (excluded)"
        print(f"{r['attr']:32s} {r['tpr']:6.3f} {r['tnr']:6.3f} "
              f"{r['bacc']:6.3f} {r['n_pos']:6d}{mark}")

    out = args.out or f"{OUTDIR}/rap2_eval_{args.arm}.json"
    json.dump({"arm": args.arm, "backend": args.backend, "adapter": args.adapter,
               "n": len(pool), "parse_fail": bad, "parse_rate": parse_rate,
               "mA": mA, "always_negative_mA": neg,
               "scored_attributes": sorted(scored), "per_attr": table,
               "records": recs}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")

    # G2
    ok = parse_rate >= 0.95 and abs(neg - 0.5) < 1e-9
    print(f"GATE G2 [{args.arm}]: parse {parse_rate:.3f}>=0.95, "
          f"always-neg {neg:.6f}==0.5 -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
