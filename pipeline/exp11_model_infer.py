"""Exp 11: age/gender on SAM3 tracks (multi-image + single) with different models
and the BEST input representation from exp10.

--model : base    (Qwen3.5-4B)
          base9b (Qwen3.5-9B, no fine-tuning — size control for the FT-9B)
          ageft  (Qwen3.5-4B + age-only LoRA adapter)
          27b    (Qwen3.5-27B, 4-bit nf4)
--rep   : crop|full_prompt|full_crop|mask|auto   (auto = best multi gender_acc in exp10)

Env: qwen35_ft. GPU, shardable. Reuses image prep from exp10_repr_infer.
"""
import argparse
import glob
import json
import os
import sys
import time

import torch
from PIL import Image

# CPU hygiene: the default is one intra-op thread per core (64 here), which both
# oversubscribes the box and thrashes when the job is pinned to a core slice.
# Preprocessing (4K JPEG decode + resize) is the CPU-heavy part; 8 threads saturates
# it. Override with TORCH_THREADS if a run really needs more.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import exp10_repr_infer as E
from utils import load_pretrained_model, get_model_name_from_path

BASE = E.BASE
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(BASE, "out"))
# Fine-tuned checkpoints from earlier arms. The shipped pipeline uses "base9b"
# and never touches these; they are kept so the ablation arms remain runnable
# if you have the checkpoints. Empty by default so a clone fails loudly rather
# than silently loading someone else's path.
AGE_CKPT = os.environ.get("AGE_CKPT", "")
AGE9B_CKPT = os.environ.get("AGE9B_CKPT", "")
QWEN4B = "Qwen/Qwen3.5-4B"
QWEN9B = "Qwen/Qwen3.5-9B"
QWEN27B = os.environ.get("QWEN27B", "Qwen/Qwen3.5-27B")


def best_rep_from_exp10():
    p = os.path.join(OUTDIR, "exp10_results.json")
    r = json.load(open(p))
    return max(r, key=lambda k: r[k]["settings"]["multi"]["gender_acc"])


def load_model(which):
    if which == "base":
        proc, model = load_pretrained_model(model_path=QWEN4B, model_base=None,
                                            model_name=get_model_name_from_path(QWEN4B),
                                            torch_dtype=torch.bfloat16, device_map="auto")
    elif which == "ageft":
        proc, model = load_pretrained_model(model_path=AGE_CKPT, model_base=QWEN4B,
                                            model_name=get_model_name_from_path(AGE_CKPT),
                                            torch_dtype=torch.bfloat16, device_map="auto")
    elif which == "base9b":
        proc, model = load_pretrained_model(model_path=QWEN9B, model_base=None,
                                            model_name=get_model_name_from_path(QWEN9B),
                                            torch_dtype=torch.bfloat16, device_map="auto")
    elif which == "age9b":
        proc, model = load_pretrained_model(model_path=AGE9B_CKPT, model_base=QWEN9B,
                                            model_name=get_model_name_from_path(AGE9B_CKPT),
                                            torch_dtype=torch.bfloat16, device_map="auto")
    elif which == "age9b_mi":
        import glob as _g
        dirs = sorted(_g.glob(os.environ.get("AGE9B_MULTIIMG_GLOB", "")))
        ckpts = sorted(_g.glob(dirs[-1] + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
        ckpt = ckpts[-1]
        print(f"[age9b_mi] using checkpoint: {ckpt}", flush=True)
        proc, model = load_pretrained_model(model_path=ckpt, model_base=QWEN9B,
                                            model_name=get_model_name_from_path(ckpt),
                                            torch_dtype=torch.bfloat16, device_map="auto")
    elif which == "27b":
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForImageTextToText.from_pretrained(
            QWEN27B, attn_implementation="sdpa", quantization_config=bnb, device_map="auto").eval()
        proc = AutoProcessor.from_pretrained(QWEN27B)
    model.eval()
    return proc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["base", "base9b", "ageft", "age9b", "age9b_mi", "27b"])
    ap.add_argument("--rep", required=True)
    ap.add_argument("--frag", default=None, help="fragments json (default: E.FRAG = exp4)")
    ap.add_argument("--tag", default="exp11", help="output tag: out/<tag>_<model>_pred_sh<i>.json")
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rep = best_rep_from_exp10() if args.rep == "auto" else args.rep

    frag_path = args.frag or E.FRAG
    frags = json.load(open(frag_path))[args.shard_idx::args.n_shards]
    if args.limit:
        frags = frags[:args.limit]
    job1 = E.load_job1_index() if rep == "mask" else {}

    proc, model = load_model(args.model)
    device = next(model.parameters()).device
    out_path = os.path.join(OUTDIR, f"{args.tag}_{args.model}_pred_sh{args.shard_idx}.json")
    pm_single = E.prompt_single(rep)

    results = []
    t0 = time.time()
    for c, fr in enumerate(frags):
        session, _ = E.parse_id_from_image(fr["frames"][0]["image"])
        per_frame_imgs = []
        for f in fr["frames"]:
            im = Image.open(os.path.join(E.IMG_ROOT, f["image"])).convert("RGB")
            s, fnum = E.parse_id_from_image(f["image"])
            per_frame_imgs.append(E.frame_images(rep, im, f["box"], job1, s, fnum))
        n = len(per_frame_imgs)
        multi_imgs = [im for imgs in per_frame_imgs for im in imgs]
        try:
            mg = E.parse_json(E.generate(model, proc, multi_imgs, E.prompt_multi(rep, n), device))
        except Exception:
            mg = {}
        singles = []
        for imgs in per_frame_imgs:
            try:
                singles.append(E.parse_json(E.generate(model, proc, imgs, pm_single, device)))
            except Exception:
                singles.append({})
        results.append({"session": session, "gt_pid": fr["gt_pid"], "rep": rep, "model": args.model,
                        "gt_gender": fr["gt_gender"], "gt_age": fr["gt_age"], "n_frames": n,
                        "multi": {"gender": mg.get("gender"), "age": mg.get("age")},
                        "single": [{"gender": s.get("gender"), "age": s.get("age")} for s in singles]})
        if (c + 1) % 20 == 0:
            print(f"[{args.model}/{rep} sh{args.shard_idx}] {c+1}/{len(frags)} "
                  f"({(time.time()-t0)/(c+1):.1f}s/frag)", flush=True)
            json.dump(results, open(out_path, "w"))
    json.dump(results, open(out_path, "w"))
    print(f"[{args.model}/{rep} sh{args.shard_idx}] DONE {len(results)} rep={rep} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
