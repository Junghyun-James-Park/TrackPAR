"""Exp 10: VLM input-representation ablation for age/gender on SAM3 tracks.

Representations (--rep):
  crop        : person box crop (+12% pad)         [current baseline]
  full_prompt : full frame (resized) with the target person's box drawn in red
  full_crop   : two images per frame — full+box AND the crop
  mask        : box crop with non-person pixels grayed out (SAM3 mask from Job1)

Both settings are produced per run: MULTI (all frames in one call) and SINGLE
(each frame independently -> vote/median). Env: qwen35_ft. GPU, shardable.
"""
import argparse
import glob
import json
import os

OUTDIR = os.environ.get("TRACKPAR_OUT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "out"))
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw

# CPU hygiene: the default is one intra-op thread per core (64 here), which both
# oversubscribes the box and thrashes when the job is pinned to a core slice.
# Preprocessing (4K JPEG decode + resize) is the CPU-heavy part; 8 threads saturates
# it. Override with TORCH_THREADS if a run really needs more.
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
from utils import load_pretrained_model, get_model_name_from_path

BASE = OUTDIR
IMG_ROOT = os.environ.get("LOTTE_IMAGES", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/images")
DET_DIR = os.path.join(BASE, "out/detect")
FRAG = os.path.join(BASE, "out/exp4_fragments.json")
MODEL = "Qwen/Qwen3.5-4B"
PAD = 0.12
FULL_LONG = 1024   # resize full image to this long side for full_prompt / full_crop


def parse_id_from_image(image):
    # .../IPS_..._<fnum>.jpg
    b = os.path.splitext(os.path.basename(image))[0]
    m = re.match(r"(.+)_(\d+)$", b)
    return m.group(1), int(m.group(2))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0


# ---- prompts -----------------------------------------------------------------
P_JSON = ("Output exactly one line of JSON and nothing else, no explanation:\n"
          "{\"gender\": \"M\" or \"F\", \"age\": <integer>}")
PM_JSON = ("Output exactly one line of JSON and nothing else, no explanation:\n"
           "{{\"gender\": \"M\" or \"F\", \"age\": <integer>}}")


def prompt_multi(rep, n):
    if rep in ("crop", "mask"):
        return (f"These {n} images are crops of the SAME person from an overhead retail CCTV camera at "
                f"different moments. Judge apparent gender and age from all crops.\n" + PM_JSON)
    if rep == "full_prompt":
        return (f"These {n} overhead CCTV frames show the SAME person, marked with a RED box in each. "
                f"Judge that person's apparent gender and age.\n" + PM_JSON)
    if rep == "full_crop":
        return (f"There are {n} frames of the SAME person; for each frame you get the full scene (target in a "
                f"RED box) followed by a close-up crop. Judge that person's apparent gender and age.\n" + PM_JSON)


def prompt_single(rep):
    if rep in ("crop", "mask"):
        return "This is a crop of a person from an overhead retail CCTV camera. Judge apparent gender and age.\n" + P_JSON
    if rep == "full_prompt":
        return "This overhead CCTV frame has one person marked with a RED box. Judge that person's gender and age.\n" + P_JSON
    if rep == "full_crop":
        return ("You get an overhead CCTV frame (target person in a RED box) then a close-up crop of that person. "
                "Judge their gender and age.\n" + P_JSON)


# ---- image preparation -------------------------------------------------------
def make_crop(im, box, pad=PAD):
    W, H = im.size
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    x1 = max(0, int(x1-pad*bw)); y1 = max(0, int(y1-pad*bh))
    x2 = min(W, int(x2+pad*bw)); y2 = min(H, int(y2+pad*bh))
    return im.crop((x1, y1, x2, y2))


def make_full_boxed(im, box):
    W, H = im.size
    s = FULL_LONG / max(W, H)
    r = im.resize((round(W*s), round(H*s)))
    d = ImageDraw.Draw(r)
    d.rectangle([box[0]*s, box[1]*s, box[2]*s, box[3]*s], outline=(255, 0, 0), width=4)
    return r


def load_job1_index():
    """session -> fnum -> list of (box, rle)."""
    idx = defaultdict(lambda: defaultdict(list))
    for fn in os.listdir(DET_DIR):
        if not fn.endswith(".json"):
            continue
        for r in json.load(open(os.path.join(DET_DIR, fn))):
            s, fnum = parse_id_from_image(r["image"])
            idx[s][fnum] = [(d["box"], d["rle"]) for d in r["detections"]]
    return idx


def make_masked_crop(im, box, job1_idx, session, fnum):
    """box crop with non-person pixels grayed, using best-IoU Job1 SAM3 mask; fallback = plain crop."""
    cands = job1_idx.get(session, {}).get(fnum, [])
    best, brle = 0.4, None
    for b, rle in cands:
        v = iou(box, b)
        if v > best:
            best, brle = v, rle
    crop = make_crop(im, box)
    if brle is None:
        return crop
    from pycocotools import mask as mask_util
    m = mask_util.decode(brle).astype(bool)          # full-res HxW
    W, H = im.size
    x1, y1, x2, y2 = box
    bw, bh = x2-x1, y2-y1
    cx1 = max(0, int(x1-PAD*bw)); cy1 = max(0, int(y1-PAD*bh))
    cx2 = min(W, int(x2+PAD*bw)); cy2 = min(H, int(y2+PAD*bh))
    sub = m[cy1:cy2, cx1:cx2]
    arr = np.array(crop)
    if sub.shape[:2] == arr.shape[:2]:
        arr[~sub] = (arr[~sub] * 0.15 + 128*0.85).astype(np.uint8)  # gray-out background
    return Image.fromarray(arr)


def frame_images(rep, im, box, job1_idx, session, fnum):
    if rep == "crop":
        return [make_crop(im, box)]
    if rep == "mask":
        return [make_masked_crop(im, box, job1_idx, session, fnum)]
    if rep == "full_prompt":
        return [make_full_boxed(im, box)]
    if rep == "full_crop":
        return [make_full_boxed(im, box), make_crop(im, box)]


def parse_json(s):
    s = s.strip()
    if s.startswith("```"):
        s = "\n".join(s.splitlines()[1:]).rstrip("`")
    m = re.search(r"\{.*\}", s, re.DOTALL)
    try:
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


def generate(model, proc, images, text, device, max_new=96):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": text})
    t = proc.apply_chat_template([{"role": "user", "content": content}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ps = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = "left"
    inputs = proc(text=[t], images=images, return_tensors="pt", padding=True).to(device)
    proc.tokenizer.padding_side = ps
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    d = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return (d.split("</think>", 1)[-1] if "</think>" in d else d).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", required=True, choices=["crop", "full_prompt", "full_crop", "mask"])
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    frags = json.load(open(FRAG))[args.shard_idx::args.n_shards]
    if args.limit:
        frags = frags[:args.limit]
    job1_idx = load_job1_index() if args.rep == "mask" else {}

    proc, model = load_pretrained_model(model_path=MODEL, model_base=None,
                                        model_name=get_model_name_from_path(MODEL),
                                        torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    out_path = os.path.join(BASE, f"out/exp10_{args.rep}_pred_sh{args.shard_idx}.json")

    pm_single = prompt_single(args.rep)
    results = []
    t0 = time.time()
    for c, fr in enumerate(frags):
        session, _ = parse_id_from_image(fr["frames"][0]["image"])
        per_frame_imgs = []
        for f in fr["frames"]:
            im = Image.open(os.path.join(IMG_ROOT, f["image"])).convert("RGB")
            s, fnum = parse_id_from_image(f["image"])
            per_frame_imgs.append(frame_images(args.rep, im, f["box"], job1_idx, s, fnum))
        n = len(per_frame_imgs)
        # MULTI: flatten all frames' images
        multi_imgs = [im for imgs in per_frame_imgs for im in imgs]
        try:
            mg = parse_json(generate(model, proc, multi_imgs, prompt_multi(args.rep, n), device))
        except Exception:
            mg = {}
        # SINGLE: per frame
        singles = []
        for imgs in per_frame_imgs:
            try:
                singles.append(parse_json(generate(model, proc, imgs, pm_single, device)))
            except Exception:
                singles.append({})
        results.append({
            "session": session, "gt_pid": fr["gt_pid"],
            "gt_gender": fr["gt_gender"], "gt_age": fr["gt_age"], "n_frames": n,
            "multi": {"gender": mg.get("gender"), "age": mg.get("age")},
            "single": [{"gender": s.get("gender"), "age": s.get("age")} for s in singles],
        })
        if (c+1) % 20 == 0:
            print(f"[{args.rep} sh{args.shard_idx}] {c+1}/{len(frags)} ({(time.time()-t0)/(c+1):.1f}s/frag)", flush=True)
            json.dump(results, open(out_path, "w"))
    json.dump(results, open(out_path, "w"))
    print(f"[{args.rep} sh{args.shard_idx}] DONE {len(results)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
