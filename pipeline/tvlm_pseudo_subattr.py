#!/usr/bin/env python3
"""
Phase 0-2 / V1 — T-VLM sub-attribute pseudo-labels on overhead Lotte crops.

Asks T-VLM (Qwen2.5-VL-72B-AWQ, served by vLLM) for the SAME nested
sub-attribute JSON the UPAR warm-up uses, but on our overhead crops. Then
synthesizes gender + age_group from those sub-attributes and scores them
against Lotte GT — the indirect V1 measurement (Lotte has no sub-attr GT).

    bash scripts/serve_tvlm_vllm.sh &                 # in another shell
    python tvlm_pseudo_subattr.py --limit 150         # V1 smoke  (~1 h)
    python tvlm_pseudo_subattr.py                     # all 2,438 tracks

V1 reads: if overhead sub-attributes survive at all, the synthesized gender
should land near the 0.92-0.93 the direct models already reach. A collapse to
~0.6 means the sub-attribute route does not transfer to overhead, which per the
master plan kills #1/#3 and concentrates everything on #5b.

One crop per track (image-level, per the locked Stage-1 decision): the frame
whose box is largest, i.e. the closest/highest-resolution view — D1 showed
person pixel-size, not face visibility, is what drives attribute error.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upar_build_subattr_trainset import COLORS  # noqa: E402

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
FRAGMENTS = f"{OUTDIR}/phase1_fragments.json"
LOTTE_ROOT = os.environ.get("LOTTE_IMAGES", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/images")
OUT = f"{OUTDIR}/tvlm_pseudo_subattr.json"

# Viewpoint wording: NEUTRAL on purpose. The earlier version asserted
# "ceiling-mounted (overhead, top-down) ... you are looking mostly at the top of
# the head and the shoulders", which the measurements contradict — Lotte box
# aspect ratio is 2.69 vs 2.20 for genuinely aerial UAV-Human and 2.66 for
# ground-level UPAR, i.e. our crops are LESS overhead-looking than real drone
# footage. Telling the model it only sees scalps biases every field.
#
# The last three fields are the momentary sub-attributes. They exist because
# PA100k ships front/side/back, which is a real public analogue of our `exposed`;
# the same decomposition is asked of the VLM here so the two teachers stay
# comparable, and so exposed/watched can be DERIVED from sub-attributes rather
# than asked for directly (the SVFD-style route).
PROMPT = """This is a cropped view of ONE person from an elevated retail CCTV camera.
The camera angle varies between cameras and across the floor, so how much of the
person is visible differs from crop to crop.

Report what you can see, using ONLY the allowed values:
- gender: "male" | "female"
- age_group: "young" | "adult" | "old"
- hair.length: "short" | "long" | "bald" | "unknown"
- upper.length: "short" | "long"                (sleeve length)
- upper.color: one of [{colors}]
- lower.length: "short" | "long"
- lower.color: one of [{colors}]
- lower.type: "trousers" | "skirt"
- accessories: a list from ["backpack", "bag", "glasses", "sunglasses", "hat"]; [] if none
- orientation: "front" | "side" | "back"        (which way the body faces the camera)
- face_visible: "clear" | "partial" | "none"    (enough of the face to identify the person?)
- gaze: "up" | "level" | "down" | "unknown"     ("up" = looking up toward the ceiling camera; rare)

Use "unknown" for hair.length only when the hair genuinely is not discernible. If
part of the body is cut off or occluded, still give your best estimate for those
fields. Be conservative with gaze="up" — most people are not looking at the camera.

Output exactly one line of JSON and nothing else, no explanation:
{{"gender": ..., "age_group": ..., "hair": {{"length": ...}}, "upper": {{"length": ..., "color": ...}}, "lower": {{"length": ..., "color": ..., "type": ...}}, "accessories": [...], "orientation": ..., "face_visible": ..., "gaze": ...}}""".format(
    colors=", ".join(f'"{c}"' for c in COLORS)
)

# GT age (integer) -> the coarse UPAR bucket, so the two are comparable at all.
# UPAR's own boundaries: young < 18, adult 18-60, old > 60.
def age_bucket(age):
    if age is None or age <= 0:
        return None
    if age < 18:
        return "young"
    if age <= 60:
        return "adult"
    return "old"


_GT_CACHE = {}


def gt_momentary(image, box, iou_thr=0.5):
    """Frame-level GT exposed/watched for one crop.

    phase1_fragments.json carries only track-level gt_gender/gt_age, so momentary
    GT has to come from the raw CSV and be matched by (image, box) IoU — the
    boxes there are the annotator's, not SAM3's, so they only overlap.
    Returns (exposed, watched) or (None, None) when no box matches.
    """
    # The CSV stores absolute paths under the dead /data1 mount
    # ("/data1/datasets/lotte_cheonho/images/lotte/1-image/<sess>/<file>.jpg")
    # while fragments store "lotte/1-image/<sess>/<file>.jpg". Key both on the
    # "lotte/..." suffix.
    def _key(p):
        # NOTE: "lotte/", not "lotte/1-image/". The dataset has 1-image, 2-image
        # AND 3-image subdirs (9,495 / 3,038 / 3,376 rows). Keying on 1-image
        # silently dropped 40% of the GT rows.
        i = p.find("lotte/")
        return p[i:] if i >= 0 else p

    if not _GT_CACHE:
        import csv as _csv
        with open(os.environ.get("LOTTE_CSV", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/lotte_tta.csv"),
                  encoding="utf-8-sig", newline="") as fh:
            for r in _csv.DictReader(fh):
                _GT_CACHE.setdefault(_key(r["image_path"]), []).append((
                    float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                    str(r["exposed"]).strip().lower() in ("true", "1"),
                    str(r["watched"]).strip().lower() in ("true", "1")))

    cands = _GT_CACHE.get(_key(image))
    if not cands:
        return None, None
    bx1, by1, bx2, by2 = box
    best, best_iou = None, 0.0
    for x1, y1, x2, y2, ex, wa in cands:
        ix = max(0.0, min(bx2, x2) - max(bx1, x1))
        iy = max(0.0, min(by2, y2) - max(by1, y1))
        inter = ix * iy
        union = (bx2-bx1)*(by2-by1) + (x2-x1)*(y2-y1) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = (ex, wa), iou
    return best if best_iou >= iou_thr else (None, None)


def derive_momentary(o):
    """sub-attributes -> exposed / watched (the SVFD-style synthesis step).

    exposed = the face is identifiable, i.e. body facing the camera AND the face
    not fully hidden. watched = gaze up at the ceiling camera.
    """
    if not o:
        return None, None
    fv, orient, gaze = o.get("face_visible"), o.get("orientation"), o.get("gaze")
    exposed = None
    if fv in ("clear", "partial", "none"):
        exposed = fv == "clear" or (fv == "partial" and orient == "front")
    watched = None
    if gaze in ("up", "level", "down", "unknown"):
        watched = gaze == "up"
    return exposed, watched


def largest_box_frame(frames):
    def area(f):
        x1, y1, x2, y2 = f["box"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(frames, key=area)


def crop_pil(frame, pad=0.12):
    """The single shared crop path — tpar_pseudo_subattr.py imports this so both
    teachers see byte-identical inputs."""
    path = os.path.join(LOTTE_ROOT, frame["image"].replace("lotte/", "", 1))
    if not os.path.exists(path):
        path = os.path.join(LOTTE_ROOT, frame["image"])
    im = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = frame["box"]
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(im.width, x2 + pad * w), min(im.height, y2 + pad * h)
    im = im.crop((int(x1), int(y1), int(x2), int(y2)))
    # Upscale tiny crops: the model needs pixels, and overhead crops are small.
    if im.width * im.height < 160 * 320:
        s = (160 * 320 / max(1, im.width * im.height)) ** 0.5
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))),
                       Image.BICUBIC)
    return im


def crop_b64(frame, pad=0.12):
    buf = io.BytesIO()
    crop_pil(frame, pad).save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def stratified(tracks, limit, seed=0):
    """Round-robin across sessions so one scene cannot decide the gate — D2 found
    gender ranges 0.76-0.98 across the 11 sessions."""
    import random
    rng = random.Random(seed)
    by_sess = {}
    for t in tracks:
        by_sess.setdefault(t["session"], []).append(t)
    for v in by_sess.values():
        rng.shuffle(v)
    picked, i = [], 0
    while len(picked) < limit:
        added = False
        for s in sorted(by_sess):
            if i < len(by_sess[s]) and len(picked) < limit:
                picked.append(by_sess[s][i])
                added = True
        if not added:
            break
        i += 1
    return picked


def score_v1(results, label=""):
    """The V1 readout, shared by both teachers."""
    ok = [r for r in results if r.get("subattr")]
    print(f"\n===== V1 {label} =====")
    print(f"parse-valid {len(ok)}/{len(results)} "
          f"({100*len(ok)/max(1,len(results)):.1f}%)")

    gm = [(r["subattr"].get("gender"), r["gt_gender"]) for r in ok]
    gm = [(p, g) for p, g in gm if p in ("male", "female")]
    if gm:
        acc = sum(1 for p, g in gm
                  if (p == "female") == (str(g).upper().startswith("F"))) / len(gm)
        print(f"gender acc      {acc:.3f}  (n={len(gm)})  "
              f"[ref: direct FT-9B 0.929 / base-9B unified 0.92]")
        print(f"  pred dist {Counter(p for p, _ in gm).most_common()}")

    ab = [(r["subattr"].get("age_group"), age_bucket(r.get("gt_age"))) for r in ok]
    ab = [(p, g) for p, g in ab if g and p in ("young", "adult", "old")]
    if ab:
        print(f"age-bucket acc  {sum(1 for p,g in ab if p==g)/len(ab):.3f}  (n={len(ab)})")
        print(f"  pred dist {Counter(p for p, _ in ab).most_common()}")
        print(f"  gt   dist {Counter(g for _, g in ab).most_common()}")

    print("--- sub-attribute distributions (is anything surviving overhead?) ---")
    for key, get in [
        ("hair.length", lambda o: (o.get("hair") or {}).get("length")),
        ("upper.length", lambda o: (o.get("upper") or {}).get("length")),
        ("upper.color", lambda o: (o.get("upper") or {}).get("color")),
        ("lower.color", lambda o: (o.get("lower") or {}).get("color")),
        ("lower.type", lambda o: (o.get("lower") or {}).get("type")),
    ]:
        c = Counter(get(r["subattr"]) for r in ok)
        if not c or set(c) == {None}:
            print(f"  {key:12s} (no values — teacher vocabulary or zero parses)")
            continue
        top = c.most_common(5)
        dom = 100 * top[0][1] / max(1, sum(c.values()))
        print(f"  {key:12s} {top}   dominant {dom:.0f}%")
    acc_c = Counter(a for r in ok for a in (r["subattr"].get("accessories") or []))
    print(f"  accessories  {acc_c.most_common()}")
    for k in ("orientation", "face_visible", "gaze", "_orientation"):
        c = Counter(r["subattr"].get(k) for r in ok if r["subattr"].get(k))
        if c:
            print(f"  {k:12s} {c.most_common()}")

    # ---- momentary derived from sub-attributes, scored on frame-level GT ----
    def f1(pred, gt):
        tp = sum(1 for p, g in zip(pred, gt) if p and g)
        fp = sum(1 for p, g in zip(pred, gt) if p and not g)
        fn = sum(1 for p, g in zip(pred, gt) if (not p) and g)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc, tp, tp + fn

    rows = [(derive_momentary(r["subattr"]), r.get("gt_exposed"), r.get("gt_watched"))
            for r in ok]
    for i, name, ref in ((0, "exposed", "SVFD-v10 0.463 / meta-prompt 0.396"),
                         (1, "watched", "SVFD-v10 0.571 / trueonly 0.185")):
        pairs = [(d[i], (ge, gw)[i]) for d, ge, gw in rows
                 if d[i] is not None and (ge, gw)[i] is not None]
        if not pairs:
            print(f"  {name}: no GT-matched frames")
            continue
        p, g = zip(*pairs)
        s, pr, rc, tp, npos = f1(p, g)
        print(f"  {name:8s} F1 {s:.3f}  P {pr:.3f} R {rc:.3f}  "
              f"(pos {npos}/{len(pairs)}, tp {tp})   [ref: {ref}]")


class TransformersChat:
    """4-bit local VLM, the HANDOFF §5 recipe.

    Critical loading rules from that section, learned the hard way on this box:
    pass `device_map="auto"` and DO NOT pass a dtype — `device_map={"":0}` or an
    explicit dtype both blow up to ~47 GB and OOM.
    """

    def __init__(self, model_path, adapter=None, quantize=True):
        import torch
        from transformers import (AutoModelForImageTextToText, AutoProcessor,
                                  BitsAndBytesConfig)
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/mnt/nvme0n1p1/pjh/.cache/huggingface"))
        torch.set_num_threads(6)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        self.torch = torch
        # Resolution floor/ceiling for the vision tower, controllable from the
        # environment. Qwen's processor downsamples a crop toward min_pixels and
        # upsamples toward it too, so raising the floor forces a small crop to be
        # interpolated up rather than passed through at its native size.
        #
        # This exists because an earlier "resolution sweep" set IMG_MIN_PX in the
        # shell and nothing read it: three runs at 256/1024/2048 returned
        # byte-identical numbers, which is the signature of a knob that is not
        # connected rather than of a knob that does not matter.
        # Qwen2VLImageProcessorFast expresses this as size={"shortest_edge",
        # "longest_edge"} in PIXEL AREA, not as min_pixels/max_pixels. Passing
        # the latter is silently ignored, which is how the first attempt produced
        # three byte-identical runs.
        #
        # The default floor is 65,536 px of area. The crops that defeat every
        # method sit at 45k-63k, so they are ALREADY being upsampled by the
        # processor; raising the floor asks whether more interpolation helps.
        self.proc = AutoProcessor.from_pretrained(model_path)
        _lo, _hi = os.environ.get("IMG_MIN_PX"), os.environ.get("IMG_MAX_PX")
        if _lo or _hi:
            sz = dict(getattr(self.proc.image_processor, "size", None) or {})
            if _lo:
                sz["shortest_edge"] = int(_lo)
            if _hi:
                sz["longest_edge"] = int(_hi)
            self.proc.image_processor.size = sz
            print(f"processor resolution override: {sz}", flush=True)

        # `device_map="auto"` alone packs GPU0 to ~47 GiB and OOMs on the 72B
        # (4-bit is ~40 GB, which does not fit one 48 GB card). An explicit
        # per-device budget forces the split across both visible GPUs and leaves
        # headroom for activations and the vision tower.
        n = torch.cuda.device_count()
        budget = int(torch.cuda.get_device_properties(0).total_memory
                     / 1024**3 * 0.78)
        max_memory = {i: f"{budget}GiB" for i in range(n)}
        print(f"loading on {n} GPU(s), budget {budget}GiB each", flush=True)

        kw = dict(attn_implementation="sdpa", device_map="auto",
                  max_memory=max_memory)
        if quantize:
            kw["quantization_config"] = bnb
        else:
            kw["dtype"] = torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kw).eval()

        # A #5b student is a LoRA adapter on the 9B base; evaluate it through the
        # exact same crop/prompt/parse path the teachers used, so student and
        # teacher numbers are directly comparable.
        if adapter:
            # These runs also train the vision tower and merger (456 M params,
            # written to non_lora_state_dict.bin). PeftModel alone drops them and
            # would evaluate a model that was never trained — see
            # rap2_eval.attach_adapter and src/utils.py:68.
            from rap2_eval import attach_adapter
            self.model = attach_adapter(self.model, adapter)
            print(f"applied adapter {adapter}", flush=True)
        dev = {str(p.device) for p in self.model.parameters()}
        print(f"loaded {model_path} across {sorted(dev)}", flush=True)

    # Prefill the assistant turn with the opening of the target JSON. Without it
    # Qwen3.5-27B spent its whole budget reasoning in prose ("Based on the visual
    # evidence... 1. **Gender:** ...") and got truncated before emitting any JSON:
    # 150/150 unparseable. Prefilling forces the completion to BE the JSON.
    PREFILL = '{"gender": "'

    def ask(self, pil_image, prompt, max_new_tokens=320, prefill=None):
        """`pil_image` may be a single PIL image or a list of them. The list form
        is the `full_mask` representation the earlier momentary work actually used:
        the whole scene with the target boxed, followed by the tight crop. A lone
        downscaled full frame is not a fair test — at 4K->1280 the person shrinks
        to ~83x212 px, which exp10 already identified as the worst representation."""
        """`prefill` must match the schema the prompt asks for. The class default
        opens the sub-attribute object; a direct-ask prompt needs its own opening,
        otherwise the prefill drags the model back into the wrong schema — which
        silently invalidated one whole control run."""
        imgs = pil_image if isinstance(pil_image, (list, tuple)) else [pil_image]
        msg = [{"role": "user", "content":
                [{"type": "image", "image": im} for im in imgs]
                + [{"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msg, tokenize=False,
                                             add_generation_prompt=True)
        pre = self.PREFILL if prefill is None else prefill
        text += pre
        inp = self.proc(text=[text], images=list(imgs),
                        return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        gen = self.proc.decode(out[0][inp["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        return pre + gen

    def ask_many(self, images, prompt, max_new_tokens=320, prefill=None,
                 batch=8):
        """Batched form of `ask` for whole-set evaluations.

        One-at-a-time generation across a pipeline-parallel model leaves the
        GPUs at 16-32%: only one shard computes at any instant and there is no
        work to overlap. A 349-track holdout took ~60 min that way. The same
        batched path in rap2_eval measures 0.76 s/image, so batching turns each
        holdout eval from an hour into minutes.

        Left padding is mandatory — with right padding every sequence starts
        generating at a different offset and the completions are garbage.

        Each element of `images` is a single PIL image or a list of them
        (the full_mask representation). Returns strings in input order, so
        callers keep their existing per-item bookkeeping.
        """
        self.proc.tokenizer.padding_side = "left"
        pre = self.PREFILL if prefill is None else prefill
        out_txt = []
        for s in range(0, len(images), batch):
            chunk = [im if isinstance(im, (list, tuple)) else [im]
                     for im in images[s:s + batch]]
            texts, flat = [], []
            for imgs in chunk:
                msg = [{"role": "user", "content":
                        [{"type": "image", "image": im} for im in imgs]
                        + [{"type": "text", "text": prompt}]}]
                texts.append(self.proc.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True) + pre)
                flat.extend(imgs)
            inp = self.proc(text=texts, images=flat, return_tensors="pt",
                            padding=True).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=max_new_tokens,
                                          do_sample=False)
            out_txt += [pre + g for g in self.proc.batch_decode(
                out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out_txt


def parse_json(text):
    """Take the FIRST complete JSON object.

    A greedy r"\\{.*\\}" is wrong here: the models reliably emit the object and
    then repeat it on a second line, so the greedy span covers both and
    json.loads dies with "Extra data". raw_decode stops at the first object.
    """
    i = text.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="V1 smoke: N tracks, stratified across sessions")
    ap.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", 8311)))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--model", default=None,
                    help="transformers backend: local path or hub id")
    ap.add_argument("--label", default=None, help="name to print in the V1 readout")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (a #5b student)")
    ap.add_argument("--no-quant", action="store_true", help="load in bf16, not 4-bit")
    ap.add_argument("--batch", type=int, default=1,
                    help="transformers backend: generation batch size. Default 1 "
                         "ON PURPOSE — armA/D1/D2/S1/S2/S3 were all measured "
                         "one-at-a-time, and left-padded batching can shift "
                         "completions. Raising it silently would make new arms "
                         "incomparable to those. Pass --batch 8 only after "
                         "verifying the two paths agree.")
    ap.add_argument("--holdout", default=None,
                    help="json list of track ids; evaluate ONLY these (leakage-free split)")
    args = ap.parse_args()

    # vLLM cannot run on this box (compute capability 12.0 + driver 570 ->
    # cudaErrorUnsupportedPtxVersion), so the transformers + bitsandbytes 4-bit
    # path from HANDOFF §5 is the working one. Kept both: the vllm branch is
    # correct and will work wherever the PTX issue does not apply.
    gen = None
    if args.backend == "transformers":
        gen = TransformersChat(args.model, adapter=args.adapter,
                               quantize=not args.no_quant)
    else:
        from openai import OpenAI
        client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="none")

    tracks = json.load(open(FRAGMENTS))
    # V1 scores against GT, so only GT-matched tracks are usable.
    tracks = [t for t in tracks if t.get("gt_pid") is not None
              and t.get("gt_gender") and t.get("frames")]

    if args.holdout:
        keep = set(json.load(open(args.holdout)))
        tracks = [t for t in tracks if t["tid"] in keep]
        print(f"held-out subset: {len(tracks)} tracks, "
              f"{len(set(t['session'] for t in tracks))} sessions")
    elif args.limit:
        tracks = stratified(tracks, args.limit, args.seed)

    print(f"{len(tracks)} tracks; {len(set(t['session'] for t in tracks))} sessions",
          flush=True)

    # Batched fast path for the local transformers backend. The per-track loop
    # below generates one image at a time, which leaves a pipeline-parallel 9B at
    # 16-32% GPU and made a 349-track holdout take ~60 min. Same prompt, same
    # prefill, same parser — only the generate() call is grouped.
    pre_txt = {}
    if gen is not None:
        crops, keys = [], []
        for t in tracks:
            fr = largest_box_frame(t["frames"])
            try:
                crops.append(crop_pil(fr))
                keys.append(t["tid"])
            except Exception:
                pass          # the loop below reports and counts it
        if crops:
            print(f"batched generation over {len(crops)} crops", flush=True)
            import time as _t
            t0 = _t.time()
            for txt, k in zip(gen.ask_many(crops, PROMPT, batch=args.batch), keys):
                pre_txt[k] = txt
            print(f"  done in {(_t.time()-t0)/60:.1f} min "
                  f"({(_t.time()-t0)/len(crops):.2f} s/track)", flush=True)

    results, bad = [], 0
    for n, t in enumerate(tracks, 1):
        fr = largest_box_frame(t["frames"])
        try:
            im = crop_pil(fr)
        except Exception as e:
            bad += 1
            print(f"  crop fail {t['tid']}: {e}", flush=True)
            continue
        try:
            if gen is not None:
                txt = pre_txt.get(t["tid"]) or gen.ask(im, PROMPT)
            else:
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=92)
                b64 = base64.b64encode(buf.getvalue()).decode()
                r = client.chat.completions.create(
                    model="tvlm",
                    messages=[{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": PROMPT},
                    ]}],
                    max_tokens=200, temperature=0.0,
                )
                txt = r.choices[0].message.content
        except Exception as e:
            bad += 1
            print(f"  api fail {t['tid']}: {e}", flush=True)
            continue

        obj = parse_json(txt)
        ge, gw = gt_momentary(fr["image"], fr["box"])
        results.append({
            "session": t["session"], "tid": t["tid"], "gt_pid": t["gt_pid"],
            "gt_gender": t["gt_gender"], "gt_age": t.get("gt_age"),
            "gt_exposed": ge, "gt_watched": gw,
            "frame": fr["image"], "box": fr["box"],
            "subattr": obj, "raw": None if obj else txt,
        })
        if n % 25 == 0:
            print(f"  {n}/{len(tracks)}", flush=True)

    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(results)} -> {args.out}  (failed {bad})", flush=True)

    score_v1(results, label=args.label or f"T-VLM ({args.model or 'vLLM'})")


if __name__ == "__main__":
    main()
