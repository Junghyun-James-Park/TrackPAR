"""Unified chunk-wise inference: ONE call per track chunk returns
  - identity (gender, age)  : one answer for the whole chunk, broadcast to its frames
  - momentary (exposed, watched) : one answer PER frame

This is the existing short-prompt multi-image setting EXTENDED to also emit the
momentary attributes. Because momentary attrs need scene context (a bare crop can't
tell if the face is toward the camera), two representations are ablated:

  full_box  : K images, each = full frame (resized) with the target in a RED box
  full_mask : 2K images, each frame = [full frame + RED box] then [background-removed
              mask-crop of the target]

Frames are selected on the fly from the track's full frame list, so K and the
selection rule (evenly | size) are both knobs. Env: qwen35_ft. Shardable.

--smoke N : run only N tracks and print input/output TOKEN counts + parse diagnostics
            (this is how we decide K and check for context-length truncation).
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))
import exp10_repr_infer as E
import exp11_model_infer as M

BASE = E.BASE
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(BASE, "out"))
ANNOT = os.environ.get("LOTTE_ANNOT", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/annotations/lotte_tta_sft.json")


def load_img_index():
    img_of = {}
    for it in json.load(open(ANNOT)):
        m = re.match(r"(.+)_(\d+)$", it["id"])
        img_of[(m.group(1), int(m.group(2)))] = it["image"]
    return img_of


def size(b):
    return ((b[2] - b[0]) * (b[3] - b[1])) ** 0.5


def select_frames(all_frames, k, rule):
    af = sorted(all_frames, key=lambda x: x["fnum"])
    n = len(af)
    if n <= k:
        return af
    if rule == "evenly":
        idx = np.linspace(0, n - 1, k).round().astype(int)
        return [af[i] for i in idx]
    # size: largest box within each of k temporal segments (keeps temporal spread)
    edges = np.linspace(0, n, k + 1).round().astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            out.append(max(range(a, b), key=lambda i: size(af[i]["box"])))
    return [af[i] for i in out]


def _view(rep, k):
    if rep == "full_box":
        return (f"These {k} images are frames of the SAME person from an overhead retail CCTV camera at "
                f"different moments. In each frame the target person is marked with a RED box.")
    return (f"You are given {k} frames of the SAME person from an overhead retail CCTV camera at "
            f"different moments. For each frame you get TWO images in this order: first the full scene "
            f"with the target person in a RED box, then a close-up crop of that same person with the "
            f"background removed. So the images are: frame1-scene, frame1-crop, frame2-scene, "
            f"frame2-crop, and so on.")


AGE_MID = {"teens": 15, "20s": 25, "30s": 35, "40s": 45, "50s": 55, "60+": 65}


def build_prompt(rep, k, style="plain"):
    view = _view(rep, k)
    if style in ("plain", "padq"):
        return (
            f"{view}\n\n"
            f"1. Judge the person's GENDER (\"M\" or \"F\") and AGE (integer years) ONCE, from all frames.\n"
            f"2. For EACH of the {k} frames, judge two momentary attributes:\n"
            f"   - exposed: true if the face or the front of the body is visible in THAT frame; "
            f"false if only the back/top of the head is visible.\n"
            f"   - watched: true only if the person is looking UP toward the ceiling camera in THAT frame "
            f"(rare; almost always false).\n\n"
            f"Output exactly one JSON object and nothing else:\n"
            f'{{"gender": "M" or "F", "age": <integer>, '
            f'"frames": [{{"exposed": true or false, "watched": true or false}}]}}  '
            f"— \"frames\" must have exactly {k} entries, in order."
        )
    if style == "trueonly":
        return (
            f"{view}\n\n"
            f"1. Judge the person's GENDER (\"M\" or \"F\") and AGE (integer years) ONCE, from all frames.\n"
            f"2. Momentary attributes are almost always FALSE. Among the {k} frames (numbered 1..{k} in order), "
            f"list ONLY the frame numbers where each is TRUE; every frame not listed is false.\n"
            f"   - exposed: the face or front of the body is visible in that frame.\n"
            f"   - watched: the person is looking UP toward the ceiling camera in that frame (very rare).\n\n"
            f"Output exactly one JSON object and nothing else:\n"
            f'{{"gender": "M" or "F", "age": <integer>, '
            f'"exposed_frames": [frame numbers], "watched_frames": [frame numbers]}}  '
            f"— use [] when none. Frame numbers are 1..{k}."
        )
    if style == "svfd":
        return (
            f"{view}\n"
            f"The camera points straight DOWN, so you usually see the TOP of heads, not faces.\n\n"
            f"── Identity (judge ONCE from all frames) ──\n"
            f"gender_direct: \"M\"/\"F\"/\"unknown\" — use clothing, hair, build, accessories.\n"
            f"age_range: teens|20s|30s|40s|50s|60+|unknown — gray/white hair or stooped posture → older; "
            f"do NOT default to 20s/30s (many shoppers are 40+).\n\n"
            f"── Per frame (one entry for each of the {k} frames, in order) ──\n"
            f"face_visibility:\n"
            f"  clearly_visible : person looking UP at the ceiling; eyes+nose+mouth seen (VERY RARE, <5%).\n"
            f"  partially_visible : BOTH forehead AND nose/chin seen, head tilted up (<15%).\n"
            f"  not_visible : only top/back of head, or hat/hair (MOST COMMON, 75-85%). "
            f"Top-of-head-only or forehead-only → not_visible.\n"
            f"  unknown : person cut off / cannot tell.\n"
            f"watched: \"yes\"/\"no\"/\"unknown\" — looking directly UP at the camera. "
            f"PRIOR: ~99% are \"no\"; only \"yes\" with strong, unambiguous evidence.\n\n"
            f"Output exactly one JSON object and nothing else:\n"
            f'{{"gender_direct": "...", "age_range": "...", '
            f'"frames": [{{"face_visibility": "...", "watched": "..."}}]}}  '
            f"— \"frames\" must have exactly {k} entries, in order."
        )


def _json_from(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.splitlines()[1:]).rsplit("```", 1)[0]
    # PADQ-style prompts answer with a LIST, one entry per detection. The greedy
    # object regex below spans from the first "{" to the last "}" and produces
    # invalid JSON for that shape, so try the list first.
    i = raw.find("[")
    if i >= 0 and (i < raw.find("{") or raw.find("{") < 0):
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw[i:])
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def parse_unified(raw, k, style="plain"):
    d = _json_from(raw)
    if d is None:
        return None
    if isinstance(d, list):
        # PADQ-style: [{"bbox_2d": [...], "exposed": bool}, ...], one entry per
        # frame here. It carries no identity fields, so gender/age stay None and
        # only the attribute the prompt asked about is filled.
        fr = []
        for e in d[:k]:
            if not isinstance(e, dict):
                continue
            fr.append({"exposed": E_bool(e.get("exposed")),
                       "watched": E_bool(e.get("watched"))})
        return {"gender": None, "age": None,
                "frames": fr, "n_frames_returned": len(d)}
    if style == "svfd":
        g = str(d.get("gender_direct", "")).upper()[:1] if d.get("gender_direct") else None
        a = AGE_MID.get(str(d.get("age_range", "")).lower())
        frames = d.get("frames") or []
        fr = []
        for e in frames[:k]:
            if isinstance(e, dict):
                vis = str(e.get("face_visibility", "")).lower()
                fr.append({"exposed": vis in ("clearly_visible", "partially_visible"),
                           "watched": str(e.get("watched", "")).lower() == "yes"})
        return {"gender": g if g in ("M", "F") else None, "age": a,
                "frames": fr, "n_frames_returned": len(frames)}
    if style == "trueonly":
        g = str(d.get("gender", "")).upper()[:1] if d.get("gender") else None
        a = d.get("age")
        ex = set(int(x) for x in (d.get("exposed_frames") or []) if str(x).lstrip("-").isdigit())
        wa = set(int(x) for x in (d.get("watched_frames") or []) if str(x).lstrip("-").isdigit())
        fr = [{"exposed": (i + 1) in ex, "watched": (i + 1) in wa} for i in range(k)]
        return {"gender": g if g in ("M", "F") else None,
                "age": a if isinstance(a, (int, float)) else None,
                "frames": fr, "n_frames_returned": k}
    # plain / padq
    g = str(d.get("gender", "")).upper()[:1] if d.get("gender") else None
    a = d.get("age")
    frames = d.get("frames") or []
    fr = []
    for e in frames[:k]:
        if not isinstance(e, dict):
            continue
        # Observation-style prompts report what is visible and leave the rule to
        # us. metav4 asks the model to apply the definitions itself; these ask it
        # only to look. Same derivation as momentary_targeted_eval, so a prompt
        # scores the same however it is run.
        if "gaze" in e and ("nose" in e or "mouth" in e):
            # The COMBINED schema: eyes/nose/mouth/gaze in one answer. It reports
            # eyes on a three-value scale ("both"/"one"/"none"), not yes/no, so
            # the features branch below scored it False on every frame — exposed
            # F1 0.000 at K=1 against 0.863 for the same answers read through
            # momentary_targeted_eval.derive. This mirrors derive's branch order
            # so the two paths cannot disagree again.
            eyes_ok = str(e.get("eyes", "")).strip().lower() in ("both", "one")
            fr.append({"exposed": eyes_ok,
                       "watched": eyes_ok
                       and str(e.get("gaze", "")).strip().lower() == "up"})
        elif "nose" in e or "mouth" in e:
            y = lambda kk: str(e.get(kk, "")).strip().lower() in ("yes", "true", "1")
            fr.append({"exposed": y("eyes") and y("nose") and y("mouth"),
                       "watched": E_bool(e.get("watched"))})
        elif "eyes" in e or "gaze" in e:
            eyes = str(e.get("eyes", "")).strip().lower()
            gz = str(e.get("gaze", "")).strip().lower()
            fr.append({"exposed": eyes in ("both", "one"),
                       "watched": eyes in ("both", "one") and gz == "up"})
        else:
            fr.append({"exposed": E_bool(e.get("exposed")),
                       "watched": E_bool(e.get("watched"))})
    return {"gender": g if g in ("M", "F") else None,
            "age": a if isinstance(a, (int, float)) else None,
            "frames": fr, "n_frames_returned": len(frames)}


def build_images(rep, fr_sel, im_cache, job1_idx, session):
    imgs = []
    for f in fr_sel:
        im = im_cache[f["fnum"]]
        full = E.make_full_boxed(im, f["box"])
        if rep == "full_box":
            imgs.append(full)
        else:
            imgs.append(full)
            imgs.append(E.make_masked_crop(im, f["box"], job1_idx, session, f["fnum"]))
    return imgs


def E_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return None


def gen_with_tokens(model, proc, images, text, device, max_new=512):
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": text})
    t = proc.apply_chat_template([{"role": "user", "content": content}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ps = proc.tokenizer.padding_side
    proc.tokenizer.padding_side = "left"
    inputs = proc(text=[t], images=images, return_tensors="pt", padding=True).to(device)
    proc.tokenizer.padding_side = ps
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    n_out = out.shape[1] - n_in
    raw = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
    raw = (raw.split("</think>", 1)[-1] if "</think>" in raw else raw).strip()
    return raw, n_in, n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base9b")
    ap.add_argument("--rep", required=True, choices=["full_box", "full_mask"])
    ap.add_argument("--prompt", default="plain", choices=["plain", "padq", "svfd", "trueonly", "meta"])
    ap.add_argument("--prompt_file", default=None, help="for --prompt meta: file whose text (with {K}) is the prompt")
    ap.add_argument("--frag", default=os.path.join(OUTDIR, "phase1_fragments.json"))
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--select", default="evenly", choices=["evenly", "size"])
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--smoke", type=int, default=0, help="run only N GT-matched tracks + print token stats")
    ap.add_argument("--gt_only", action="store_true", help="restrict to GT-matched tracks (for scoring)")
    ap.add_argument("--tag", default="exp20")
    args = ap.parse_args()

    frags = json.load(open(args.frag))
    if args.gt_only or args.smoke:
        frags = [f for f in frags if f.get("gt_gender") and isinstance(f.get("gt_age"), (int, float))]
    if args.smoke:
        frags = frags[:args.smoke]
    else:
        frags = frags[args.shard_idx::args.n_shards]

    img_of = load_img_index()
    job1_idx = E.load_job1_index() if args.rep == "full_mask" else {}
    proc, model = M.load_model(args.model)
    model.eval()
    device = next(model.parameters()).device
    padq_style = False
    if args.prompt == "meta":
        raw_p = open(args.prompt_file).read()
        if "{bboxes}" in raw_p:
            # PADQ-style prompt. It carries its own preamble and names the
            # detections by box, so _view must NOT be prepended (two preambles
            # contradict each other) and {n}/{bboxes} are filled per track, in
            # the loop, from that track's selected frames.
            #
            # ADAPTATION, stated plainly: this prompt was written for ONE image
            # holding n different people. Here it gets K frames of the SAME
            # person. The K boxes are passed as the n detections, so the model
            # returns K entries. The wording is untouched, but "n person(s)"
            # now means "the same person at n moments" — that is a change in
            # meaning, and the K=8 padq row must be read as an adaptation
            # rather than as the prompt in its native setting.
            padq_style = True
            prompt = raw_p
        else:
            prompt = _view(args.rep, args.K) + "\n\n" + raw_p.replace("{K}", str(args.K))
    else:
        prompt = build_prompt(args.rep, args.K, args.prompt)
    if args.smoke:
        print("=" * 70 + f"\nPROMPT ({args.rep}, K={args.K}, {args.prompt}):\n" + prompt + "\n" + "=" * 70, flush=True)

    ptag = "" if args.prompt == "plain" else f"_{args.prompt}"
    out_path = os.path.join(OUTDIR, f"{args.tag}_{args.model}_{args.rep}_K{args.K}_{args.select}{ptag}_sh{args.shard_idx}.json")
    results, tok_in, tok_out, t0 = [], [], [], time.time()
    n_parse_ok = n_frame_ok = 0
    for c, fr in enumerate(frags):
        session = fr["session"]
        allf = fr.get("all_frames") or fr["frames"]
        sel = select_frames(allf, args.K, args.select)
        need = sorted({f["fnum"] for f in sel})
        im_cache = {}
        for fn in need:
            rel = img_of.get((session, fn))
            if rel:
                im_cache[fn] = Image.open(os.path.join(E.IMG_ROOT, rel)).convert("RGB")
        sel = [f for f in sel if f["fnum"] in im_cache]
        imgs = build_images(args.rep, sel, im_cache, job1_idx, session)
        this_prompt = prompt
        if padq_style:
            bbs = []
            for f in sel:
                im0 = im_cache[f["fnum"]]
                x1, y1, x2, y2 = f["box"]
                bbs.append([round(x1 * 1000 / im0.width), round(y1 * 1000 / im0.height),
                            round(x2 * 1000 / im0.width), round(y2 * 1000 / im0.height)])
            this_prompt = (prompt.replace("{n}", str(len(sel)))
                                 .replace("{bboxes}", json.dumps(bbs)))
        raw, n_in, n_out = gen_with_tokens(model, proc, imgs, this_prompt, device, args.max_new)
        pr = parse_unified(raw, len(sel), args.prompt)
        tok_in.append(n_in); tok_out.append(n_out)
        rec = {"session": session, "tid": fr["tid"], "gt_pid": fr.get("gt_pid"),
               "gt_gender": fr.get("gt_gender"), "gt_age": fr.get("gt_age"),
               "K_used": len(sel), "fnums": [f["fnum"] for f in sel],
               "multi": {"gender": pr and pr["gender"], "age": pr and pr["age"]},
               "frames": []}
        if pr:
            n_parse_ok += 1
            if pr["n_frames_returned"] == len(sel):
                n_frame_ok += 1
            for f, fpred in zip(sel, pr["frames"]):
                rec["frames"].append({"fnum": f["fnum"], **fpred})
        else:
            rec["raw_head"] = raw[:200]
        results.append(rec)
        if args.smoke:
            print(f"[{c+1}/{len(frags)}] tid={fr['tid']} tok_in={n_in} tok_out={n_out} "
                  f"parse={'OK' if pr else 'FAIL'} "
                  f"frames_returned={pr['n_frames_returned'] if pr else '-'}/{len(sel)} "
                  f"g={rec['multi']['gender']}/gt {fr.get('gt_gender')} "
                  f"age={rec['multi']['age']}/gt {fr.get('gt_age')}", flush=True)
        elif (c + 1) % 50 == 0:
            print(f"[{args.rep} K{args.K} sh{args.shard_idx}] {c+1}/{len(frags)} "
                  f"({(time.time()-t0)/(c+1):.1f}s/track)", flush=True)

    json.dump(results, open(out_path, "w"))
    print(f"\n=== {args.rep} K={args.K} {args.select} {args.model} ===")
    print(f"tracks: {len(results)}  parse_ok: {n_parse_ok}  frame_count_ok: {n_frame_ok}")
    print(f"tok_in : mean {np.mean(tok_in):.0f}  max {max(tok_in)}")
    print(f"tok_out: mean {np.mean(tok_out):.0f}  max {max(tok_out)}  (max_new={args.max_new})")
    if max(tok_out) >= args.max_new:
        print("  !! some outputs hit max_new — possible truncation, raise --max_new")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
