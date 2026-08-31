#!/usr/bin/env python3
"""Momentary (exposed / watched) — a targeted, actually-measurable evaluation.

The V1 run could not score momentary at all: its 150 stratified tracks GT-matched
53 frames and contained ZERO exposed positives. That is a sampling problem, not a
model problem — exposed is 8.5% of frames overall and, as D2 found, is
concentrated in 4 of the 11 sessions (1,353 of all 1,356 positives). watched is
1.3%, with 158 of 201 positives in one single session.

So this samples person INSTANCES straight from the GT csv, balanced positive vs
negative, drawn from the sessions where the events actually occur.

The question it answers: does deriving exposed/watched from sub-attributes
(orientation / face_visible / gaze) beat asking for them directly?
References from S4 on the same attributes: SVFD-v10 exposed F1 0.463, the
VLM-written "meta" prompt 0.396, plain prompt 0.190.

    python momentary_targeted_eval.py --n-pos 200 --n-neg 200
"""
import argparse
import csv
import json
import os
import random
import re
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
sys.path.insert(0, JOBS)
from PIL import Image, ImageDraw  # noqa: E402

import tvlm_pseudo_subattr as tv  # noqa: E402


def full_box_image(frame, max_w=1280):
    """Full scene with the target in a red box — the representation SVFD-v10 and the
    K=8 runs actually used. HANDOFF §9g reports a bare crop caps near F1 0.21 for
    momentary because it discards the scene/gaze context, so a crop-only result
    cannot be compared against those references. This gives the missing arm."""
    path = os.path.join(tv.LOTTE_ROOT, frame["image"].replace("lotte/", "", 1))
    if not os.path.exists(path):
        path = os.path.join(tv.LOTTE_ROOT, frame["image"])
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = frame["box"]
    w = max(6, int(0.004 * im.width))
    d.rectangle([x1, y1, x2, y2], outline=(255, 30, 30), width=w)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    return im

CSV = os.environ.get("LOTTE_CSV", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/lotte_tta.csv")

# The control this evaluation was missing: ask for exposed/watched DIRECTLY on the
# exact same single-crop sample. The published 0.463 / 0.396 baselines were measured
# on full scenes and on K=8 multi-image respectively, and HANDOFF §9g already found a
# bare crop caps out near F1 0.21 because it drops the scene/gaze context. So a
# sub-attribute number can only be judged against a direct-ask number on the SAME
# representation.
DIRECT_PROMPT = """This is a cropped view of ONE person from an elevated retail CCTV camera.

Answer two questions about THIS person in THIS frame:
- exposed: true only when the face or front identity cues are clearly visible enough to
  identify the person (person facing the camera); false if the back is turned or the face
  is obscured. Roughly 8% of frames are true.
- watched: true only when the person appears to be looking up toward the ceiling camera.
  This is extremely rare — about 1% of frames. If uncertain, answer false.

Output exactly one line of JSON and nothing else:
{"exposed": true or false, "watched": true or false}"""


# ------------------------------------------------------- T4 calibration arms
#
# The 2x3 representation grid closed with a specific diagnosis: exposed is
# predicted at 50-78% against a true rate of 7.6% (6.6-9.9x over-call) while
# recall stays at 0.55-0.82. The model SEES the cue; it applies the wrong
# threshold. Note that DIRECT_PROMPT above already states the 8% prior in words
# and still over-calls 8.5x, so "tell it the prior" is the CONTROL here, not a
# treatment. Each arm below therefore changes a different mechanism:
#
#   prior    — makes the bar operational (default-false + an explicit rate
#              self-check) instead of quoting a percentage
#   tier     — moves the decision out of the model: it reports a graded
#              observability tier, WE apply the threshold
#   graded   — same idea pushed to a continuous score, so the operating point
#              is chosen rather than assumed
#   trueonly — sparse output format for a sparse prior; already the best
#              watched arm in the earlier prompt work

PRIOR_PROMPT = """This is a cropped view of ONE person from an elevated retail CCTV camera.

Your default answer is FALSE. Only override it when the evidence is unmistakable.

- exposed: true ONLY if you can actually make out the person's facial features
  (eyes, nose, mouth) well enough to recognise them again. A visible head is not
  enough. The back or top of a head, a blurred face, or a face turned away is FALSE.
- watched: true ONLY if the person's face is tilted up toward the ceiling camera
  and their eyes are directed at it. Anything else is FALSE.

Calibration check before you answer: in this camera's footage only about 8 people
in 100 are exposed, and about 1 in 100 is watched. If you find yourself answering
true for more than roughly 1 person in 12, your bar is too low — raise it.

Output exactly one line of JSON and nothing else:
{"exposed": true or false, "watched": true or false}"""

TIER_PROMPT = """This is a cropped view of ONE person from an elevated retail CCTV camera.

Do NOT judge whether the person is identifiable. Report only what you observe,
using these exact scales:

- body_orientation: "front" | "side" | "back"        (which way the torso faces)
- face_visibility: "none" | "back_of_head" | "partial" | "clear"
    none          = no head visible at all
    back_of_head  = only hair / the back or top of the head
    partial       = part of the face visible but features are not resolvable
    clear         = eyes, nose and mouth are all distinguishable
- head_pitch: "down" | "level" | "up"                (where the head is aimed)
- gaze_at_camera: "yes" | "no" | "unclear"

Output exactly one line of JSON and nothing else:
{"body_orientation": ..., "face_visibility": ..., "head_pitch": ..., "gaze_at_camera": ...}"""

GRADED_PROMPT = """This is a cropped view of ONE person from an elevated retail CCTV camera.

Rate each statement from 0 to 100, where 0 = certainly not and 100 = certainly yes.
Use the whole range; do not round to 0 or 100.

- exposed_score: how clearly the person's facial features are visible enough to
  identify them.
- watched_score: how likely the person is looking up at the ceiling camera.

Output exactly one line of JSON and nothing else:
{"exposed_score": <0-100 integer>, "watched_score": <0-100 integer>}"""

TRUEONLY_PROMPT = """This is a cropped view of ONE person from an elevated retail CCTV camera.

List ONLY the labels that clearly apply. Most people match NONE of them, and an
empty list is the normal, expected answer.

- "exposed"  : facial features are clearly visible enough to identify the person
- "watched"  : the person is looking up toward the ceiling camera

Output exactly one line of JSON and nothing else:
{"labels": []}   or   {"labels": ["exposed"]}   etc."""

STYLES = {
    "prior":    (PRIOR_PROMPT,    '{"exposed": '),
    "tier":     (TIER_PROMPT,     '{"body_orientation": "'),
    "graded":   (GRADED_PROMPT,   '{"exposed_score": '),
    # NOT '{"labels": [' — opening the bracket makes a string the natural next
    # token and structurally suppresses `[]`, which is the exact answer this arm
    # exists to elicit. A 6-sample smoke with that prefill labelled 6/6 exposed
    # and 0 empty lists. Stopping at the colon still forces JSON (which is all
    # the prefill is for) while leaving `[]` reachable.
    "trueonly": (TRUEONLY_PROMPT, '{"labels":'),
}


def tier_rule(obj):
    """TIER_PROMPT's observability fields -> the two booleans.

    The threshold lives HERE, in code we can inspect and move, rather than
    inside the model's own notion of "clearly visible". `clear` alone is the
    exposed bar: admitting `partial` is what produced the 6.6x over-call, since
    the models label almost every visible head partial.
    """
    fv = str(obj.get("face_visibility", "")).strip().lower()
    gz = str(obj.get("gaze_at_camera", "")).strip().lower()
    pitch = str(obj.get("head_pitch", "")).strip().lower()
    exposed = fv == "clear"
    watched = exposed and gz == "yes" and pitch == "up"
    return exposed, watched


def f1(pred, gt):
    tp = sum(1 for p, g in zip(pred, gt) if p and g)
    fp = sum(1 for p, g in zip(pred, gt) if p and not g)
    fn = sum(1 for p, g in zip(pred, gt) if (not p) and g)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return (2 * pr * rc / (pr + rc) if pr + rc else 0.0), pr, rc, tp, fp, fn


def derive(obj, name):
    """Derive the attribute from whatever schema the prompt emits.

    The prompts do not agree on an output shape, and the deployment
    parser picks its branch from the --prompt flag rather than from the
    content. Feeding a prompt through the file path there silently routes
    it to the wrong branch: svfd reports face_visibility and trueonly
    reports frame-number lists, and the plain branch looks for
    frames[].exposed in both, finding nothing. Here the shape is read off
    the answer, so any prompt can go through one path.

    The rules below are copied from exp20_unified_infer.parse_unified so
    a prompt scores identically whichever path runs it."""
    fr0 = obj.get("frames")
    f0 = (fr0[0] if isinstance(fr0, list) and fr0 and isinstance(fr0[0], dict)
          else obj)

    # The combined observation prompt reports eyes, nose, mouth AND
    # gaze, so both attributes come from one call and several exposed
    # rules can be compared afterwards from the same stored answers
    # without re-running. `eyes` alone is the primary rule here because
    # it beat the three-way conjunction on 09.20 (0.871 vs 0.828); the
    # alternatives are scored in analysis.
    if "gaze" in f0 and ("nose" in f0 or "mouth" in f0):
        eyes = str(f0.get("eyes", "")).strip().lower()
        if name == "watched":
            return (eyes in ("both", "one")
                    and str(f0.get("gaze", "")).strip().lower() == "up")
        return eyes in ("both", "one")

    # svfd: a visibility tier per frame, plus a yes/no gaze
    if "face_visibility" in f0:
        if name == "exposed":
            return str(f0.get("face_visibility", "")).lower() in (
                "clearly_visible", "partially_visible")
        return str(f0.get("watched", "")).lower() == "yes"

    # trueonly: lists the 1-based frame numbers that are true
    if "exposed_frames" in obj or "watched_frames" in obj:
        key = "exposed_frames" if name == "exposed" else "watched_frames"
        nums = {int(x) for x in (obj.get(key) or [])
                if str(x).lstrip("-").isdigit()}
        return 1 in nums          # this path shows a single frame

    return _derive_observation(obj, name)

def _derive_observation(obj, name):
    # Surface-observation prompts report `eyes` and `gaze` and leave the
    # decision to us. This is the T4 pattern: every step that moved the
    # threshold out of the model improved calibration monotonically, and
    # metav4's watched clause asks the model to apply a four-part rule
    # ("pupils visible AND directed straight up AND at the lens, not
    # merely upward") in one shot.
    fr0 = obj.get("frames")
    f0 = fr0[0] if isinstance(fr0, list) and fr0 and isinstance(fr0[0], dict) else obj
    if "nose" in f0 or "mouth" in f0:
        # metav4 defines exposed as eyes AND nose AND mouth resolvable
        # simultaneously, and asks the model to apply that conjunction.
        # Here the model reports the three features and the conjunction
        # happens in code — the same move that lifted watched.
        y = lambda k: str(f0.get(k, "")).strip().lower() in ("yes", "true", "1")
        if name == "exposed":
            return y("eyes") and y("nose") and y("mouth")
        return None
    # `eyes` must be present, not merely `gaze`. The subattr schema reports
    # gaze without eyes, so matching on gaze alone made this branch return
    # `eyes in ("both","one")` == False for every subattr answer and scored
    # that prompt at 0.000 where its own eval log read 0.855.
    if "eyes" in f0:
        eyes = str(f0.get("eyes", "")).strip().lower()
        gz = str(f0.get("gaze", "")).strip().lower()
        if name == "watched":
            return eyes in ("both", "one") and gz == "up"
        # exposed = an eye is resolvable at all
        return eyes in ("both", "one")
    fr = obj.get("frames")
    if isinstance(fr, list) and fr and isinstance(fr[0], dict) \
            and name in fr[0]:
        v = fr[0][name]
    elif name in obj:
        v = obj[name]
    else:
        # Native --style runs (subattr and friends) emit the sub-attribute schema
        # this file's own scoring path reads through tv.derive_momentary. Falling
        # back to it here keeps ONE derivation for both: without this, anything
        # re-scoring a native-style run through derive() reads subattr as 0.000
        # across the board while the eval log shows 0.855.
        import tvlm_pseudo_subattr as _tv
        got = _tv.derive_momentary(obj)
        return got[0 if name == "exposed" else 1]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default="base-9B sub-attribute derivation")
    ap.add_argument("--n-pos", type=int, default=200)
    ap.add_argument("--n-neg", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repr", dest="repr_", default="crop",
                    choices=["crop", "full_box", "full_mask"],
                    help="crop = tight person crop; full_box = whole frame with the "
                         "target boxed in red (what the 0.463/0.396 refs used)")
    ap.add_argument("--style", default="subattr",
                    choices=["subattr", "direct"] + list(STYLES),
                    help="subattr = derive exposed/watched from orientation/face_visible/"
                         "gaze; direct = ask for them straight out (the control); "
                         "prior/tier/graded/trueonly = the T4 calibration arms")
    ap.add_argument("--target", default="exposed", choices=["exposed", "watched"],
                    help="which attribute to balance the sample on")
    ap.add_argument("--natural", action="store_true",
                    help="sample uniformly from ALL sessions instead of balancing. "
                         "The balanced runs give bAcc; only this gives an F1 that is "
                         "comparable to the SVFD-v10 / meta-prompt references, which "
                         "were all measured at natural prevalence.")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N of each class — use to get a DISJOINT "
                         "sample from an earlier run (rule validation)")
    ap.add_argument("--session", default=None,
                    help="restrict to a session, or a comma-separated list of "
                         "them. Needed to put a prompt on the same population as "
                         "exposed_probe.py, which trains and tests on whole "
                         "sessions. Without it --natural draws from all 11 "
                         "sessions, seven of which carry no positives at all, and "
                         "the resulting F1 is not comparable to the probe's.")
    ap.add_argument("--all", dest="use_all", action="store_true",
                    help="score EVERY instance of the selected sessions instead of "
                         "sampling n-pos/n-neg. The four annotated sessions hold "
                         "5,168 instances carrying 99.8%% of all exposed "
                         "positives and 100%% of all watched positives, so this is "
                         "the largest scorable set that exists. The other seven "
                         "sessions sit at 0-0.1%% exposed against 13.7-40.2%% here, "
                         "which is a 300x gap that reads as missing annotation "
                         "rather than real variation — including them would count "
                         "correct positives as false alarms.")
    ap.add_argument("--prefill", default=None,
                    help="text to prefill the assistant turn with, so the "
                         "completion IS the JSON. Without it a short custom "
                         "prompt loses to the model's own habits: the eyes "
                         "prompt got 786/800 valid JSON objects and not one of "
                         "them used the requested schema.")
    ap.add_argument("--prompt-file", default=None,
                    help="read the prompt text from a file, e.g. the OPRO-optimised "
                         "out/metav4_best_prompt.txt. Overrides --style.")
    ap.add_argument("--out", default=f"{OUTDIR}/momentary_targeted.json")
    a = ap.parse_args()

    custom_prompt = None
    if a.prompt_file:
        custom_prompt = open(a.prompt_file).read().strip()
        # The OPRO prompts were written for exp20, which shows K frames at once
        # and templates {K} in. This eval feeds exactly ONE image, so the
        # placeholder has to be filled or the model receives a literal "{K}".
        custom_prompt = custom_prompt.replace("{K}", "1")
        print(f"prompt from {a.prompt_file} ({len(custom_prompt)} chars, K=1); "
              f"--style {a.style} ignored", flush=True)

    rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
    def sess(p):
        m = re.search(r"(IPS_[^/]+)/", p)
        return m.group(1) if m else "?"

    def key(p):
        # 1-image / 2-image / 3-image all exist — match the generic prefix.
        i = p.find("lotte/")
        return p[i:] if i >= 0 else p

    # only the sessions where exposed actually happens; a negative drawn from a
    # session with no positives would make the task artificially easy
    T = a.target
    if a.session and "," in a.session:
        want = {x.strip() for x in a.session.split(",") if x.strip()}
        before = len(rows)
        rows = [r for r in rows if sess(r["image_path"]) in want]
        print(f"session filter {len(want)} sessions: {len(rows)} of {before} rows",
              flush=True)
    elif a.session:
        before = len(rows)
        rows = [r for r in rows if sess(r["image_path"]) == a.session]
        npos_s = sum(1 for r in rows if str(r[T]).strip().lower() == "true")
        print(f"session filter {a.session}: {len(rows)} of {before} rows, "
              f"{npos_s} {T}-positive ({100*npos_s/max(len(rows),1):.1f}%)",
              flush=True)
        if not rows:
            print("no rows for that session"); return 1
    if a.natural and not a.use_all:
        # --all means "every instance", which already preserves the natural
        # prevalence; without this guard --natural silently capped the run at
        # n_pos + n_neg and --all never took effect.
        rng0 = random.Random(a.seed)
        sample = rng0.sample(rows, min(a.n_pos + a.n_neg, len(rows)))
        npos = sum(1 for r in sample if str(r[T]).strip().lower() == "true")
        print(f"NATURAL sample: {len(sample)} rows from all sessions, "
              f"{npos} {T}-positive ({100*npos/len(sample):.1f}%)", flush=True)
    else:
        sample = None
    live = {s for s in {sess(r["image_path"]) for r in rows}
            if any(str(r[T]).strip().lower() == "true"
                   for r in rows if sess(r["image_path"]) == s)}
    pool = [r for r in rows if sess(r["image_path"]) in live]
    pos = [r for r in pool if str(r[T]).strip().lower() == "true"]
    neg = [r for r in pool if str(r[T]).strip().lower() != "true"]
    rng = random.Random(a.seed)
    rng.shuffle(pos); rng.shuffle(neg)
    # --skip makes the sample disjoint from a previous run with the same seed,
    # which is what a held-out check of a post-hoc rule requires.
    pos = pos[a.skip:]; neg = neg[a.skip:]
    if sample is None:
        if a.use_all:
            # Every instance of the selected sessions, in a fixed shuffled order
            # so a partial run is still a random subset rather than one session.
            sample = pos + neg
            rng.shuffle(sample)
        else:
            sample = pos[: a.n_pos] + neg[: a.n_neg]
            rng.shuffle(sample)
    print(f"sessions with events: {len(live)}  |  sampled "
          f"{min(a.n_pos,len(pos))} pos + {min(a.n_neg,len(neg))} neg = {len(sample)}",
          flush=True)

    gen = tv.TransformersChat(a.model, adapter=a.adapter)

    results = []
    for n, r in enumerate(sample, 1):
        frame = {"image": key(r["image_path"]),
                 "box": [float(r["x1"]), float(r["y1"]),
                         float(r["x2"]), float(r["y2"])]}
        try:
            if a.repr_ == "crop":
                im = tv.crop_pil(frame)
            elif a.repr_ == "full_box":
                im = full_box_image(frame)
            else:
                im = [full_box_image(frame), tv.crop_pil(frame)]
            if a.style == "subattr":
                pr, pf = tv.PROMPT, None
            elif a.style == "direct":
                pr, pf = DIRECT_PROMPT, '{"exposed": '
            else:
                pr, pf = STYLES[a.style]
            if custom_prompt is not None:
                pr, pf = custom_prompt, a.prefill
                if "{bboxes}" in pr:
                    # PADQ-style prompt: it names the detections by box rather
                    # than showing a crop, so the box has to be substituted per
                    # instance. Coordinates are 0-1000 of the ORIGINAL frame,
                    # which is the scale prompt_eng_eval.py fed it.
                    _p = os.path.join(tv.LOTTE_ROOT,
                                      frame["image"].replace("lotte/", "", 1))
                    if not os.path.exists(_p):
                        _p = os.path.join(tv.LOTTE_ROOT, frame["image"])
                    _w, _h = Image.open(_p).size
                    x1, y1, x2, y2 = frame["box"]
                    bb = [round(x1 * 1000 / _w), round(y1 * 1000 / _h),
                          round(x2 * 1000 / _w), round(y2 * 1000 / _h)]
                    pr = (pr.replace("{n}", "1")
                            .replace("{bboxes}", json.dumps([bb])))
            txt = gen.ask(im, pr, prefill=pf)
        except Exception as e:
            print(f"  fail {n}: {e}", flush=True)
            continue
        obj = tv.parse_json(txt)
        results.append({
            "image": frame["image"], "box": frame["box"],
            "gt_exposed": str(r["exposed"]).strip().lower() == "true",
            "gt_watched": str(r["watched"]).strip().lower() == "true",
            "subattr": obj, "raw": None if obj else txt,
        })
        if n % 50 == 0:
            print(f"  {n}/{len(sample)}", flush=True)

    json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)
    ok = [r for r in results if r["subattr"]]
    print(f"\n===== momentary targeted — {a.label} =====")
    print(f"parse-valid {len(ok)}/{len(results)}")

    from collections import Counter

    def report(name, pairs, extra=""):
        pairs = [(p_, g) for p_, g in pairs if p_ is not None]
        if not pairs:
            print(f"  {name}: no usable answers")
            return
        p, g = zip(*pairs)
        s_, pr, rc, tp, fp, fn = f1(p, g)
        print(f"  {name:16s} F1 {s_:.3f}  P {pr:.3f}  R {rc:.3f}   "
              f"tp{tp} fp{fp} fn{fn}  pred+ {sum(p)/len(p):.1%} vs gt {sum(g)/len(g):.1%}"
              f"{extra}")
        return s_

    if custom_prompt is not None:
        # metav4 emits {"gender":..,"age":..,"frames":[{"exposed":..,"watched":..}]}
        # so the pair lives one level down. Top level is accepted as a fallback
        # for prompts that answer straight out.
        pick = derive

        nested = sum(1 for r in ok if isinstance(r["subattr"].get("frames"), list))
        print(f"  metav4 schema: {nested}/{len(ok)} answers carry a frames[] array")
        # DERIVE RATE, not parse rate. A prompt whose schema the model ignores
        # still yields valid JSON: the PADQ runs read "parse-valid 799/800" while
        # only 536 answers carried an `exposed` field, because the model replied
        # with a free-form description instead of the requested list. Parse rate
        # cannot see that; this can.
        for _n in ("exposed", "watched"):
            _got = sum(1 for r in ok if pick(r["subattr"], _n) is not None)
            _pct = _got / len(ok) if ok else 0
            _flag = "   <-- SCHEMA IGNORED" if 0 < _pct < 0.9 else ""
            print(f"  derive {_n:8s}: {_got}/{len(ok)} answers usable "
                  f"({_pct:.0%}){_flag}")
        best = {}
        for name in ("exposed", "watched"):
            best[name] = report(name, [(pick(r["subattr"], name), r[f"gt_{name}"])
                                       for r in ok])
        print("\n  [this is the SAME session and the SAME natural prevalence the "
              "CLIP probe was scored on, so these F1s are comparable to it]")
        return

    if a.style in STYLES:
        for k in ("body_orientation", "face_visibility", "head_pitch",
                  "gaze_at_camera", "labels"):
            c = Counter(str(r["subattr"].get(k)) for r in ok if k in r["subattr"])
            if c:
                print(f"  {k:16s} {c.most_common()}")

        best = {}
        if a.style == "tier":
            for i, name in ((0, "exposed"), (1, "watched")):
                best[name] = report(name, [(tier_rule(r["subattr"])[i],
                                            r[f"gt_{name}"]) for r in ok])
        elif a.style == "trueonly":
            for name in ("exposed", "watched"):
                best[name] = report(name, [
                    (name in [str(x).strip().lower()
                              for x in (r["subattr"].get("labels") or [])],
                     r[f"gt_{name}"]) for r in ok])
        elif a.style == "graded":
            # A threshold picked to maximise F1 on the same data is fitted, not
            # measured, so it is reported ONLY as an upper bound. The headline
            # number is the prior-matched operating point: take the top-k scores
            # where k = the GT positive count, which is the decision any
            # deployment could actually make from the known base rate.
            for name in ("exposed", "watched"):
                sc = []
                for r in ok:
                    v = r["subattr"].get(f"{name}_score")
                    try:
                        sc.append((float(v), r[f"gt_{name}"]))
                    except (TypeError, ValueError):
                        pass
                if not sc:
                    print(f"  {name}: no scores parsed"); continue
                k = sum(1 for _, g in sc if g)
                cut = sorted((s for s, _ in sc), reverse=True)[:k][-1] if k else 101
                best[name] = report(f"{name}@prior",
                                    [(s >= cut, g) for s, g in sc],
                                    f"  (cut={cut:.0f})")
                oracle = max((f1([s >= t for s, _ in sc], [g for _, g in sc])[0], t)
                             for t in range(0, 101, 5))
                print(f"  {name+'@oracle':16s} F1 {oracle[0]:.3f} at cut={oracle[1]}"
                      f"   [FITTED on this data — upper bound, not a measurement]")
        else:  # prior
            def as_bool(v):
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return bool(v)
                if isinstance(v, str):
                    return v.strip().lower() in ("true", "yes", "1")
                return None
            for name in ("exposed", "watched"):
                best[name] = report(name, [(as_bool(r["subattr"].get(name)),
                                            r[f"gt_{name}"]) for r in ok])

        print("\n  [refs at natural prevalence: exposed — best of the 2x3 grid 0.178, "
              "SVFD-v10 0.463 | watched — full_mask+direct 0.390, SVFD-v10 0.571]")
        g6 = (best.get("exposed") or 0) >= 0.25
        print(f"  GATE G6 [{a.style}]: exposed F1 "
              f"{best.get('exposed') or 0:.3f} >= 0.25 -> {'PASS' if g6 else 'fail'}")
        return

    for k in ("orientation", "face_visible", "gaze"):
        print(f"  {k:13s} {Counter(r['subattr'].get(k) for r in ok).most_common()}")

    if a.style == "direct":
        def as_bool(v):
            # the model answers true/false, but also 1/0 and "true"/"yes" — a
            # bool-only check silently dropped 5 of every 12 answers
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("true", "yes", "1")
            return None

        for name in ("exposed", "watched"):
            pairs = [(as_bool(r["subattr"].get(name)), r[f"gt_{name}"]) for r in ok]
            pairs = [(p_, g) for p_, g in pairs if p_ is not None]
            if not pairs:
                print(f"  {name}: no boolean answers"); continue
            p, g = zip(*pairs)
            s_, pr, rc, tp, fp, fn = f1(p, g)
            print(f"  {name:8s} F1 {s_:.3f}  P {pr:.3f}  R {rc:.3f}   "
                  f"tp{tp} fp{fp} fn{fn}  (pos {sum(g)}/{len(g)})")
        return

    for i, name, ref in ((0, "exposed", "SVFD-v10 0.463 / meta-prompt 0.396 / plain 0.190"),
                         (1, "watched", "SVFD-v10 0.571 / trueonly 0.185")):
        pairs = [(tv.derive_momentary(r["subattr"])[i], (r["gt_exposed"], r["gt_watched"])[i])
                 for r in ok]
        pairs = [(p, g) for p, g in pairs if p is not None]
        if not pairs:
            print(f"  {name}: nothing derivable")
            continue
        p, g = zip(*pairs)
        s, pr, rc, tp, fp, fn = f1(p, g)
        print(f"  {name:8s} F1 {s:.3f}  P {pr:.3f}  R {rc:.3f}   "
              f"tp{tp} fp{fp} fn{fn}  (pos {sum(g)}/{len(g)})")
        print(f"           [ref: {ref}]")


if __name__ == "__main__":
    main()
