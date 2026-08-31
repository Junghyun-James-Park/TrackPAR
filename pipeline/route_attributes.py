#!/usr/bin/env python3
"""Stage 0 — decide how each attribute should be labelled, before anything runs.

An auto-labeller is handed an attribute and has to work out the rest. Everything
downstream follows from one question: is this a property of the PERSON or of the
FRAME?

    identity   holds for the whole track   -> track, then K frames in one call
    momentary  can differ frame to frame   -> no tracking, one call per frame
      facial       decided from face/gaze  -> exemplars from the eyes/svfd family
      non-facial   decided from body/hands -> exemplars from the PADQ template

`facial` is asked only for momentary attributes. For identity it makes no
difference: the route is the same whether the attribute sits on the face or the
trousers.

The prompt states the deployment window, and that is not decoration. Asked in
general, the same object routes two ways depending on what it is called:
`Accessory-Bag` came back identity ("a fixed accessory") while
`carrying_by_hand` came back momentary ("hand position and object held"). The
answer also moved with the batch — the accessories routed identity when listed
beside other appearance attributes and momentary when listed beside actions.
Naming the window (a fixed camera, a person crossing it in seconds) fixed eight
of twenty boundary cases and broke none. It then over-corrected: every one of
RAP v2's nine action attributes came back identity, because "when unsure, choose
identity" swallows verbs too.

The carve-out below fixes that, and it is there because the two mistakes do not
cost the same. A stable attribute called momentary costs extra calls and nothing
else — the per-frame answers agree and collapse back by majority vote. A
changing attribute called identity was never observed per frame, so if the value
did move there is one answer for the whole track, it is wrong for part of it,
and nothing records that. So states the person arrived with default to identity
and actions they are performing default to momentary.

Measured on three checks, with the anchors (exposed, watched momentary; gender,
age identity) holding throughout:

    boundary cases   original 12/20   window 17/20   window+verb  20/20
    RAP v2 actions   momentary 9/9    0/9            9/9
    UPAR 40 (all appearance)   identity 40/40 for all three

`window+verb` is the only version that passes all three.

Results are cached in out/attr_routing.json. An attribute already routed is not
sent to the model again, so re-running is free and the routing of a given
attribute does not drift between runs.

    python pipeline/route_attributes.py --attrs exposed watched gender age
    python pipeline/route_attributes.py --attrs-file my_attributes.json
    python pipeline/route_attributes.py --show          # print the cache, no GPU
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(ROOT, "out"))
CACHE = os.path.join(OUTDIR, "attr_routing.json")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPT = """You are the router of an automatic labelling system for CCTV video.

The setting matters for every answer below. The camera is FIXED and mounted
overhead. A person walks into view, crosses it and leaves, so one person's
track is a few seconds long — often only a handful of frames. Decide each
attribute for THAT SHORT WINDOW, not for the person's whole day.

Q1 "kind":
  - "identity"  : the value is the same in every frame of that person's track.
                  This includes anything that could change in principle but will
                  not change while someone walks past a camera: clothing and its
                  colour, hair, glasses, a hat, a bag or backpack they are
                  carrying. They arrived with it and they leave with it.
                  Several frames are then several views of ONE answer, and
                  seeing them together helps.
  - "momentary" : the value genuinely DIFFERS between frames of the SAME person
                  inside that short window. Ask yourself: if I looked at this
                  person in frame 1 and again in frame 5, could the honest
                  answer flip? Things that do flip: whether the face is visible,
                  which way the head is turned, where they are looking, whether
                  a shelf is in the way. Each frame then needs its own answer.

One case is decided the other way. If the attribute names an ACTION the person
is PERFORMING — carrying, holding, pushing, pulling, talking, calling, reaching,
bending — answer "momentary", even when you think the action probably lasts.
A verb describes a moment. Note that WEARING or HAVING something is a state, not
an action: a backpack on someone's back, a hat on their head or a bag on their
shoulder is identity, because they arrived with it.

The two mistakes do not cost the same, which is why the default splits that way.
Labelling a stable attribute "momentary" costs extra model calls and nothing
else: the per-frame answers agree, and one answer per track can be recovered
from them afterwards. Labelling a changing attribute "identity" cannot be undone
— the value was never observed per frame, so if it did change there is one
answer for the whole track, it is wrong for part of it, and nothing records that.

So: when unsure about a STATE the person arrived with, choose "identity". When
unsure about an ACTION they are performing, choose "momentary".

Q2 "facial" — answer only when kind is "momentary", otherwise "n/a". Is the
   attribute decided by looking at the person's FACE, HEAD ORIENTATION or GAZE?
  - "yes" : you determine it from the face, head pose, or where they are looking
  - "no"  : you determine it from the body, hands, posture, or objects nearby

Attributes:
{attrs}

Output exactly one JSON object and nothing else:
{{"<name>": {{"kind": "identity" or "momentary", "facial": "yes"/"no"/"n/a",
"why": "<short reason>"}}, ...}}"""


def load_cache():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    return {}


def save_cache(c):
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(c, open(CACHE, "w"), indent=1, ensure_ascii=False)


def show(cache):
    if not cache:
        print("nothing routed yet")
        return
    by = {}
    for a, v in cache.items():
        key = v["kind"] + (f" / {v['facial']}" if v["kind"] == "momentary" else "")
        by.setdefault(key, []).append(a)
    for key in sorted(by):
        print(f"\n{key}  ({len(by[key])})")
        for a in sorted(by[key]):
            print(f"    {a:34s} {cache[a].get('why','')[:44]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrs", nargs="*", default=[],
                    help="attribute names, optionally 'name: definition'")
    ap.add_argument("--attrs-file", default=None,
                    help="JSON list of names, or object of name -> definition")
    ap.add_argument("--model-id", default=os.environ.get("BASE_MODEL",
                                                         "Qwen/Qwen3.5-9B"))
    ap.add_argument("--refresh", action="store_true",
                    help="re-route attributes already in the cache")
    ap.add_argument("--show", action="store_true", help="print the cache and exit")
    a = ap.parse_args()

    cache = load_cache()
    if a.show:
        show(cache)
        return 0

    wanted = {}
    for item in a.attrs:
        if ":" in item:
            n, d = item.split(":", 1)
            wanted[n.strip()] = d.strip()
        else:
            wanted[item.strip()] = ""
    if a.attrs_file:
        obj = json.load(open(a.attrs_file))
        if isinstance(obj, list):
            wanted.update({x: "" for x in obj})
        else:
            wanted.update(obj)
    if not wanted:
        print("nothing to route; pass --attrs or --attrs-file")
        return 1

    todo = [n for n in wanted if a.refresh or n not in cache]
    cached = [n for n in wanted if n not in todo]
    if cached:
        print(f"{len(cached)} already routed, reusing: {', '.join(sorted(cached)[:6])}"
              + (" ..." if len(cached) > 6 else ""))
    if not todo:
        show({n: cache[n] for n in wanted})
        return 0
    print(f"routing {len(todo)} attribute(s)", flush=True)

    import torch
    import tvlm_pseudo_subattr as tv
    from transformers import AutoModelForImageTextToText, AutoProcessor

    proc = AutoProcessor.from_pretrained(a.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()

    listing = "\n".join(f"- {n}" + (f": {wanted[n]}" if wanted[n] else "")
                        for n in todo)
    msg = [{"role": "user", "content": [{"type": "text",
                                         "text": PROMPT.format(attrs=listing)}]}]
    s = proc.apply_chat_template(msg, tokenize=False,
                                 add_generation_prompt=True) + '{"'
    inp = proc(text=[s], return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=max(600, 90 * len(todo)),
                           do_sample=False)
    txt = '{"' + proc.decode(o[0][inp["input_ids"].shape[1]:],
                             skip_special_tokens=True)
    obj = tv.parse_json(txt)
    if not obj:
        print("could not parse the routing answer:\n" + txt[:600])
        return 1

    missing = []
    for n in todo:
        v = obj.get(n)
        if not isinstance(v, dict) or "kind" not in v:
            missing.append(n)
            continue
        kind = str(v["kind"]).strip().lower()
        if kind not in ("identity", "momentary"):
            missing.append(n)
            continue
        f = str(v.get("facial", "")).strip().lower()
        facial = None
        if kind == "momentary":
            facial = "facial" if f.startswith("y") else "non-facial"
        cache[n] = {"kind": kind, "facial": facial,
                    "why": str(v.get("why", "")).strip(),
                    "definition": wanted[n],
                    "routed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_cache(cache)

    if missing:
        # Better to name them than to let a silent default decide the route.
        print(f"\nNOT routed (no usable answer): {', '.join(missing)}")
        print("Re-run with a definition after the name, e.g. "
              "--attrs 'holding_item: the person is holding a product'")

    show({n: cache[n] for n in wanted if n in cache})
    print(f"\ncache: {CACHE}  ({len(cache)} attributes)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
