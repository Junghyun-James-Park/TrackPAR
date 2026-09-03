#!/usr/bin/env python3
"""Label ONE attribute on your own images or video, end to end.

This is the entry point for an attribute the pipeline has never seen. Give it a
name, a definition and some data; it routes the attribute, obtains a prompt,
runs inference and writes JSON and CSV.

    python pipeline/label_attribute.py \
        --attr holding_item \
        --definition "the person is holding a product in their hand" \
        --images /data/my_frames \
        --out out/holding_item

What happens, in order:

    0  route      identity or momentary? momentary -> facial or not?  (cached)
    1  rule or    already in config/prompt_registry.json -> reuse it
       prompt     identity and facial momentary -> write a rule over the
                  observation fields; non-facial momentary -> write a prompt
                  from the PADQ template. Both are validated, retried 3x, and
                  fall back to the definition on the fourth failure.
    2  infer      a rule is applied to the stored fields, no model call
                  a prompt runs one call per image (K=1, no tracking)
                  identity:  K frames of one subject in a single call
    3  write      <out>.json and <out>.csv

The output field is named after your attribute. You do not rename anything.

A note on how that is guaranteed. The prompt body is generated, but the last
paragraph of every generated prompt (the output contract) is written by this
script, not by the model. The contract fixes the field name, so a generated
prompt cannot answer under a name the parser is not looking for. This also keeps
generated prompts clear of the field names `eyes`, `nose`, `mouth` and `gaze`,
which the exposed/watched parser treats as a schema signal.

`exposed` and `watched` do NOT work here. Use run_all.sh for those.

Their measured prompts answer with observations — eyes, gaze — and the label is
derived from those in code by exp20_unified_infer.parse_unified. This script
reads answers by looking for a field named after the attribute, which such an
answer does not contain, so every row comes back "no-field" and the attribute is
empty. The two readers are not interchangeable and this script only has one of
them.

Input formats
-------------
--images DIR            every image in DIR; the whole image is the subject.
                        This is the right mode for a crop dataset.
--images DIR --boxes B  B is JSON, either
                          {"frame1.jpg": [[x1,y1,x2,y2], ...], ...}
                        or a list of
                          [{"image": "frame1.jpg", "box": [x1,y1,x2,y2],
                            "track_id": 7}, ...]
                        Each box becomes one subject. track_id is optional and
                        only used by identity attributes, to group frames.
--video FILE --fps N    extract frames with ffmpeg first, then as above.

Checking it works before spending GPU hours
-------------------------------------------
    python pipeline/label_attribute.py --self-test

runs the parser against a set of hand-written answers, including the ones that
used to fail silently. No GPU and no model download.
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(ROOT, "out"))
REGISTRY = os.path.join(ROOT, "config", "prompt_registry.json")
GEN_DIR = os.path.join(ROOT, "prompts", "generated")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Field names the exposed/watched parser reads as a schema signal. An attribute
# called one of these would be ambiguous, so it is refused rather than guessed.
RESERVED = {"eyes", "nose", "mouth", "gaze", "frames", "gender", "age",
            "bbox_2d", "exposed", "watched"}


# --------------------------------------------------------------- the contract
def output_contract(attr):
    """The last paragraph of every generated prompt.

    Written here rather than by the generator so the field name is known before
    the model runs. Everything downstream reads exactly this name.
    """
    return (
        f'Answer for the person shown. Return exactly one JSON object and '
        f'nothing else, in this form:\n'
        f'{{"{attr}": true}}   or   {{"{attr}": false}}\n'
        f'Use the field name "{attr}" exactly as written. Do not add other '
        f'fields, explanations or code fences. If you cannot tell, answer false.'
    )


def parse_answer(raw, attr):
    """Read one answer.

    Returns (value, status) where status is one of:
        "ok"          a boolean was found under `attr`
        "no-json"     the text did not parse
        "no-field"    it parsed but carried no `attr` field
        "bad-value"   the field was there but was not boolean-like

    The three failure modes are kept apart because they need different fixes,
    and because "parsed" alone is not evidence the answer is usable: one prompt
    here parsed 799 of 800 answers while only 536 carried the requested field.
    """
    import tvlm_pseudo_subattr as tv
    d = tv.parse_json(raw)
    if d is None:
        return None, "no-json"
    # A list answer is the PADQ shape: one entry per detected person. With one
    # subject per call the first entry is the answer.
    if isinstance(d, list):
        d = next((e for e in d if isinstance(e, dict)), None)
        if d is None:
            return None, "no-field"
    if not isinstance(d, dict):
        return None, "no-field"
    if attr not in d:
        # Tolerate a nested {"frames": [{...}]} wrapper, which the crop prompts
        # use, so a prompt copied from that family still works.
        fr = d.get("frames")
        if isinstance(fr, list) and fr and isinstance(fr[0], dict) and attr in fr[0]:
            d = fr[0]
        else:
            return None, "no-field"
    v = d[attr]
    if isinstance(v, bool):
        return v, "ok"
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True, "ok"
    if s in ("false", "no", "0", "n"):
        return False, "ok"
    return None, "bad-value"


# ------------------------------------------------------------------ stage 0
def route(attr, definition, model=None, proc=None, refresh=False):
    """Identity or momentary, and if momentary, facial or not. Cached."""
    import route_attributes as RT
    cache = RT.load_cache()
    if attr in cache and not refresh:
        r = cache[attr]
        print(f"  cached: {attr} -> {r['kind']}"
              + (f" / {r['facial']}" if r.get("facial") else ""))
        return r
    if model is None:
        raise RuntimeError("routing needs the model")
    import tvlm_pseudo_subattr as tv
    import torch
    listing = f"- {attr}" + (f": {definition}" if definition else "")
    msg = [{"role": "user",
            "content": [{"type": "text", "text": RT.PROMPT.format(attrs=listing)}]}]
    s = proc.apply_chat_template(msg, tokenize=False,
                                 add_generation_prompt=True) + '{"'
    inp = proc(text=[s], return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=400, do_sample=False)
    txt = '{"' + proc.decode(o[0][inp["input_ids"].shape[1]:],
                             skip_special_tokens=True)
    obj = tv.parse_json(txt) or {}
    v = obj.get(attr)
    if not isinstance(v, dict) or str(v.get("kind", "")).lower() not in (
            "identity", "momentary"):
        print(f"  could not route {attr}. Model said:\n    {txt[:300]}")
        print("  Give a clearer --definition, or force it with --kind.")
        return None
    kind = str(v["kind"]).strip().lower()
    facial = None
    if kind == "momentary":
        facial = ("facial" if str(v.get("facial", "")).strip().lower().startswith("y")
                  else "non-facial")
    r = {"kind": kind, "facial": facial, "why": str(v.get("why", "")).strip(),
         "definition": definition,
         "routed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache[attr] = r
    RT.save_cache(cache)
    print(f"  routed: {attr} -> {kind}" + (f" / {facial}" if facial else ""))
    print(f"          reason: {r['why'][:80]}")
    return r


def extract_prompt_body(out):
    """Pull the prompt out of whatever the generator returned."""
    m = re.search(r"<PROMPT>(.*?)</PROMPT>", out, re.S)
    if m:
        out = m.group(1)
    out = out.strip()
    if out.startswith("```"):
        out = "\n".join(out.splitlines()[1:]).rsplit("```", 1)[0]
    # A reasoning preamble, when one leaks through anyway.
    out = re.sub(r"\A\s*(thinking process|reasoning|analysis)\s*:.*?\n\s*\n",
                 "", out, flags=re.S | re.I)
    return out.strip()


def strip_output_spec(body, attr):
    """Remove an output spec the generator wrote for itself.

    The PADQ exemplars all end with one, so the generator reproduces it however
    firmly it is told not to. Two output specs in one prompt is a coin flip over
    which the model obeys, and only ours names a field the reader accepts.

    Cutting is conditional: if what remains no longer looks like a prompt, the
    body is kept whole. A confusing prompt beats an empty one.
    """
    pat = re.compile(r"^\s*(return|output|respond|answer)\b[^\n]*\b(json|\[|\{)",
                     re.I)
    lines = body.split("\n")
    cut = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if cut is None:
        return body
    head = "\n".join(lines[:cut]).rstrip()
    return head if not prompt_body_problems(head, attr) else body


def prompt_body_problems(body, attr):
    """Reasons this text is not a usable prompt. Empty string means it is."""
    if len(body) < 150:
        return f"too short ({len(body)} chars)"
    low = body.lower()
    if low.startswith(("thinking", "reasoning", "analysis", "1.", "okay", "sure")):
        return "starts like commentary rather than an instruction"
    if "exemplar" in low or "the request" in low:
        return "talks about the writing task instead of the labelling task"
    for f in ("eyes", "nose", "mouth", "gaze"):
        if f'"{f}"' in low:
            return f"uses the reserved field name {f!r}"
    return ""


def fallback_body(attr, definition, kind):
    """A plain prompt, used when generation fails.

    Deliberately unremarkable. It states the question, the rarity prior and the
    tie-break, which is the part every prompt measured here needed.
    """
    what = definition or f"the attribute '{attr}' applies to the person"
    scene = ("You are looking at a full CCTV frame with one person marked."
             if kind == "non-facial" else
             "You are looking at a cropped image of one person from an "
             "overhead CCTV camera.")
    return (f"{scene}\n\n"
            f"Decide whether this is true of that person right now, in this "
            f"image:\n\n    {what}\n\n"
            f"Judge only what you can actually see in this image. Do not infer "
            f"it from what would usually be the case, and do not carry an "
            f"answer over from a different moment. This is usually false: "
            f"answer true only when you can point to what makes it true. If "
            f"the person is too small, blurred or occluded to tell, answer "
            f"false.")


def identity_body(attr, definition):
    """The prompt for an identity attribute.

    Not generated. The exemplar sets in the registry are both momentary: the
    facial one is the eyes/svfd family and the non-facial one is the PADQ
    template, and the shipped identity attributes (gender, age) build their
    prompts in code rather than from a text file. There is nothing to imitate,
    so borrowing the momentary exemplars would produce a prompt that asks about
    a single moment for a property that is supposed to hold across the track.
    """
    what = definition or f"the attribute '{attr}' applies to the person"
    return ("The images you are given are {K} views of the SAME person, taken "
            "at different moments by an overhead CCTV camera.\n\n"
            "Together they describe one fixed fact about that person. Use all "
            "of them: a detail that is unclear in one view is often plain in "
            "another. Give ONE answer for the person, not one per image.\n\n"
            f"Decide whether this is true of them:\n\n    {what}\n\n"
            "Answer true only when at least one view shows it clearly. If no "
            "view settles it, answer false.")


# --------------------------------------------------------- stage 3-5, rules
def synthesis_branch(routing):
    """Which attributes are labelled by a rule rather than by a prompt.

    identity and facial momentary compose from the stage 1 observation fields;
    non-facial does not, because nothing in that vocabulary names an action. The
    split is measured, not assumed: on facing_camera and facing_away a rule beat
    a generated prompt, a one-line definition and a hand-written prompt, all six
    paired intervals separated. On the five action attributes it did not.
    """
    return routing["kind"] == "identity" or routing.get("facial") == "facial"


def get_rule(attr, definition, routing, prompt_model, vocab=None):
    """Write and validate a rule. Returns (rule, info); rule is None only when
    three tries all failed validation, which sends the attribute to a prompt."""
    import make_rule as MR
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    reg = json.load(open(REGISTRY))
    known = (reg.get("synthesis") or {}).get(attr)
    if known and known.get("rule"):
        print(f"  registry: rule already recorded for {attr}")
        print(f"            {known.get('measured', '')}")
        return known["rule"], {"tries": 0, "confidence": known.get("confidence"),
                               "n_distinct": None, "from_registry": True}
    gid = prompt_model or os.environ.get("PROMPT_MODEL", "Qwen/Qwen3.5-27B")
    print(f"  writing a rule with {gid}  (text only, no images)", flush=True)
    gproc = AutoProcessor.from_pretrained(gid)
    gmodel = AutoModelForImageTextToText.from_pretrained(
        gid, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()
    rule, info = MR.make_rule(attr, definition, gmodel, gproc, vocab=vocab)
    del gmodel, gproc
    torch.cuda.empty_cache()
    return rule, info


# ------------------------------------------------------------------ stage 1
def get_prompt(attr, definition, routing, model, proc, force_regen=False,
               prompt_model=None):
    """Reuse a measured prompt, or write one. Returns (text, source).

    Writing a prompt is done by `prompt_model`, which defaults to a larger model
    than the one that labels. Routing asks a two-way question and 9B answers it;
    a prompt is read on every one of thousands of calls afterwards, so the cost
    of a bigger model is paid once and its effect is not once. The generator is
    loaded only when there is something to generate, and freed straight after,
    so the labelling model does not have to share the cards with it.
    """
    reg = json.load(open(REGISTRY))
    known = reg.get("momentary", {}).get(attr)
    if known and known.get("prompt") and not force_regen:
        p = os.path.join(ROOT, known["prompt"])
        print(f"  registry: {known['prompt']}")
        print(f"            {known.get('measured', '')}")
        return open(p).read(), "registry"

    gen_path = os.path.join(GEN_DIR, f"{attr}.txt")
    if os.path.exists(gen_path) and not force_regen:
        print(f"  reusing generated prompt: prompts/generated/{attr}.txt")
        return open(gen_path).read(), "generated-cached"

    import make_prompt as MP
    import torch
    if routing["kind"] == "identity":
        print("  identity attribute: using the built-in multi-frame template.")
        print("  Both exemplar sets in the registry are momentary, so there is "
              "nothing to generate from without changing what the prompt asks.")
        text = identity_body(attr, definition) + "\n\n" + output_contract(attr)
        os.makedirs(GEN_DIR, exist_ok=True)
        open(os.path.join(GEN_DIR, f"{attr}.txt"), "w").write(text + "\n")
        return text, "identity-template"
    kind = routing.get("facial") or "non-facial"
    ex_paths = reg["exemplars"][kind]
    exemplars = []
    for p in ex_paths:
        fp = os.path.join(ROOT, p)
        if os.path.exists(fp):
            exemplars.append(f"--- exemplar: {os.path.basename(p)} ---\n"
                             + open(fp).read().strip())
    instr = (MP.FACIAL_INSTR if kind == "facial" else MP.PADQ_INSTR).format(
        attr=attr, definition=definition or "(none given)")
    # The generator writes the observation guidance. It does not write the
    # output spec: that is appended below, so the field name cannot drift.
    #
    # It answers inside delimiters because the first version of this asked for
    # "the prompt text only" and got the model's own reasoning instead, headed
    # "Thinking Process:". Trying to strip that afterwards removed the prompt
    # along with it, and the run still looked fine, because the output contract
    # alone is enough to produce parseable answers. A generated prompt can fail
    # completely without the parse rate noticing.
    instr += ("\n\nWrite the instructions only. Do NOT write the output format "
              "or the JSON schema — those are supplied separately. Do not use "
              "the words eyes, nose, mouth or gaze as field names.\n\n"
              "Put the finished prompt between <PROMPT> and </PROMPT>, with no "
              "reasoning, commentary or headings outside those tags.")
    full = "\n\n".join(exemplars) + "\n\n" + instr

    gen_id = prompt_model or os.environ.get("PROMPT_MODEL") or None
    gproc, gmodel, borrowed = proc, model, True
    if gen_id and gen_id != os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-9B"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"  loading the prompt writer: {gen_id}", flush=True)
        gproc = AutoProcessor.from_pretrained(gen_id)
        gmodel = AutoModelForImageTextToText.from_pretrained(
            gen_id, dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="auto").eval()
        borrowed = False

    print(f"  generating a prompt ({kind}, {len(exemplars)} exemplars)", flush=True)
    body = ""
    for attempt in (1, 2):
        msg = [{"role": "user", "content": [{"type": "text", "text": full}]}]
        s = gproc.apply_chat_template(msg, tokenize=False,
                                      add_generation_prompt=True,
                                      enable_thinking=False)
        inp = gproc(text=[s], return_tensors="pt").to(gmodel.device)
        with torch.no_grad():
            o = gmodel.generate(**inp, max_new_tokens=900, do_sample=False)
        out = gproc.decode(o[0][inp["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
        body = strip_output_spec(extract_prompt_body(out), attr)
        bad = prompt_body_problems(body, attr)
        if not bad:
            break
        print(f"  attempt {attempt}: {bad}")
        full += ("\n\nYour previous answer was not usable: " + bad +
                 " Return only the prompt, inside <PROMPT></PROMPT>.")

    if not borrowed:
        # Free it before the labelling model needs the cards.
        del gmodel, gproc
        torch.cuda.empty_cache()

    if prompt_body_problems(body, attr):
        print("  generation did not produce a usable prompt; falling back to a "
              "plain template built from the definition")
        body = fallback_body(attr, definition, kind)

    text = body + "\n\n" + output_contract(attr)
    os.makedirs(GEN_DIR, exist_ok=True)
    open(gen_path, "w").write(text + "\n")
    idx = os.path.join(OUTDIR, "generated_prompts.json")
    os.makedirs(OUTDIR, exist_ok=True)
    allg = json.load(open(idx)) if os.path.exists(idx) else {}
    allg[attr] = {"path": os.path.relpath(gen_path, ROOT), "kind": kind,
                  "definition": definition, "exemplars": ex_paths,
                  "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(allg, open(idx, "w"), indent=1, ensure_ascii=False)
    print(f"  wrote prompts/generated/{attr}.txt  ({len(text)} chars)")
    print("  NOT measured. No generated prompt in this repo has been scored "
          "against ground truth; read the numbers it produces as a first pass.")
    return text, "generated"


# ------------------------------------------------------------------- inputs
def collect_units(a):
    """One 'unit' is one call: an image, or a box within an image."""
    img_dir = a.images
    if a.video:
        img_dir = a.frames_dir or os.path.join(OUTDIR, "frames_" + a.attr)
        os.makedirs(img_dir, exist_ok=True)
        if not glob.glob(os.path.join(img_dir, "*.jpg")):
            print(f"  extracting frames from {os.path.basename(a.video)} "
                  f"at {a.fps} fps -> {img_dir}")
            cmd = ["ffmpeg", "-loglevel", "error", "-i", a.video,
                   "-vf", f"fps={a.fps}", os.path.join(img_dir, "f%06d.jpg")]
            if subprocess.call(cmd) != 0:
                print("  ffmpeg failed. Extract frames yourself and pass --images.")
                return []
        else:
            print(f"  reusing frames already in {img_dir}")

    files = sorted(f for f in glob.glob(os.path.join(img_dir, "**", "*"),
                                        recursive=True)
                   if f.lower().endswith(IMG_EXT))
    if not files:
        print(f"  no images under {img_dir}")
        return []

    units = []
    if a.boxes:
        b = json.load(open(a.boxes))
        if isinstance(b, dict):
            for rel, boxes in b.items():
                p = rel if os.path.isabs(rel) else os.path.join(img_dir, rel)
                for i, box in enumerate(boxes):
                    units.append({"image": p, "box": box,
                                  "track_id": f"{rel}#{i}"})
        else:
            for e in b:
                rel = e["image"]
                p = rel if os.path.isabs(rel) else os.path.join(img_dir, rel)
                units.append({"image": p, "box": e.get("box"),
                              "track_id": str(e.get("track_id",
                                                    os.path.basename(rel)))})
        missing = [u for u in units if not os.path.exists(u["image"])]
        if missing:
            print(f"  {len(missing)} box entries point at images that do not "
                  f"exist, e.g. {missing[0]['image']}")
            units = [u for u in units if os.path.exists(u["image"])]
    else:
        for p in files:
            tid = os.path.basename(p)
            if a.track_regex:
                m = re.search(a.track_regex, os.path.basename(p))
                if m:
                    tid = m.group(1) if m.groups() else m.group(0)
            units.append({"image": p, "box": None, "track_id": tid})
    return units


def load_subject(unit, pad=0.12):
    """The image the model sees: the box crop, or the whole image."""
    from PIL import Image
    im = Image.open(unit["image"]).convert("RGB")
    if not unit.get("box"):
        return im
    x1, y1, x2, y2 = [float(v) for v in unit["box"]]
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - w * pad), max(0, y1 - h * pad)
    x2, y2 = min(im.width, x2 + w * pad), min(im.height, y2 + h * pad)
    if x2 <= x1 or y2 <= y1:
        return im
    return im.crop((int(x1), int(y1), int(x2), int(y2)))


# ---------------------------------------------------------------- self-test
def self_test():
    """Exercise the parser without a GPU. Every case here has bitten once."""
    A = "holding_item"
    cases = [
        ('{"holding_item": true}', (True, "ok"), "the contract's own shape"),
        ('{"holding_item": false}', (False, "ok"), "negative"),
        ('{"holding_item": "yes"}', (True, "ok"), "string instead of bool"),
        ('```json\n{"holding_item": true}\n```', (True, "ok"), "code fence"),
        ('Sure! {"holding_item": true}', (True, "ok"), "chatty preamble"),
        ('{"frames": [{"holding_item": true}]}', (True, "ok"),
         "crop-family wrapper"),
        ('[{"bbox_2d": [1,2,3,4], "holding_item": true}]', (True, "ok"),
         "PADQ list shape"),
        ('{"exposed": true}', (None, "no-field"),
         "answered under the OLD hardcoded name -> caught, not silently dropped"),
        ('{"holding_item": "maybe"}', (None, "bad-value"), "unusable value"),
        ('not json at all', (None, "no-json"), "parse failure"),
        ('{"holding_item": true, "eyes": "none"}', (True, "ok"),
         "eyes present no longer hijacks the branch"),
    ]
    print("parser self-test")
    print("=" * 74)
    ok = 0
    for raw, want, why in cases:
        got = parse_answer(raw, A)
        good = got == want
        ok += good
        print(f"  [{'ok  ' if good else 'FAIL'}] {why}")
        if not good:
            print(f"         input {raw!r}\n         want {want}, got {got}")
    print("=" * 74)
    print(f"{ok}/{len(cases)} passed")
    if ok == len(cases):
        print("\nThe contract in output_contract() and the reader in "
              "parse_answer() agree.")
    return 0 if ok == len(cases) else 1


# ---------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser(
        description="Label one attribute on your own images or video.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", help="attribute name, e.g. holding_item")
    ap.add_argument("--definition", default="",
                    help="one sentence. Routing and prompt quality both depend "
                         "on this; it is worth writing carefully")
    ap.add_argument("--images", help="directory of images")
    ap.add_argument("--video", help="video file; frames are extracted first")
    ap.add_argument("--fps", type=float, default=2.0, help="frames per second")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--boxes", default=None, help="JSON of person boxes")
    ap.add_argument("--track-regex", default=None,
                    help="regex over the filename; group 1 is the subject id. "
                         "Identity attributes group frames by it")
    ap.add_argument("--kind", choices=["identity", "momentary"],
                    help="skip stage 0 and force the route")
    ap.add_argument("--facial", choices=["facial", "non-facial"],
                    help="force the momentary sub-branch")
    ap.add_argument("--k", type=int, default=4,
                    help="frames per call for identity attributes")
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N units (0 = all). Applied after a "
                         "stable sort, so the same N are chosen every run")
    ap.add_argument("--model-id", default=os.environ.get("BASE_MODEL",
                                                         "Qwen/Qwen3.5-9B"),
                    help="the model that labels, and that stage 0 routes with")
    ap.add_argument("--prompt-model",
                    default=os.environ.get("PROMPT_MODEL", "Qwen/Qwen3.5-27B"),
                    help="the model that WRITES a prompt when one has to be "
                         "generated. Larger by default: it runs a handful of "
                         "times, and what it writes is read on every call after. "
                         "Set it to --model-id to use one model for everything")
    ap.add_argument("--adapter", default=None,
                    help="identity LoRA. The shipped adapter was trained for "
                         "gender and age only; leave unset for a new attribute")
    ap.add_argument("--regenerate", action="store_true",
                    help="write a new prompt even if one is cached")
    ap.add_argument("--force-prompt", action="store_true",
                    help="skip the rule path and write a prompt, even for a "
                         "branch that composes. For comparing the two")
    ap.add_argument("--out", default=None, help="path stem for .json and .csv")
    ap.add_argument("--self-test", action="store_true",
                    help="check the parser, no GPU")
    a = ap.parse_args()

    if a.self_test:
        return self_test()
    if not a.attr:
        ap.error("--attr is required (or use --self-test)")
    if not a.images and not a.video:
        ap.error("give --images DIR or --video FILE")

    if a.attr in RESERVED and a.attr not in ("exposed", "watched"):
        print(f"'{a.attr}' is a field name the exposed/watched parser reads as a "
              f"schema signal. Rename the attribute, e.g. '{a.attr}_visible'.")
        return 1

    stem = a.out or os.path.join(OUTDIR, a.attr)
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)

    print(f"\n=== {a.attr} ===")
    if a.definition:
        print(f"  definition: {a.definition}")
    else:
        print("  no --definition given. Routing and prompt generation both read "
              "it; without one they work from the name alone.")

    print("\n[0/3] routing")
    routing = None
    if a.kind:
        routing = {"kind": a.kind,
                   "facial": a.facial or ("non-facial" if a.kind == "momentary"
                                          else None),
                   "why": "forced on the command line", "definition": a.definition}
        print(f"  forced: {a.kind}" + (f" / {routing['facial']}"
                                       if routing["facial"] else ""))
    else:
        import route_attributes as RT
        cached = RT.load_cache().get(a.attr)
        if cached:
            routing = route(a.attr, a.definition)

    print("\n[1/3] input")
    units = collect_units(a)
    if not units:
        return 1
    n_all = len(units)
    units.sort(key=lambda u: (u["image"], str(u.get("box"))))
    if a.limit and a.limit < len(units):
        units = units[:a.limit]
        print(f"  {n_all} units found, --limit {a.limit} applied -> {len(units)}")
        print("  This is a subset. Do not report a rate from it as if it were "
              "the whole corpus.")
    else:
        print(f"  {len(units)} units"
              + (" (one per box)" if a.boxes else " (one per image)"))

    # The model is needed for routing (if uncached), generation and inference,
    # so it is loaded once, here, and reused for all three.
    print("\n  loading the model", flush=True)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model_id)
    # Pinned to the first visible card rather than device_map="auto". Both cards
    # are exposed so the larger writer can use the pair, but a single-stream
    # labelling model spread over two of them leaves each about a third busy.
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()
    if a.adapter:
        # Through attach_adapter, never PeftModel.from_pretrained directly.
        # It strips the key prefixes the trainer wrote, loads the non-LoRA
        # tensors BEFORE the LoRA wrapper renames the module tree, and raises if
        # none of them land. Doing it by hand here reintroduced exactly the bug
        # the rest of the repo documents: loaded after wrapping, with strict off,
        # every tensor is dropped and nothing says so.
        from rap2_eval import attach_adapter
        model = attach_adapter(model, a.adapter)
        print(f"  adapter attached: {a.adapter}")

    if routing is None:
        routing = route(a.attr, a.definition, model, proc)
        if routing is None:
            return 1

    # ---- stage 3-5: a rule when the branch composes, a prompt otherwise ----
    rule = rule_info = None
    if synthesis_branch(routing) and not a.force_prompt:
        print("\n[2/3] rule")
        rule, rule_info = get_rule(a.attr, a.definition, routing, a.prompt_model)
        if rule is None:
            print(f"  no valid rule after {rule_info['tries']} tries — "
                  f"falling back to a prompt")
        else:
            import synthesise as SY
            print(f"  {a.attr} := {SY.describe_rule(rule)}")
            print(f"  confidence {rule_info['confidence']}"
                  + (f"  ({rule_info['n_distinct']} distinct rules over "
                     f"{rule_info['n_samples']} samples)"
                     if rule_info.get('n_distinct') else ""))
            if rule_info["confidence"] == "low":
                print("  LOW CONFIDENCE — the writer gave a different rule each "
                      "time it was asked.")
                print("  The labels still ship, flagged, so a reviewer can start "
                      "there.")

    prompt = src = None
    if rule is None:
        print("\n[2/3] prompt")
        prompt, src = get_prompt(a.attr, a.definition, routing, model, proc,
                                 a.regenerate, a.prompt_model)
    registry_path = src == "registry"
    if registry_path and a.attr in ("exposed", "watched"):
        # Loud, because the failure downstream is quiet: the prompt loads, the
        # model answers, the answers parse, and every one of them is unusable.
        print("\n  WARNING: this prompt reports observations (eyes, gaze) and "
              "leaves the")
        print("  label to be derived in code. This runner reads answers by "
              "looking for a")
        print(f"  field called '{a.attr}', which is not in them, so every row "
              f"will come back")
        print("  unusable. Use run_all.sh for exposed and watched.\n")

    print(f"\n[3/3] inference — {routing['kind']}", flush=True)
    import exp20_unified_infer as X

    groups = {}
    if routing["kind"] == "identity":
        for u in units:
            groups.setdefault(u["track_id"], []).append(u)
        print(f"  {len(groups)} subjects, up to K={a.k} frames per call")
        if len(groups) == len(units):
            if a.track_regex:
                print("  --track-regex matched a different id in every "
                      "filename, so each image is its own subject. Check the "
                      "capture group if you expected frames to group.")
            elif a.boxes:
                print("  No track_id repeated in the boxes file, so each box "
                      "is its own subject.")
            else:
                print("  Every image became its own subject. If your filenames "
                      "carry a subject id, pass --track-regex to group them; "
                      "otherwise this is one call per image.")
        jobs = [(tid, us[:a.k]) for tid, us in groups.items()]
    else:
        print(f"  {len(units)} calls, K=1, no tracking")
        jobs = [(u["track_id"], [u]) for u in units]

    field = a.attr
    base_dir = os.path.abspath(a.images) if a.images else (
        os.path.abspath(a.frames_dir) if a.frames_dir else None)
    if base_dir and not all(os.path.abspath(u["image"]).startswith(base_dir)
                            for u in units):
        base_dir = None       # boxes pointed outside the directory

    rows, stat, t0 = [], {}, time.time()
    for i, (tid, us) in enumerate(jobs, 1):
        imgs = [load_subject(u) for u in us]

        if rule is not None:
            # Stage 2 first, then the rule. The extraction is the only thing the
            # model is asked, and it is asked once whatever the attribute is —
            # which is why a second attribute on the same data costs a rule and
            # nothing else. Not cached here yet; run_all reuses a stored
            # extraction, and this path re-reads the image per attribute.
            import synthesise as SY
            import tvlm_pseudo_subattr as tv
            raw, _, _ = X.gen_with_tokens(model, proc, imgs, tv.PROMPT,
                                          model.device, 320)
            obs = tv.parse_json(raw)
            if not isinstance(obs, dict):
                val, st = None, "no-json"
            else:
                val = SY.apply_rule(rule, obs)
                st = "ok" if isinstance(val, bool) else "no-field"
            stat[st] = stat.get(st, 0) + 1
            rows.append({"subject": tid,
                         "image": os.path.relpath(us[0]["image"], base_dir)
                         if base_dir else us[0]["image"],
                         "n_frames": len(us),
                         a.attr: val,
                         "status": st,
                         "confidence": rule_info["confidence"],
                         "subattr": obs,
                         "raw": raw[:200] if st != "ok" else ""})
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el/i:.1f}s each  "
                      f"eta {(len(jobs)-i)*el/i/60:.0f} min", flush=True)
            continue

        body = prompt.replace("{K}", str(len(imgs)))
        if "{n}" in body:
            body = body.replace("{n}", str(len(imgs)))
        if "{bboxes}" in body:
            body = body.replace("{bboxes}", json.dumps(
                [u.get("box") or [0, 0, 1000, 1000] for u in us]))
        raw, _, _ = X.gen_with_tokens(model, proc, imgs, body, model.device, 256)
        val, st = parse_answer(raw, field)
        stat[st] = stat.get(st, 0) + 1
        rows.append({"subject": tid,
                     # Relative to the input directory, not the cwd: relpath
                     # from a scratch directory produced ../../../.. chains that
                     # nothing downstream could open.
                     "image": os.path.relpath(us[0]["image"], base_dir)
                     if base_dir else us[0]["image"],
                     "n_frames": len(us),
                     a.attr: val,
                     "status": st,
                     "raw": raw[:200] if st != "ok" else ""})
        if i % 25 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"  {i}/{len(jobs)}  {el/i:.1f}s each  "
                  f"eta {(len(jobs)-i)*el/i/60:.0f} min", flush=True)

    n = len(rows)
    usable = stat.get("ok", 0)
    print(f"\n  parsed as JSON : {n - stat.get('no-json', 0)}/{n}")
    print(f"  usable answers : {usable}/{n}   <-- the number that matters")
    for k in ("no-json", "no-field", "bad-value"):
        if stat.get(k):
            print(f"    {k:10s} {stat[k]}")
    if usable and usable < n * 0.9:
        print("\n  Under 90% usable. The prompt is answering in a shape the "
              "reader does not accept. Look at the `raw` column, then re-run "
              "with --regenerate.")
    if usable == 0:
        print("\n  Nothing usable. Check the `raw` column in the output before "
              "spending more GPU time.")
    if usable:
        pos = sum(1 for r in rows if r[a.attr] is True)
        print(f"\n  predicted positive: {pos}/{usable} ({100*pos/usable:.1f}%)")
        print("  Compare that against the rate you expect. A prompt that is far "
              "off is usually mis-thresholded rather than blind — see "
              "docs/RESULTS.md.")

    with open(stem + ".json", "w") as f:
        json.dump({"attribute": a.attr, "definition": a.definition,
                   "routing": routing,
                   "method": "synthesis" if rule is not None else "prompt",
                   "rule": rule,
                   "confidence": (rule_info or {}).get("confidence"),
                   "prompt_source": src,
                   "field": field, "units": n, "usable": usable,
                   "counts": stat, "results": rows}, f, indent=1,
                  ensure_ascii=False)
    cols = ["subject", "image", "n_frames", a.attr, "status"]
    if rule is not None:
        cols.append("confidence")
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {stem}.json and {stem}.csv")
    if rule is not None:
        import synthesise as SY
        print("  Labels were derived from a rule over the observation fields, "
              "not from a prompt")
        print(f"  written for this attribute: {SY.describe_rule(rule)}")
        if rule_info["confidence"] == "low":
            print("  Confidence is low — start a review here.")
    elif src and src.startswith("generated"):
        print("  The prompt was generated and has not been scored against "
              "ground truth. Treat these labels as a first pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
