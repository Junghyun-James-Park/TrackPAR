#!/usr/bin/env python3
"""Write a labelling prompt for an attribute that does not have one yet.

Stage 0 routes an attribute. If it is momentary and `config/prompt_registry.json`
already names a prompt written for it, that prompt is reused. Otherwise this
writes a new one, using the existing prompts as exemplars.

Which exemplars depends on the routing:

    facial      the eyes / svfd / combined prompts — they ask what is visible on
                the face and let the caller apply the rule
    non-facial  the PADQ template — full scene, target named by bounding box,
                one attribute per call. exposed_v3 and watched_v2 differ only in
                the attribute name and its definition, so the template is what
                gets carried over

The generated prompt is written to prompts/generated/<attr>.txt and registered in
out/generated_prompts.json. It is never written into config/prompt_registry.json
automatically: that table is the human-verified one.

    python pipeline/make_prompt.py --attr holding_item \
        --definition "the person is holding a product in their hand"
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(ROOT, "out"))
REGISTRY = os.path.join(ROOT, "config", "prompt_registry.json")
GEN_DIR = os.path.join(ROOT, "prompts", "generated")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FACIAL_INSTR = """You are writing a labelling prompt for a vision-language model
that reads ONE cropped view of ONE person from an overhead retail CCTV camera.

Below are prompts that already work for other attributes of this kind. Study what
they have in common:

  * they ask the model to REPORT WHAT IT OBSERVES rather than to apply the
    definition itself, and the caller turns those observations into a label
  * they name the fields explicitly and fix the allowed values
  * they state how rare the attribute is, because the model over-predicts
    otherwise
  * they end with a single JSON object and nothing else

Write a prompt in the same style for this attribute:

  name       : {attr}
  definition : {definition}

Output ONLY the prompt text. No preamble, no explanation, no code fences."""

PADQ_INSTR = """You are writing a labelling prompt for a vision-language model
that reads a FULL CCTV FRAME with the people already detected and given as
bounding boxes.

Below are prompts that already work for other attributes of this kind. They are
the same template with the attribute swapped, so keep the structure exactly:

  * the scene is described first, then "{{n}} person(s) have been detected at
    these bounding boxes (0-1000 coordinate scale): {{bboxes}}"
  * the attribute is defined as a true/false question about EACH person
  * both the true and the false case are spelled out
  * ambiguity is resolved toward false
  * it returns a JSON list in the same order as the input boxes, and ends with an
    open bracket so the model continues the list

Keep the `{{n}}` and `{{bboxes}}` placeholders exactly as written — the runner
substitutes them.

Write a prompt in the same style for this attribute:

  name       : {attr}
  definition : {definition}

Output ONLY the prompt text. No preamble, no explanation, no code fences."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True)
    ap.add_argument("--definition", default="")
    ap.add_argument("--kind", default=None, choices=["facial", "non-facial"],
                    help="override the routing; normally taken from stage 0")
    ap.add_argument("--model-id", default=os.environ.get("BASE_MODEL",
                                                         "Qwen/Qwen3.5-9B"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    reg = json.load(open(REGISTRY))

    # Reuse beats generate: if the table already names a prompt for this
    # attribute, writing a new one would silently replace a measured choice.
    known = reg.get("momentary", {}).get(a.attr)
    if known and known.get("prompt"):
        print(f"{a.attr} already has a prompt in the registry: {known['prompt']}")
        print(f"  {known.get('measured', '')}")
        print("Nothing generated. Pass a different --attr, or edit the registry "
              "if you mean to replace it.")
        return 0

    kind = a.kind
    if kind is None:
        rp = os.path.join(OUTDIR, "attr_routing.json")
        if not os.path.exists(rp):
            print("no routing yet — run pipeline/route_attributes.py first, "
                  "or pass --kind")
            return 1
        r = json.load(open(rp)).get(a.attr)
        if not r:
            print(f"{a.attr} is not routed. Run:\n"
                  f"  python pipeline/route_attributes.py --attrs "
                  f"'{a.attr}: {a.definition or '<definition>'}'")
            return 1
        if r["kind"] != "momentary":
            print(f"{a.attr} routed as {r['kind']}, which the identity adapter "
                  f"handles. No prompt needed.")
            return 0
        kind = r["facial"]
        a.definition = a.definition or r.get("definition", "")

    ex_paths = reg["exemplars"][kind]
    exemplars = []
    for p in ex_paths:
        fp = os.path.join(ROOT, p)
        if os.path.exists(fp):
            exemplars.append(f"--- exemplar: {os.path.basename(p)} ---\n"
                             + open(fp).read().strip())
    if not exemplars:
        print(f"no exemplars found for {kind}")
        return 1

    instr = (FACIAL_INSTR if kind == "facial" else PADQ_INSTR).format(
        attr=a.attr, definition=a.definition or "(none given)")
    full = "\n\n".join(exemplars) + "\n\n" + instr

    print(f"{a.attr}: {kind}, {len(exemplars)} exemplar(s)", flush=True)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()
    msg = [{"role": "user", "content": [{"type": "text", "text": full}]}]
    s = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[s], return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=1200, do_sample=False)
    text = proc.decode(o[0][inp["input_ids"].shape[1]:],
                       skip_special_tokens=True).strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:]).rsplit("```", 1)[0].strip()

    # Checks the runner depends on. A prompt missing these produces answers that
    # parse as JSON and derive nothing, which the parse rate cannot see.
    problems = []
    if kind == "non-facial":
        for ph in ("{n}", "{bboxes}"):
            if ph not in text:
                problems.append(f"missing placeholder {ph}")
    if "{" not in text or "}" not in text:
        problems.append("no JSON shape in the output spec")

    os.makedirs(GEN_DIR, exist_ok=True)
    out = a.out or os.path.join(GEN_DIR, f"{a.attr}.txt")
    open(out, "w").write(text + "\n")

    print("\n" + "=" * 70)
    print(text[:1200] + ("\n..." if len(text) > 1200 else ""))
    print("=" * 70)
    print(f"\nwrote {out}  ({len(text)} chars)")
    if problems:
        print("\nCHECK BEFORE USING:")
        for p in problems:
            print(f"  - {p}")

    idx = os.path.join(OUTDIR, "generated_prompts.json")
    all_gen = json.load(open(idx)) if os.path.exists(idx) else {}
    all_gen[a.attr] = {"path": os.path.relpath(out, ROOT), "kind": kind,
                       "definition": a.definition, "exemplars": ex_paths,
                       "problems": problems,
                       "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(all_gen, open(idx, "w"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
