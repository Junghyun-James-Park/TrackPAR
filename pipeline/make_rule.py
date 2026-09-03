#!/usr/bin/env python3
"""Stage 2a: write the rule that turns sub-attributes into one attribute.

A text-only call. The model never sees an image; it decides which observed
values mean the attribute is true. Cost is seconds per attribute against hours
for stage 1, so the retry loop below is nearly free.

Two decisions in the prompt are load-bearing.

The vocabulary is quoted in full and declared closed. That is what makes the
answer checkable: `synthesise.validate` rejects a field or a value that is not
in it, and says which, and the retry is given that sentence. A generated PROMPT
has no equivalent — its failure mode is an answer that parses and carries
nothing the reader wants, which no validator sees.

There is no way to decline. An earlier version let the model answer "not
expressible"; it then used that on attributes the vocabulary did cover — 9B
refused Hair-Length-Long while `hair.length: long` sat in the list it had been
given. An auto-labeller has to return a label, so a rough rule beats no rule and
the prompt says so.

    python pipeline/make_rule.py --attr facing_camera \
        --definition "the person's head and body are turned toward the camera"
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesise as S  # noqa: E402

MAX_TRIES = 3
N_SAMPLES = 3

INSTR = """You are configuring an automatic labelling system for CCTV images.

A vision model has already looked at each image and reported these observation
fields. This is the ENTIRE vocabulary available to you:

{vocab}

Your job is to express one target attribute as a rule over those fields. You do
not see any image. You only decide which observed values mean the attribute is
true.

Target attribute
  name       : {attr}
  definition : {definition}

You MUST produce a rule. The vocabulary may not name your attribute directly; in
that case choose the fields and values that come CLOSEST to it. A rough rule is
useful, no rule is not. Do not refuse, do not explain, do not say the attribute
cannot be expressed.

Answer with exactly one JSON object and nothing else:

{{"combine": "and",
  "clauses": [{{"field": "<a field name from the list>",
               "op": "eq" | "in" | "contains",
               "values": ["<values that field can take>"]}}]}}

  op "eq"       : the field equals one of the listed values
  op "in"       : same as eq, for a list of several values
  op "contains" : the field is a LIST and contains one of the listed values
                  (only `accessories` is a list)
  combine       : "and" if every clause must hold, "or" if any one suffices

Every field name and every value must be copied exactly from the list above."""


def _ask(model, proc, text, sample=False, seed=None):
    import torch
    import tvlm_pseudo_subattr as tv
    msg = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    s = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False) + '{"'
    inp = proc(text=[s], return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=320)
    if sample:
        if seed is not None:
            torch.manual_seed(seed)
        kw.update(do_sample=True, temperature=0.8, top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        o = model.generate(**inp, **kw)
    raw = '{"' + proc.decode(o[0][inp["input_ids"].shape[1]:],
                             skip_special_tokens=True)
    return tv.parse_json(raw), raw


def make_rule(attr, definition, model, proc, vocab=None):
    """Return (rule, info). `rule` is None only if every attempt failed
    validation, which the caller must handle by falling back to a prompt."""
    v = vocab or S.VOCAB
    base = INSTR.format(vocab=S.describe(v), attr=attr, definition=definition)
    text, tries, history = base, 0, []
    rule = None
    while tries < MAX_TRIES:
        tries += 1
        rule, raw = _ask(model, proc, text)
        probs = S.validate(rule, v) if rule else ["the answer was not JSON"]
        history.append({"try": tries, "problems": probs,
                        "rule": rule, "raw": raw[:300]})
        if not probs:
            break
        # The retry is told exactly what was wrong. This is the whole reason a
        # rule is easier to generate than a prompt.
        text = (base + "\n\nYour previous answer was rejected: "
                + "; ".join(probs)
                + "\nEvery field name and every value must be copied exactly "
                  "from the vocabulary above. Answer again.")
        rule = None

    if rule is None:
        return None, {"tries": tries, "history": history,
                      "confidence": None, "n_distinct": None}

    # The greedy rule is the one that ships. Sampling a few more is not a vote —
    # a majority over answers the model is confident about only re-elects its own
    # bias. It is a cheap confidence signal, and it earns that: on the four
    # attributes probed at three temperatures, the two whose rules were wrong
    # (holding, lb-Jeans) produced 4-5 distinct rules out of 5 while the two that
    # were right produced 1-2. Disagreement tracked the error; a majority over
    # the same samples would have discarded that and picked one at random.
    cands = [json.dumps(rule, sort_keys=True)]
    for i in range(N_SAMPLES - 1):
        r2, _ = _ask(model, proc, base, sample=True, seed=1234 + i)
        if r2 and not S.validate(r2, v):
            cands.append(json.dumps(r2, sort_keys=True))
    n_distinct = len(set(cands))
    conf = "high" if n_distinct == 1 else "low"
    return rule, {"tries": tries, "history": history,
                  "confidence": conf, "n_distinct": n_distinct,
                  "n_samples": len(cands)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True)
    ap.add_argument("--definition", default="")
    ap.add_argument("--model-id",
                    default=os.environ.get("PROMPT_MODEL", "Qwen/Qwen3.5-27B"))
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(a.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_id, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()
    rule, info = make_rule(a.attr, a.definition, model, proc)
    if rule is None:
        print(f"no valid rule after {info['tries']} tries")
        for h in info["history"]:
            print(f"  try {h['try']}: {h['problems']}")
        return 1
    print(f"{a.attr} := {S.describe_rule(rule)}")
    print(f"  tries {info['tries']}, confidence {info['confidence']} "
          f"({info['n_distinct']} distinct over {info['n_samples']} samples)")
    print(json.dumps(rule, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
