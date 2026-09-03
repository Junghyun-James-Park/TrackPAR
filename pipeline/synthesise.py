#!/usr/bin/env python3
"""Stage 2: turn observed sub-attributes into a label, using a rule.

Stage 1 asks the model what it can see and stores the answer. This module turns
that answer into a label for an attribute the model was never asked about. One
extraction serves every attribute derivable from it, so adding an attribute
costs a rule and no inference at all.

The rule is a small structured object rather than free text, and that is the
whole point. `validate` checks it against the vocabulary before it can run, so a
field that does not exist is caught in microseconds. The comparable failure in a
generated PROMPT is invisible: the model answers, the answer parses, and every
field the reader wants is absent.

    rule = {"combine": "and",
            "clauses": [{"field": "orientation", "op": "eq",
                         "values": ["front"]}]}
    apply_rule(rule, {"orientation": "front"})   -> True
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The closed vocabulary a rule may compose from. Every entry is a field the
# stage 1 prompt asks for, with the values it is allowed to answer.
VOCAB = {
    "gender":       ["male", "female"],
    "age_group":    ["young", "adult", "old"],
    "hair.length":  ["short", "long", "bald", "unknown"],
    "upper.length": ["short", "long"],
    "upper.color":  ["black", "white", "grey", "red", "green", "blue",
                     "yellow", "brown", "purple", "pink", "orange", "other"],
    "lower.length": ["short", "long"],
    "lower.color":  ["black", "white", "grey", "red", "green", "blue",
                     "yellow", "brown", "purple", "pink", "orange", "other"],
    "lower.type":   ["trousers", "skirt"],
    "accessories":  ["backpack", "bag", "glasses", "sunglasses", "hat"],
    "orientation":  ["front", "side", "back"],
    "face_visible": ["clear", "partial", "none"],
    "gaze":         ["up", "level", "down", "unknown"],
}
LIST_FIELDS = {"accessories"}
OPS = ("eq", "in", "contains")


def describe(vocab=None):
    """The vocabulary as the rule writer sees it."""
    v = vocab or VOCAB
    return "\n".join(f"  {k}: " + " | ".join(vals) for k, vals in v.items())


def get(obj, field):
    """Read a dotted field out of a sub-attribute dict."""
    cur = obj or {}
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def validate(rule, vocab=None):
    """Problems with a rule. An empty list means it may run.

    Every message names the field and the offending value, because the retry is
    given this text verbatim: a writer told "unknown field 'body_pose'" fixes
    itself, one told "invalid" does not.
    """
    v = vocab or VOCAB
    p = []
    if not isinstance(rule, dict):
        return ["not a JSON object"]
    cl = rule.get("clauses")
    if not isinstance(cl, list) or not cl:
        return ["no clauses"]
    if rule.get("combine") not in ("and", "or", None):
        p.append(f"combine must be 'and' or 'or', got {rule.get('combine')!r}")
    for c in cl:
        if not isinstance(c, dict):
            p.append("a clause is not an object")
            continue
        f = c.get("field")
        if f not in v:
            p.append(f"unknown field {f!r}")
            continue
        if c.get("op") not in OPS:
            p.append(f"{f}: op must be one of {OPS}, got {c.get('op')!r}")
        if c.get("op") == "contains" and f not in LIST_FIELDS:
            p.append(f"{f}: op 'contains' is only for {sorted(LIST_FIELDS)}")
        vs = c.get("values")
        if not isinstance(vs, list) or not vs:
            p.append(f"{f}: values must be a non-empty list")
            continue
        bad = [x for x in vs if x not in v[f]]
        if bad:
            p.append(f"{f}: {bad} are not values of {f}; allowed: {v[f]}")
    return p


def apply_rule(rule, subattr):
    """Label one sub-attribute dict. None when the rule cannot be read."""
    if not isinstance(rule, dict) or not rule.get("clauses"):
        return None
    out = []
    for c in rule["clauses"]:
        got = get(subattr, c["field"])
        vals = c["values"]
        if c["op"] == "contains":
            out.append(isinstance(got, list) and any(x in got for x in vals))
        else:
            out.append(got in vals)
    if not out:
        return None
    return all(out) if rule.get("combine", "and") == "and" else any(out)


def describe_rule(rule):
    """One line, for a log or a registry comment."""
    if not isinstance(rule, dict) or not rule.get("clauses"):
        return "(no rule)"
    parts = [f"{c.get('field')} {c.get('op')} {c.get('values')}"
             for c in rule["clauses"]]
    return f" {rule.get('combine', 'and')} ".join(parts)
