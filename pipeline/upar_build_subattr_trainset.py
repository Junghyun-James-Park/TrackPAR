#!/usr/bin/env python3
"""
Phase 0-1 — UPAR-40 warm-up SFT builder.

Converts the UPAR unified annotations (path + 40 binary attributes) into VLM SFT
samples whose TARGET is a nested sub-attribute JSON.

    python upar_build_subattr_trainset.py                      # full train split
    python upar_build_subattr_trainset.py --balanced 25000     # + balanced subset
    python upar_build_subattr_trainset.py --split val          # UPAR-side eval set
    python upar_build_subattr_trainset.py --limit 20 --dump    # smoke test

Schema decisions (locked with the user, 2026-08-03):
  * Exclusive groups  -> single value : age_group, hair.length,
                         upper.{length,color}, lower.{length,color,type}
  * True multi-label  -> list         : accessories
  * unknown allowed   : hair all-zero (9.7% of rows) -> "unknown";
                        colour all-zero -> "other" (a safety net; never fires
                        on the current CSV, where every row has >=1 colour)
  * multi-colour rows (1.2%) -> keep the PRIMARY colour = first positive in the
                        fixed CSV column order. Accepted ~1.2% information loss.
  * mA evaluation is NOT blocked by the nested schema: `subattr_to_40binary()`
    below maps a nested prediction deterministically back to the 40 binaries.

Input  = single crop, no box, GENERIC person-crop prompt (NOT overhead — the
         overhead framing belongs to the in-domain Stage-2, not the warm-up).
Output = <anno_dir>/upar_subattr_{split}.json  (repo SFT format: id/image/conversations)
         `image` stays RELATIVE to UPAR_ROOT; pass that as --image_folder.
"""

import argparse
import csv
import json
import os
import random
from collections import Counter

UPAR_ROOT = os.environ.get("UPAR_ROOT", "/mnt/sdb1/pjh/par_datasets/upar_data")
CSV_PATH = {
    "train": f"{UPAR_ROOT}/annotations/phase1/train/train.csv",
    "val": f"{UPAR_ROOT}/annotations/phase1/val_task1/val.csv",
}
OUT_DIR = f"{UPAR_ROOT}/sft"

# ---------------------------------------------------------------- vocabulary

COLORS = ["black", "blue", "brown", "green", "grey", "orange",
          "pink", "purple", "red", "white", "yellow", "other"]
# csv column suffix <- our lowercase value
COLOR_COL = {c: c.capitalize() for c in COLORS}
COLOR_COL["grey"] = "Grey"

ACCESSORY_COL = {
    "backpack": "Accessory-Backpack",
    "bag": "Accessory-Bag",
    "glasses": "Accessory-Glasses-Normal",
    "sunglasses": "Accessory-Glasses-Sun",
    "hat": "Accessory-Hat",
}

PROMPT = """<image>
This is a cropped photograph of a single person. Describe the person's visible
attributes, using ONLY the allowed values listed below.

- gender: "male" | "female"
- age_group: "young" | "adult" | "old"
- hair.length: "short" | "long" | "bald" | "unknown"   (unknown if not visible)
- upper.length: "short" | "long"                       (sleeve length)
- upper.color: one of [{colors}]
- lower.length: "short" | "long"
- lower.color: one of [{colors}]
- lower.type: "trousers" | "skirt"                     (trousers covers shorts; skirt covers dresses)
- accessories: a list containing any of ["backpack", "bag", "glasses", "sunglasses", "hat"]; [] if none

If a garment shows more than one colour, report the dominant one.
Output exactly one line of JSON and nothing else, no explanation:
{{"gender": ..., "age_group": ..., "hair": {{"length": ...}}, "upper": {{"length": ..., "color": ...}}, "lower": {{"length": ..., "color": ..., "type": ...}}, "accessories": [...]}}""".format(
    colors=", ".join(f'"{c}"' for c in COLORS)
)


def load_csv(path):
    """UPAR quirk: the header names the 40 attributes but NOT the leading image-path
    column, so data rows have 41 fields. Header i maps to data field i+1."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lstrip("# ").strip() for h in rows[0]]
    assert len(header) == 40, f"expected 40 attribute columns, got {len(header)}"
    body = [r for r in rows[1:] if r]
    for r in body:
        assert len(r) == len(header) + 1, f"row width {len(r)} != 41"
    return header, body


def _first_positive(row, idx, cols):
    """Exclusive group -> the first positive in fixed column order, else None."""
    for c in cols:
        if row[idx[c]] == "1":
            return c
    return None


def row_to_subattr(row, header, idx):
    """40 binaries -> nested sub-attribute dict."""
    age = _first_positive(row, idx, ["Age-Young", "Age-Adult", "Age-Old"])
    age_group = age.split("-")[1].lower() if age else "adult"  # 9/97669 rows are all-zero

    # Hair is NOT truly exclusive: 57.5% of Bald rows also carry Short (bald implies
    # short). Column order would collapse those 816 rows to "short" and cut the bald
    # training signal from 1,422 -> 604. Resolve toward the MORE SPECIFIC value.
    hair = _first_positive(row, idx,
                           ["Hair-Length-Bald", "Hair-Length-Long", "Hair-Length-Short"])
    hair_len = hair.rsplit("-", 1)[1].lower() if hair else "unknown"

    def colour(part):
        col = _first_positive(row, idx, [f"{part}-Color-{COLOR_COL[c]}" for c in COLORS])
        return col.rsplit("-", 1)[1].lower() if col else "other"

    lt = _first_positive(row, idx,
                         ["LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress"])
    lower_type = {"LowerBody-Type-Trousers&Shorts": "trousers",
                  "LowerBody-Type-Skirt&Dress": "skirt"}.get(lt, "trousers")

    return {
        "gender": "female" if row[idx["Gender-Female"]] == "1" else "male",
        "age_group": age_group,
        "hair": {"length": hair_len},
        "upper": {
            "length": "short" if row[idx["UpperBody-Length-Short"]] == "1" else "long",
            "color": colour("UpperBody"),
        },
        "lower": {
            "length": "short" if row[idx["LowerBody-Length-Short"]] == "1" else "long",
            "color": colour("LowerBody"),
            "type": lower_type,
        },
        "accessories": [name for name, col in ACCESSORY_COL.items()
                        if row[idx[col]] == "1"],
    }


def subattr_to_40binary(obj, header):
    """Deterministic reverse map, nested prediction -> 40 binaries, for mA scoring.

    Tolerant of malformed model output: anything unrecognised simply leaves the
    corresponding attributes at 0.
    """
    out = {h: 0 for h in header}

    if obj.get("gender") == "female":
        out["Gender-Female"] = 1

    ag = obj.get("age_group")
    if ag in ("young", "adult", "old"):
        out[f"Age-{ag.capitalize()}"] = 1

    hl = (obj.get("hair") or {}).get("length")
    if hl in ("short", "long", "bald"):
        out[f"Hair-Length-{hl.capitalize()}"] = 1

    for part, key in (("UpperBody", "upper"), ("LowerBody", "lower")):
        sub = obj.get(key) or {}
        if sub.get("length") == "short":
            out[f"{part}-Length-Short"] = 1
        c = sub.get("color")
        if c in COLOR_COL:
            out[f"{part}-Color-{COLOR_COL[c]}"] = 1

    t = (obj.get("lower") or {}).get("type")
    if t == "trousers":
        out["LowerBody-Type-Trousers&Shorts"] = 1
    elif t == "skirt":
        out["LowerBody-Type-Skirt&Dress"] = 1

    for a in obj.get("accessories") or []:
        col = ACCESSORY_COL.get(a)
        if col:
            out[col] = 1

    return out


def make_sample(row, header, idx, i):
    target = row_to_subattr(row, header, idx)
    rel = row[0]
    return {
        "id": f"upar_{i:06d}_{os.path.splitext(os.path.basename(rel))[0]}",
        "image": rel,
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
        ],
    }


def balanced_subset(samples, rows, header, idx, n, seed=0):
    """Inverse-frequency weighted sampling without replacement (Efraimidis-Spirakis).

    A row's weight is driven by its RAREST positive attribute, so rows carrying
    e.g. LowerBody-Color-Purple (0.2%) are pulled in preferentially. Plain 1-epoch
    training over the full corpus barely sees those classes.
    """
    freq = Counter()
    for r in rows:
        for h in header:
            if r[idx[h]] == "1":
                freq[h] += 1

    rng = random.Random(seed)
    keyed = []
    for s, r in zip(samples, rows):
        pos = [h for h in header if r[idx[h]] == "1"]
        w = max((1.0 / freq[h] for h in pos), default=1e-9)
        u = rng.random()
        keyed.append((u ** (1.0 / w), s, r))
    keyed.sort(key=lambda t: t[0], reverse=True)
    picked = keyed[:n]
    return [t[1] for t in picked], [t[2] for t in picked]


def report(rows, header, idx, label):
    n = len(rows)
    print(f"\n--- per-attribute positives: {label} (n={n:,}) ---")
    for h in header:
        p = sum(1 for r in rows if r[idx[h]] == "1")
        print(f"  {h:34s} {p:7,d}  ({100*p/n:5.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--balanced", type=int, default=0,
                    help="also write a balanced subset of this many samples")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows")
    ap.add_argument("--dump", action="store_true", help="print a few samples")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    header, rows = load_csv(CSV_PATH[args.split])
    idx = {h: i + 1 for i, h in enumerate(header)}
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows):,} rows from {CSV_PATH[args.split]}")

    samples = [make_sample(r, header, idx, i) for i, r in enumerate(rows)]

    # self-test: nested -> 40-binary round-trip must be exact except for the
    # deliberately-dropped secondary colours / secondary exclusive-group positives.
    exact = 0
    for s, r in zip(samples, rows):
        back = subattr_to_40binary(json.loads(s["conversations"][1]["value"]), header)
        if all(back[h] == (r[idx[h]] == "1") for h in header):
            exact += 1
    print(f"round-trip exact: {exact:,}/{len(rows):,} ({100*exact/len(rows):.2f}%)")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"upar_subattr_{args.split}.json")
    with open(out, "w") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"wrote {len(samples):,} samples -> {out}")

    if args.balanced:
        sub, subrows = balanced_subset(samples, rows, header, idx,
                                       args.balanced, args.seed)
        outb = os.path.join(args.out_dir,
                            f"upar_subattr_{args.split}_bal{args.balanced}.json")
        with open(outb, "w") as f:
            json.dump(sub, f, ensure_ascii=False)
        print(f"wrote {len(sub):,} balanced samples -> {outb}")
        report(rows, header, idx, "FULL")
        report(subrows, header, idx, f"BALANCED {args.balanced}")

    if args.dump:
        for s in samples[:3]:
            print(json.dumps(s, ensure_ascii=False, indent=2)[:1200])


if __name__ == "__main__":
    main()
