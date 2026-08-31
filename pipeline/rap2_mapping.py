#!/usr/bin/env python3
"""
Track 1.1 — RAP v2 (152 attributes) -> our UPAR-40 schema, test-split eval pool.

Why this exists
---------------
Every Phase-4 conclusion so far was scored on ONE attribute (gender), because the
Lotte GT carries no body attributes. The 40 body attributes that the public-PAR
warm-up was actually built to teach have never been measured on held-out data.

RAP v2 is the only corpus that closes that gap:
  * 84,928 indoor-surveillance crops, overhead-ish shopping-mall cameras
  * an official train/val/test partition (50,957 / 16,986 / 16,985)
  * VERIFIED not to appear in UPAR: UPAR train is PA100k (79,001) +
    Market1501 (10,000) + PETA (8,668) and nothing else. So RAP v2 is unseen by
    every arm, including the Stage-1 adapter.

Age boundary
------------
UPAR's per-source rates (Young .017-.052 / Adult .90-.97 / Old .014-.063) are the
PA100K convention: <18 / 18-60 / >60. RAP's bins are 16 / 17-30 / 31-45 / 46-60 /
60+, so the split lands at 16 rather than 18. That ~2-year offset is the one
unavoidable seam in the mapping and it is documented in the output metadata.

Colour
------
RAP carries 14 colour binaries per garment, ours 12. Gray->grey; Silver, Mixture
and Other all fold into "other". Like UPAR, a multi-colour row keeps the PRIMARY
colour = first positive in fixed column order.

Sampling
--------
The eval pool is inverse-frequency weighted (same Efraimidis-Spirakis scheme as
`upar_build_subattr_trainset.balanced_subset`). mA is a mean of per-attribute
BALANCED accuracies, so it is prevalence-independent and re-weighting the pool
cannot bias it -- it only buys tighter TPR estimates on the rare attributes.
Any prefix of the emitted list is itself a valid sample, so a 800-image run and a
2,000-image run stay directly comparable.

    python rap2_mapping.py --n 2000 --out out/rap2_eval_pool.json
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upar_build_subattr_trainset import COLORS, COLOR_COL, load_csv  # noqa: E402

RAP_ROOT = os.environ.get("RAP_ROOT", "/mnt/sdb1/pjh/par_datasets/rap/extracted")
MAT = f"{RAP_ROOT}/RAPv2_annotation/RAP_annotation/RAP_annotation.mat"
IMG_DIR = f"{RAP_ROOT}/v2/RAP_dataset"
UPAR_TRAIN_CSV = os.path.join(
    os.environ.get("UPAR_ANNOT", "/mnt/sdb1/pjh/par_datasets/upar_data/annotations/phase1/"),
    "train", "train.csv")

# ------------------------------------------------------------------ mapping
# 1-based column numbers as printed by RAP_annotation.attribute.
A = {
    "female": 1,          # 0=male 1=female 2=unknown (161 rows -> dropped)
    "age_le16": 2, "age_17_30": 3, "age_31_45": 4, "age_46_60": 5, "age_gt60": 6,
    "bald": 14, "long_hair": 15,
    "hat": 17, "glasses": 18, "sunglasses": 19,
    "ub_short_sleeve": 30,
    "lb_long_trousers": 46, "lb_shorts": 47, "lb_skirt": 48,
    "lb_short_skirt": 49, "lb_long_skirt": 50, "lb_dress": 51,
    "lb_jeans": 52, "lb_tight_trousers": 53,
    "backpack": 89, "shoulder_bag": 90, "hand_bag": 91,
    "plastic_bag": 94, "paper_bag": 95,
    "viewpoint": 112,
}

# RAP colour column -> our colour value. Order below IS the primary-colour
# precedence, mirroring the fixed CSV column order UPAR uses.
UB_COLOR = [(32, "black"), (33, "white"), (34, "grey"), (35, "red"),
            (36, "green"), (37, "blue"), (38, "other"), (39, "yellow"),
            (40, "brown"), (41, "purple"), (42, "pink"), (43, "orange"),
            (44, "other"), (45, "other")]
LB_COLOR = [(54, "black"), (55, "white"), (56, "grey"), (57, "red"),
            (58, "green"), (59, "blue"), (60, "other"), (61, "yellow"),
            (62, "brown"), (63, "purple"), (64, "pink"), (65, "orange"),
            (66, "other"), (67, "other")]

VIEWPOINT = {1: "front", 2: "back", 3: "left", 4: "right"}


def row_to_subattr(r):
    """One RAP row (152 ints, 0-indexed array) -> our nested sub-attribute dict."""
    g = lambda k: int(r[A[k] - 1])  # noqa: E731

    if g("age_le16"):
        age = "young"
    elif g("age_gt60"):
        age = "old"
    else:
        age = "adult"

    # Same precedence rule as UPAR: resolve toward the MORE SPECIFIC value, so a
    # row flagged both bald and long-haired reads as bald.
    hair = "bald" if g("bald") else ("long" if g("long_hair") else "short")

    def colour(table):
        for col, name in table:
            if int(r[col - 1]):
                return name
        return "other"

    skirt = g("lb_skirt") or g("lb_short_skirt") or g("lb_long_skirt") or g("lb_dress")
    acc = []
    if g("backpack"):
        acc.append("backpack")
    if g("shoulder_bag") or g("hand_bag") or g("plastic_bag") or g("paper_bag"):
        acc.append("bag")
    if g("glasses"):
        acc.append("glasses")
    if g("sunglasses"):
        acc.append("sunglasses")
    if g("hat"):
        acc.append("hat")

    return {
        "gender": "female" if g("female") == 1 else "male",
        "age_group": age,
        "hair": {"length": hair},
        "upper": {
            "length": "short" if g("ub_short_sleeve") else "long",
            "color": colour(UB_COLOR),
        },
        "lower": {
            # RAP has no "short trousers/skirt" umbrella; shorts and short skirts
            # are the only two short lower garments it labels.
            "length": "short" if (g("lb_shorts") or g("lb_short_skirt")) else "long",
            "color": colour(LB_COLOR),
            "type": "skirt" if skirt else "trousers",
        },
        "accessories": acc,
    }


def subattr_to_40binary(obj, header):
    """Nested dict -> the 40 UPAR binaries, in UPAR header order.

    Deliberately a copy of upar_build_subattr_trainset.subattr_to_40binary rather
    than an import of it: the eval must not silently change if the trainset
    builder's tolerance rules are ever loosened.
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
        col = {"backpack": "Accessory-Backpack", "bag": "Accessory-Bag",
               "glasses": "Accessory-Glasses-Normal",
               "sunglasses": "Accessory-Glasses-Sun",
               "hat": "Accessory-Hat"}.get(a)
        if col:
            out[col] = 1
    return out


def weighted_sample(bins, n, seed=0):
    """Efraimidis-Spirakis: key = U^(1/w), take the n largest. w = 1/rarest rate."""
    rng = random.Random(seed)
    rate = bins.mean(axis=0)
    rate[rate <= 0] = 1.0
    keyed = []
    for i, row in enumerate(bins):
        pos = np.where(row > 0)[0]
        w = 1.0 / rate[pos].min() if len(pos) else 1.0
        keyed.append((rng.random() ** (1.0 / w), i))
    keyed.sort(reverse=True)
    return [i for _, i in keyed[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="eval pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--natural", action="store_true",
                    help="uniform random instead of inverse-frequency weighted")
    ap.add_argument("--min-pos", type=int, default=30,
                    help="G1: attributes with fewer positives in the pool are "
                         "excluded from mA (still reported)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "out", "rap2_eval_pool.json"))
    args = ap.parse_args()

    header, _ = load_csv(UPAR_TRAIN_CSV)
    ann = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)["RAP_annotation"]
    data, names = ann.data, [str(x) for x in ann.name]
    test_idx = np.asarray(ann.partition_attribute[0].test_index).astype(int) - 1

    kept, dropped_gender, missing_img = [], 0, 0
    for i in test_idx:
        if int(data[i, A["female"] - 1]) == 2:      # gender annotated "unknown"
            dropped_gender += 1
            continue
        path = os.path.join(IMG_DIR, names[i])
        if not os.path.exists(path):
            missing_img += 1
            continue
        sub = row_to_subattr(data[i])
        kept.append({
            "rap_index": int(i),
            "name": names[i],
            "image": path,
            "viewpoint": VIEWPOINT.get(int(data[i, A["viewpoint"] - 1]), "unknown"),
            "subattr": sub,
            "binary": subattr_to_40binary(sub, header),
        })

    print(f"test split          : {len(test_idx)}")
    print(f"  dropped gender==2 : {dropped_gender}")
    print(f"  missing image     : {missing_img}")
    print(f"  usable            : {len(kept)}")

    bins = np.array([[s["binary"][h] for h in header] for s in kept])
    pick = (random.Random(args.seed).sample(range(len(kept)), min(args.n, len(kept)))
            if args.natural else weighted_sample(bins, args.n, args.seed))
    pool = [kept[i] for i in pick]
    pbin = np.array([[s["binary"][h] for h in header] for s in pool])

    print(f"\n{'attribute':32s} {'full-test':>9s} {'pool+':>6s} {'pool%':>7s}  mA?")
    scored = []
    for j, h in enumerate(header):
        npos = int(pbin[:, j].sum())
        ok = args.min_pos <= npos <= len(pool) - args.min_pos
        if ok:
            scored.append(h)
        print(f"{h:32s} {bins[:, j].mean():9.4f} {npos:6d} {npos/len(pool):7.2%}"
              f"  {'yes' if ok else 'NO'}")

    print(f"\npool               : {len(pool)}")
    print(f"attributes scored  : {len(scored)}/40")
    print("viewpoint mix      :", dict(Counter(s["viewpoint"] for s in pool)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "meta": {
                "source": "RAP v2 official test split",
                "mat": MAT,
                "sampling": "natural" if args.natural else "inverse-frequency",
                "seed": args.seed,
                "usable_test_rows": len(kept),
                "min_pos": args.min_pos,
                "age_boundary_note": "RAP splits at 16/60, UPAR(PA100K) at 18/60",
                "upar_contains_rap": False,
            },
            "header": header,
            "scored_attributes": scored,
            "samples": pool,
        }, f)
    print("wrote", args.out)

    # G1
    ok = len(scored) >= 20
    print(f"\nGATE G1: {len(scored)} scored attributes (need >=20) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
