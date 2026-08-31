#!/usr/bin/env python3
"""Every momentary prompt, scored on the deployment path.

This is the table that decides what ships. The deliverable runs
exp20_unified_infer: K=8 frames per call, eight answers out, over SAM3 track
frames. `momentary_targeted_eval` shows one image and samples annotated
instances directly, which is a useful ablation and a different population — F1
from one does not transfer to the other.

Everything here is restricted to the region that was actually annotated.
exposed/watched are labelled in only 4 of the 11 sessions, and 09.32.15.5080 is
labelled only from frame 5251 onward; scoring across those boundaries counts
unlabelled positives as false alarms and roughly halves every figure.

    python momentary_deploy_grid.py
"""
import glob
import json
import os
import re
import sys

JOBS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(JOBS, "out"))
O = OUTDIR

POS = {"IPS_2025-09-29.09.04.56.4089", "IPS_2025-09-29.09.12.56.6420",
       "IPS_2025-09-29.09.20.56.8680", "IPS_2025-09-29.09.32.15.5080"}
PARTIAL = "IPS_2025-09-29.09.32.15.5080"
PARTIAL_FROM = 5251

# label -> filename glob. The first five were run before the observation prompts
# existed; the last two come from E33.
# Each entry lists the fresh run first and the pre-existing one as a fallback, so
# the table shows what is already known while stage 2 works through the rerun and
# switches over as each new arm lands.
RUNS = [
    ("plain", ["pm_plain_base9b_full_mask_K8_evenly_sh*.json", "exp20_base9b_full_mask_K8_evenly_sh*.json"]),
    # The exp20 --prompt padq FLAG is currently an alias of plain: build_prompt
    # tests `if style in ("plain", "padq")` and parse_unified shares one branch.
    # So this row is a plain repeat, and is labelled as one. It is NOT the real
    # PADQ, which is the per-attribute full-scene design two rows below; the two
    # only ever shared a name.
    ("plain (repeat of row 1)", ["pm_padq_base9b_full_mask_K8_evenly_padq_sh*.json", "exp20_base9b_full_mask_K8_evenly_padq_sh*.json"]),
    ("svfd", ["pm_svfd_base9b_full_mask_K8_evenly_svfd_sh*.json", "exp20_base9b_full_mask_K8_evenly_svfd_sh*.json"]),
    ("trueonly", ["pm_trueonly_base9b_full_mask_K8_evenly_trueonly_sh*.json", "exp20_base9b_full_mask_K8_evenly_trueonly_sh*.json"]),
    ("metav4  (deliverable)", ["pm_metav4_base9b_full_mask_K8_evenly_meta_sh*.json", "exp20_base9b_full_mask_K8_evenly_meta_sh*.json"]),
    ("eyes / gaze", ["pm_eyes_base9b_full_mask_K8_evenly_meta_sh*.json"]),
    ("eyes+nose+mouth", ["pm_features_base9b_full_mask_K8_evenly_meta_sh*.json"]),
    ("combined", ["pm_combined_base9b_full_mask_K8_evenly_meta_sh*.json"]),
    # The REAL PADQ: full scene, target named by bounding box, one attribute per
    # call, prompts/prompt_eng/{exposed_v3,watched_v2}.txt. Two runs, so exposed
    # comes from one and watched from the other. ADAPTED for K=8 — the prompt was
    # written for one image holding n people and here gets K frames of the same
    # person, with the K boxes passed as the n detections.
    ("PADQ exposed_v3 *+", ["pm_padqE_base9b_full_mask_K8_evenly_meta_sh*.json"]),
    ("PADQ watched_v2 *+", ["pm_padqW_base9b_full_mask_K8_evenly_meta_sh*.json"]),
]

# The diagnostic path: one image in, one answer out, sampled from the annotated
# instances of a single session. Same prompt files as above, so a row compares a
# prompt against itself across settings rather than against a different prompt.
# Each entry lists the filename stems to try, in order. The eyes prompt was run
# before this matrix existed and its results live under e21_eyes2_*, so a
# single-name lookup showed that row as absent — and it is the best exposed arm
# on both sessions, which made the table read backwards.
SINGLE = [("combined", ["combined"]), ("plain", ["plain"]),
          ("padq  (= plain, repeat)", ["padq"]), ("svfd", ["svfd"]),
          ("trueonly", ["trueonly"]), ("metav4  (deliverable)", ["metav4"]),
          ("sub-attribute schema", ["subattr"]),
          ("eyes / gaze", ["eyes", "@e21_eyes2"]),
          ("eyes+nose+mouth", ["features"])]


def noise_floor(deploy_rows):
    """What a difference has to beat before it means anything.

    plain and padq are the same prompt (see RUNS above), so the gap between them
    is pure run-to-run variation rather than a prompt effect. Worth printing
    beside the table: without it, two arms a hundredth apart look ranked.
    """
    # Score every stored run of the plain prompt, not just the two rows the table
    # happens to show. There are three: the original exp20 plain, exp20 padq
    # (same prompt, different filename), and the fresh pm_plain from this matrix.
    # Comparing only the latter two reported a 0.000 spread, because those two
    # agree exactly — which would license reading a 0.005 gap as a ranking.
    fm, tv = frame_index()
    runs = [("exp20 plain", "exp20_base9b_full_mask_K8_evenly_sh*.json"),
            ("exp20 padq", "exp20_base9b_full_mask_K8_evenly_padq_sh*.json"),
            ("pm_plain (fresh)", "pm_plain_base9b_full_mask_K8_evenly_sh*.json"),
            ("pm_padq (fresh)", "pm_padq_base9b_full_mask_K8_evenly_padq_sh*.json")]
    got = []
    for lab, pat in runs:
        d = score([pat], fm, tv)
        if d and d.get("exposed"):
            got.append((lab, d["exposed"]["f1"], d["exposed"]["pred"]))
    if len(got) < 2:
        return
    import datetime
    def when(pat):
        fs = sorted(glob.glob(os.path.join(O, pat)))
        if not fs:
            return "?"
        return datetime.datetime.fromtimestamp(
            os.path.getmtime(fs[0])).strftime("%m-%d")
    f1s = [f for _, f, _ in got]
    spread = max(f1s) - min(f1s)
    dates = {lab: when(pat) for (lab, pat) in runs}
    print(f"\nNOISE FLOOR: {len(got)} runs of the SAME prompt (plain; padq is the "
          f"same text\n  under another filename):")
    for lab, f, pr in got:
        print(f"    {lab:20s} {dates.get(lab,'?')}   exposed F1 {f:.4f}   "
              f"predicted+ {pr:.1%}")
    gens = {dates.get(lab, "?")[:2] for lab, _, _ in got}
    print(f"  spread {spread:.4f} exposed F1.")
    if len(gens) > 1:
        # This matters: exp20_unified_infer.py changed between the two dates, and
        # these files store predictions parsed by the code of their own day. A
        # spread measured across generations confounds run noise with a code
        # change, so it is a ceiling on the noise floor, not the noise floor.
        print("  WARNING: these runs span more than one month, and")
        print("  exp20_unified_infer.py changed in between. The stored files hold")
        print("  predictions parsed by the code of their day, so this spread mixes")
        print("  run-to-run noise with a code difference and can only be read as an")
        print("  UPPER BOUND. A same-generation pair is needed to pin it down.")
    else:
        print(f"  Treat any gap in this table below ~{spread:.3f} as "
              f"indistinguishable from a rerun.")
    same = [(lab, f) for lab, f, _ in got if dates.get(lab, "?") == max(dates.values())]
    if len(same) >= 2:
        sp = max(f for _, f in same) - min(f for _, f in same)
        print(f"  SAME-GENERATION PAIR ({max(dates.values())}): "
              + ", ".join(f"{lab} {f:.4f}" for lab, f in same))
        print(f"  spread {sp:.4f} — this is the floor to use. It is measured with "
              f"one\n  code version, so it isolates run-to-run variation.")
    print("  The single-image path does not have this problem: the same prompt "
          "run\n  twice there produced byte-identical output files.")


def frame_index():
    sys.path.insert(0, JOBS)
    import tvlm_pseudo_subattr as tv
    fm = {}
    for t in json.load(open(tv.FRAGMENTS)):
        for fr in t.get("frames") or []:
            fm[(t["tid"], fr.get("fnum"))] = (fr["image"], fr["box"])
    return fm, tv


def trusted(img):
    m = re.search(r"(IPS_[0-9.\-]+)", img)
    s = m.group(1) if m else "?"
    if s not in POS:
        return False
    if s != PARTIAL:
        return True
    n = re.search(r"_(\d{4,})\.jpg", img)
    return bool(n) and int(n.group(1)) >= PARTIAL_FROM


def score(pats, fm, tv):
    if isinstance(pats, str):
        pats = [pats]
    got = {"exposed": [[], []], "watched": [[], []]}
    files = []
    for pat in pats:
        files = sorted(glob.glob(os.path.join(O, pat)))
        if files:
            break
    if not files:
        return None
    for f in files:
        for r in json.load(open(f)):
            for fr in r.get("frames") or []:
                k = (r.get("tid"), fr.get("fnum"))
                if k not in fm:
                    continue
                img, box = fm[k]
                if not trusted(img):
                    continue
                for a in ("exposed", "watched"):
                    v = fr.get(a)
                    if v is None:
                        continue
                    gt = tv.gt_momentary(img, box)[0 if a == "exposed" else 1]
                    if gt is None:
                        continue
                    got[a][0].append(bool(v))
                    got[a][1].append(bool(gt))
    out = {}
    for a, (pr, gt) in got.items():
        if not pr:
            out[a] = None
            continue
        tp = sum(1 for x, y in zip(pr, gt) if x and y)
        fp = sum(1 for x, y in zip(pr, gt) if x and not y)
        fn = sum(1 for x, y in zip(pr, gt) if (not x) and y)
        tn = len(pr) - tp - fp - fn
        rc = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        out[a] = {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                  "bacc": (rc + tnr) / 2, "pred": sum(pr) / len(pr),
                  "gt": sum(gt) / len(gt), "n": len(pr)}
    return out


def score_single(tag):
    """One prompt on the diagnostic path: one image in, one answer out.

    The derivation is imported from momentary_targeted_eval rather than
    reimplemented. An earlier hand-rolled copy of these rules read exposed at
    0.133 where the eval log said 0.752, which is the whole reason there is one
    reader now.
    """
    # A leading "@" names a file that is NOT a pm_single_* run — an older result
    # kept under its original name.
    p = (os.path.join(O, f"{tag[1:]}.json") if tag.startswith("@")
         else os.path.join(O, f"pm_single_{tag}.json"))
    if not os.path.exists(p):
        return None
    sys.path.insert(0, JOBS)
    import momentary_targeted_eval as M

    rows = [r for r in json.load(open(p)) if isinstance(r.get("subattr"), dict)]
    if not rows:
        return None
    out = {"n": len(rows)}
    for a in ("exposed", "watched"):
        pairs = [(M.derive(r["subattr"], a), bool(r[f"gt_{a}"])) for r in rows]
        pairs = [(pr, gt) for pr, gt in pairs if pr is not None]
        if not pairs:
            out[a] = None
            continue
        pr = [x for x, _ in pairs]
        gt = [y for _, y in pairs]
        tp = sum(1 for x, y in zip(pr, gt) if x and y)
        fp = sum(1 for x, y in zip(pr, gt) if x and not y)
        fn = sum(1 for x, y in zip(pr, gt) if (not x) and y)
        tn = len(pr) - tp - fp - fn
        rc = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        out[a] = {"f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
                  "bacc": (rc + tnr) / 2, "pred": sum(pr) / len(pr),
                  "gt": sum(gt) / len(gt), "n": len(pr)}
    return out


def single_rows():
    rows = []
    for lab, stems in SINGLE:
        w = f3 = None
        for base in stems:
            w = w or score_single(f"{base}_w")
            f3 = f3 or score_single(f"{base}_f3")
        if w or f3:
            rows.append((lab, w, f3))
    return rows


def single_table(rows):
    """Both sessions side by side, because one number hides the whole story.

    09.20 carries the high-prevalence exposed labels and is the only session
    where watched is scorable at all; 09.32 is the low-prevalence half where
    every prompt so far has collapsed. A prompt that wins on 09.20 and falls
    over on 09.32 has not won.
    """
    if not rows:
        return

    print("\n\nSINGLE-IMAGE PATH — one crop per call, 800 sampled instances")
    print("=" * 78)
    print(f"{'prompt':24s} {'09.20 exp':>10s} {'09.32 exp':>10s} "
          f"{'09.20 wat':>10s} {'pred+ 09.20':>12s} {'pred+ 09.32':>12s}")
    cell = lambda d, a: (f"{d[a]['f1']:.3f}" if d and d.get(a) else "—")
    rate = lambda d: (f"{d['exposed']['pred']:.0%}" if d and d.get("exposed")
                      else "—")
    for lab, w, f3 in rows:
        print(f"{lab:24s} {cell(w,'exposed'):>10s} {cell(f3,'exposed'):>10s} "
              f"{cell(w,'watched'):>10s} {rate(w):>12s} {rate(f3):>12s}")
    # The two predicted-positive columns are the point of this table. The true
    # rates differ by 2.8x between the sessions; a prompt whose predicted rate
    # does NOT move with them is applying a threshold it got from the prompt
    # rather than from the image, and its F1 on the high-prevalence session is
    # measuring the base rate.
    print(f"{'':24s} {'':>10s} {'':>10s} {'':>10s} "
          f"{'true 39.5%':>12s} {'true 13.9%':>12s}")
    e = next((w for _, w, _ in rows if w and w.get("exposed")), None)
    e3 = next((f for _, _, f in rows if f and f.get("exposed")), None)
    if e:
        print(f"\ntrue exposed rate: 09.20 {e['exposed']['gt']:.1%}"
              + (f", 09.32 {e3['exposed']['gt']:.1%}" if e3 else ""))
    print("F1 moves with prevalence, so the two session columns are NOT")
    print("comparable to each other — only down a column. None of these are")
    print("comparable to the deployment table above either: different population.")


def dump_json(deploy_rows, single_rows):
    """Write the same numbers the tables above print, as data.

    The report builder used to scrape this script's stdout and pick out lines
    with enough whitespace-separated fields. That worked until a prose line had
    six words in it, at which point the caveats started rendering as table rows.
    Anything that needs these numbers should read this file.
    """
    obj = {"deploy": {lab: d for lab, d in deploy_rows},
           "single": {lab: {"09.20": w, "09.32": f3}
                      for lab, w, f3 in single_rows}}
    p = os.path.join(O, "momentary_grid.json")
    json.dump(obj, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


def main():
    fm, tv = frame_index()
    rows = []
    for lab, pat in RUNS:
        d = score(pat, fm, tv)
        if d:
            rows.append((lab, d))
    if not rows:
        # The single-image screen usually runs first, so print it even when no
        # deployment arm has landed yet.
        print("no deployment runs found")
        sr = single_rows()
        single_table(sr)
        dump_json([], sr)
        return 1

    print("\nDEPLOYMENT PATH — exp20, K=8, full_mask, trusted region")
    print("=" * 78)
    print(f"{'prompt':24s} {'exp F1':>7s} {'exp bAcc':>9s} {'exp pred+':>10s} "
          f"{'wat F1':>7s} {'wat bAcc':>9s} {'n':>6s}")
    bx = max((r[1]["exposed"]["f1"] for r in rows if r[1].get("exposed")), default=0)
    bw = max((r[1]["watched"]["f1"] for r in rows if r[1].get("watched")), default=0)
    for lab, d in rows:
        e, w = d.get("exposed"), d.get("watched")
        mark = lambda v, best: f"{v:.3f}" + ("  <-" if v == best and v > 0 else "")
        print(f"{lab:24s} "
              f"{(mark(e['f1'], bx) if e else '—'):>7s} "
              f"{(f'{e[chr(98)+chr(97)+chr(99)+chr(99)]:.3f}' if e else '—'):>9s} "
              f"{(f'{e[chr(112)+chr(114)+chr(101)+chr(100)]:.1%}' if e else '—'):>10s} "
              f"{(mark(w['f1'], bw) if w else '—'):>7s} "
              f"{(f'{w[chr(98)+chr(97)+chr(99)+chr(99)]:.3f}' if w else '—'):>9s} "
              f"{(e['n'] if e else 0):6d}")
    e0 = next((r[1]["exposed"] for r in rows if r[1].get("exposed")), None)
    w0 = next((r[1]["watched"] for r in rows if r[1].get("watched")), None)
    if e0 and w0:
        print(f"\ntrue rate: exposed {e0['gt']:.1%}, watched {w0['gt']:.1%}")
    print("* PADQ sees the full scene with the target named by box, not a crop, and")
    print("  answers ONE attribute per call, so its exposed and watched come from")
    print("  different runs and each row is blank in the other column.")
    print("+ ADAPTED at K=8: written for one image with n people, given K frames of")
    print("  one person. Not PADQ in its native setting.")
    print("Same frames, same parser, same restriction for every row, so these F1s")
    print("are comparable to each other. They are NOT comparable to the")
    print("single-image numbers from momentary_targeted_eval, which sample a")
    print("different population at a different base rate.")
    noise_floor(rows)
    sr = single_rows()
    single_table(sr)
    dump_json(rows, sr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
