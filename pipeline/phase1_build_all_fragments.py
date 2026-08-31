"""Phase 1 prep: build inference fragments for EVERY SAM3 track (not only the
GT-matched ones), which is what a real run on unlabelled data looks like.

Differences from exp4_build_fragments (the eval-set builder):
  - no GT requirement: a track is kept on its own merits (length / detection score),
    because at deploy time we do not know which people are annotated.
  - GT labels are attached WHEN AVAILABLE (dominant matched pid) purely so the run
    can be scored; tracks without GT still get a fragment and will get a pseudo-label.
  - every selected frame keeps its fnum + box, so the track-level answer can be
    written back onto each frame's person box.

Output: out/phase1_fragments.json
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(BASE, "out"))
ANNOT = os.environ.get("LOTTE_ANNOT", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/annotations/lotte_tta_sft.json")
TRACK_DIR = os.path.join(OUTDIR, "track_sam3")
IOU_MATCH = 0.5
K = 4


def parse_id(sid):
    m = re.match(r"(.+)_(\d+)$", sid)
    return m.group(1), int(m.group(2))


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_len", type=int, default=2, help="min frames a track must live")
    ap.add_argument("--min_prob", type=float, default=0.5, help="min mean detection score")
    ap.add_argument("--out", default=os.path.join(OUTDIR, "phase1_fragments.json"))
    args = ap.parse_args()

    data = json.load(open(ANNOT))
    gt = defaultdict(lambda: defaultdict(dict))
    pid_gender, pid_age = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    img_of, n_gt_boxes = {}, 0
    for it in data:
        s, fnum = parse_id(it["id"])
        img_of[(s, fnum)] = it["image"]
        for p in json.loads(it["conversations"][1]["value"]):
            if not p.get("bbox"):
                continue
            n_gt_boxes += 1
            pid = p.get("pid", -1)
            if pid == -1:
                continue
            gt[s][fnum][pid] = p["bbox"]
            if p.get("gender"):
                pid_gender[s][pid].append(str(p["gender"]).upper())
            if isinstance(p.get("age"), (int, float)):
                pid_age[s][pid].append(int(p["age"]))

    frags, dropped = [], Counter()
    gt_hit = 0
    for tf in sorted(os.listdir(TRACK_DIR)):
        tpath = os.path.join(TRACK_DIR, tf, "track.json")
        if not os.path.isfile(tpath):
            continue
        d = json.load(open(tpath))
        session = d["session"]
        idx2fnum = {m["global_idx"]: m["fnum"] for m in d["mapping"]}
        tid_data = defaultdict(list)      # tid -> [(fnum, box, prob, matched_pid|None)]
        for gidx_str, recs in d["per_frame"].items():
            fnum = idx2fnum[int(gidx_str)]
            gtf = gt[session].get(fnum, {})
            pairs = []
            for ri, r in enumerate(recs):
                for pid, gbox in gtf.items():
                    v = iou(r["box"], gbox)
                    if v >= IOU_MATCH:
                        pairs.append((v, ri, pid))
            pairs.sort(reverse=True)
            used_r, used_pid, match = set(), set(), {}
            for v, ri, pid in pairs:
                if ri in used_r or pid in used_pid:
                    continue
                used_r.add(ri); used_pid.add(pid); match[ri] = pid
            for ri, r in enumerate(recs):
                tid_data[r["tid"]].append((fnum, r["box"], r.get("prob", 1.0), match.get(ri)))

        for tid, entries in tid_data.items():
            if len(entries) < args.min_len:
                dropped["short"] += 1
                continue
            mprob = float(np.mean([e[2] for e in entries]))
            if mprob < args.min_prob:
                dropped["low_score"] += 1
                continue
            ent = sorted(entries, key=lambda x: x[0])
            if len(ent) <= K:
                sel = ent
            else:
                idxs = np.linspace(0, len(ent) - 1, K).round().astype(int)
                sel = [ent[i] for i in idxs]
            pids = [e[3] for e in ent if e[3] is not None]
            dom_pid, purity, g, a = None, 0.0, None, None
            if pids:
                dom_pid, dom_n = Counter(pids).most_common(1)[0]
                purity = dom_n / len(ent)
                gg = Counter(pid_gender[session][dom_pid]).most_common(1)
                g = gg[0][0] if gg else None
                aa = pid_age[session][dom_pid]
                a = int(np.median(aa)) if aa else None
                gt_hit += 1
            frags.append({
                "session": session, "tid": tid, "gt_pid": dom_pid,
                "purity": round(purity, 3), "n_track_frames": len(ent),
                "mean_prob": round(mprob, 3),
                "gt_gender": g, "gt_age": a,
                "frames": [{"image": img_of[(session, fn)], "box": box, "fnum": fn}
                           for fn, box, _p, _pid in sel if (session, fn) in img_of],
                "all_frames": [{"fnum": fn, "box": box} for fn, box, _p, _pid in ent],
            })

    frags = [f for f in frags if f["frames"]]
    json.dump(frags, open(args.out, "w"))
    lens = [f["n_track_frames"] for f in frags]
    print(f"tracks kept: {len(frags)}   dropped: {dict(dropped)}")
    print(f"  with a GT-matched pid: {gt_hit} ({100*gt_hit/max(1,len(frags)):.0f}%)")
    print(f"  track length: mean {np.mean(lens):.1f}  median {np.median(lens):.0f}")
    print(f"  person-boxes covered by kept tracks: {sum(lens)}  (GT boxes in dataset: {n_gt_boxes})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
