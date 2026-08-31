"""SAM3 chunked full-dataset tracking.

Treats each session's sparse frames as video, but processes them in fixed-length
CHUNKS (fresh start_session + text prompt per chunk) so the memory bank stays
bounded (full-session at 4K/1920 OOMs). Objects get new ids per chunk -> fragments,
which is acceptable (fragments treated as separate; no cross-chunk re-linking).

Downscales frames to LONG_SIDE. Boxes stored in ORIGINAL pixel coords.
Env: sam3_poc.  Run with CUDA_VISIBLE_DEVICES=<gpu>.
"""
import argparse
import gc
import json
import os

OUTDIR = os.environ.get("TRACKPAR_OUT",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "out"))
import re
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_video_predictor

ANNOT = os.environ.get("LOTTE_ANNOT", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/annotations/lotte_tta_sft.json")
IMG_ROOT = os.environ.get("LOTTE_IMAGES", "/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/images")
WORK = OUTDIR
LONG_SIDE = 1920


def parse_id(sid):
    m = re.match(r"(.+)_(\d+)$", sid)
    return m.group(1), int(m.group(2))


def ensure_frames(session, frames):
    """Downscale all session frames once; return (dir, scale_to_orig)."""
    fdir = os.path.join(WORK, session, f"frames_{LONG_SIDE}")
    os.makedirs(fdir, exist_ok=True)
    scale = None
    for i, (fnum, sid, rel_img) in enumerate(frames):
        dst = os.path.join(fdir, f"{i}.jpg")
        if scale is None:
            w0, h0 = Image.open(os.path.join(IMG_ROOT, rel_img)).size
            scale = max(w0, h0) / LONG_SIDE
        if not os.path.exists(dst):
            im = Image.open(os.path.join(IMG_ROOT, rel_img)).convert("RGB")
            w0, h0 = im.size
            s = LONG_SIDE / max(w0, h0)
            im.resize((round(w0 * s), round(h0 * s))).save(dst, quality=90)
    return fdir, scale


def chunk_frame_dir(base_dir, idxs, chunk_dir):
    os.makedirs(chunk_dir, exist_ok=True)
    for j, gi in enumerate(idxs):
        dst = os.path.join(chunk_dir, f"{j}.jpg")
        if not os.path.exists(dst):
            os.symlink(os.path.join(base_dir, f"{gi}.jpg"), dst)
    return chunk_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--sessions", nargs="*", default=None, help="subset of sessions; default all")
    ap.add_argument("--max_sessions", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    with open(ANNOT) as f:
        data = json.load(f)
    sess_frames = defaultdict(list)
    for it in data:
        s, fnum = parse_id(it["id"])
        sess_frames[s].append((fnum, it["id"], it["image"]))
    for s in sess_frames:
        sess_frames[s].sort(key=lambda x: x[0])

    sessions = args.sessions or sorted(sess_frames.keys())
    if args.max_sessions:
        sessions = sessions[: args.max_sessions]

    print("loading SAM3 video predictor ...", flush=True)
    predictor = build_sam3_video_predictor(gpus_to_use=[0])
    print("predictor ready", flush=True)

    summary = []
    t_all = time.time()
    for session in sessions:
        frames = sess_frames[session]
        out_path = os.path.join(WORK, session, "track.json")
        if os.path.exists(out_path):
            print(f"skip {session} (exists)", flush=True)
            summary.append({"session": session, "skipped": True})
            continue

        base_dir, scale = ensure_frames(session, frames)
        W0, H0 = Image.open(os.path.join(IMG_ROOT, frames[0][2])).size  # original px
        n = len(frames)
        print(f"\n=== {session}: {n} frames, chunk={args.chunk}, orig={W0}x{H0} ===", flush=True)

        per_frame = {}   # global_idx -> {track_key, box_xyxy_orig, prob}
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        next_global_tid = 0

        for cstart in range(0, n, args.chunk):
            cidxs = list(range(cstart, min(cstart + args.chunk, n)))
            cdir = chunk_frame_dir(base_dir, cidxs, os.path.join(WORK, session, f"chunk_{cstart}"))
            resp = predictor.handle_request(request=dict(type="start_session", resource_path=cdir))
            sid = resp["session_id"]
            predictor.handle_request(request=dict(
                type="add_prompt", session_id=sid, frame_index=0, text="person"))
            # map this chunk's local obj_id -> a globally-unique track key
            local2global = {}
            for response in predictor.handle_stream_request(
                    request=dict(type="propagate_in_video", session_id=sid)):
                lidx = response["frame_index"]
                gidx = cidxs[lidx]
                out = response["outputs"]
                recs = []
                for k in range(len(out["out_obj_ids"])):
                    loid = int(out["out_obj_ids"][k])
                    if loid not in local2global:
                        local2global[loid] = f"{session[-8:]}_c{cstart}_{loid}"
                    b = np.asarray(out["out_boxes_xywh"][k], dtype=float)  # xywh, NORMALIZED [0,1]
                    x, y, w, h = b
                    box_orig = [round(x * W0, 1), round(y * H0, 1),
                                round((x + w) * W0, 1), round((y + h) * H0, 1)]
                    recs.append({"tid": local2global[loid],
                                 "box": box_orig,
                                 "prob": round(float(out["out_probs"][k]), 4)})
                per_frame[gidx] = recs
            predictor.handle_request(request=dict(type="close_session", session_id=sid))
            torch.cuda.empty_cache()
            gc.collect()

        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        all_tids = {r["tid"] for recs in per_frame.values() for r in recs}
        gt_pids = set()
        for fnum, sidd, rel in frames:
            it = next(x for x in data if x["id"] == sidd)
            for p in json.loads(it["conversations"][1]["value"]):
                if p.get("pid", -1) != -1:
                    gt_pids.add(p["pid"])

        mapping = [{"global_idx": i, "fnum": frames[i][0], "id": frames[i][1]} for i in range(n)]
        with open(out_path, "w") as f:
            json.dump({"session": session, "chunk": args.chunk, "scale_to_orig": scale,
                       "long_side": LONG_SIDE, "mapping": mapping, "per_frame": per_frame}, f)
        row = {"session": session, "n_frames": n, "elapsed_s": round(elapsed, 1),
               "s_per_frame": round(elapsed / n, 2), "peak_gpu_gb": round(peak, 1),
               "n_sam3_tracks": len(all_tids), "n_gt_pids": len(gt_pids)}
        summary.append(row)
        print(f"  DONE {row}", flush=True)

    with open(os.path.join(WORK, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== ALL DONE in {(time.time()-t_all)/60:.1f} min ===", flush=True)
    for r in summary:
        print(r, flush=True)


if __name__ == "__main__":
    main()
