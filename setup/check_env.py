#!/usr/bin/env python3
"""Pre-flight. Run this before spending GPU hours.

    source config/paths.sh && python setup/check_env.py

Every check below exists because the corresponding mistake happened during
development and cost real time. Exits non-zero if anything required is wrong, so
it can gate a chain.
"""
import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join(ROOT, "out"))
results = []


def check(name, ok, detail="", required=True):
    tag = "ok  " if ok else ("FAIL" if required else "warn")
    results.append(tag)
    print(f"[{tag}] {name}" + (f"  —  {detail}" if detail else ""))
    return ok


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true",
                    help="also check the SAM3 environment, which only stage 1 needs")
    a = ap.parse_args()

    print("TrackPAR pre-flight\n" + "=" * 66)

    # --- config ----------------------------------------------------------
    imgs = os.environ.get("LOTTE_IMAGES", "")
    check("config/paths.sh has been sourced", bool(imgs),
          "" if imgs else "run `source config/paths.sh` first")
    check("frame directory exists", os.path.isdir(imgs), imgs or "(unset)")
    check("instance annotations exist",
          os.path.isfile(os.environ.get("LOTTE_ANNOT", "")),
          os.environ.get("LOTTE_ANNOT", "(unset)"))
    check("ground-truth CSV exists (needed only to score)",
          os.path.isfile(os.environ.get("LOTTE_CSV", "")),
          os.environ.get("LOTTE_CSV", "(unset)"), required=False)

    # --- model cache ------------------------------------------------------
    # A shell profile pointing HF_HOME at an unwritable mount makes every model
    # load fail with a PermissionError that reads like a HuggingFace outage.
    hf = os.environ.get("HF_HOME", "")
    w = False
    if hf:
        try:
            os.makedirs(hf, exist_ok=True)
            w = os.access(hf, os.W_OK)
        except OSError:
            w = False
    check("HF_HOME is writable", w,
          hf if w else f"{hf or '(unset)'} — model downloads will fail")

    # --- python deps ------------------------------------------------------
    try:
        import torch
        import transformers
        tv = transformers.__version__
        check("torch sees a GPU", torch.cuda.is_available(),
              f"{torch.cuda.device_count()} device(s), cuda {torch.version.cuda}")
        # transformers 4.x lacks the Qwen3.5-VL classes the pipeline imports.
        check("transformers >= 5", int(tv.split(".")[0]) >= 5,
              f"{tv}" + ("" if int(tv.split(".")[0]) >= 5 else
                         " — 4.x cannot import the Qwen3.5-VL model classes"))
    except Exception as e:
        check("torch / transformers import", False, str(e))

    # --- the adapter, and the file everyone forgets ------------------------
    ad = os.environ.get("IDENTITY_ADAPTER", "")
    have = os.path.isdir(ad)
    check("identity adapter present", have,
          ad if have else f"{ad} — run setup/fetch_weights.sh, or unset "
                          f"IDENTITY_ADAPTER to run identity on the base model",
          required=False)
    if have:
        lora = os.path.join(ad, "adapter_model.safetensors")
        nonlora = os.path.join(ad, "non_lora_state_dict.bin")
        check("adapter_model.safetensors", size(lora) > 0, human(size(lora)))
        # PeftModel.from_pretrained loads the LoRA and silently ignores the
        # other file, which holds the trained vision tower and merger. Nothing
        # raises; you evaluate a model that never existed.
        n = size(nonlora)
        check("non_lora_state_dict.bin", n > 0,
              human(n) + " (vision tower + merger)" if n else
              "MISSING — loading without it gives a base vision tower under a "
              "fine-tuned adapter, and nothing warns you")

    # --- prompts ----------------------------------------------------------
    # STYLE names the parser and PROMPT names the text, and they have to agree.
    # A mismatch is the one failure here that produces no error at all: the model
    # answers, the answer is valid JSON, the fields the parser wants are absent,
    # and the attribute comes back empty. Cheaper to catch before the run.
    STYLES = ("meta", "svfd", "plain", "trueonly", "padq")
    for attr in ("EXPOSED", "WATCHED"):
        style = os.environ.get(f"{attr}_STYLE", "")
        p = os.environ.get(f"{attr}_PROMPT", "")
        check(f"{attr}_STYLE is known", style in STYLES,
              f"{style or '(unset)'} — expected one of {', '.join(STYLES)}")
        note = ""
        if style in STYLES and style != "meta":
            note = (f" — note {attr}_STYLE={style} builds its own text, so this "
                    f"file is not read")
        check(f"{attr}_PROMPT file", os.path.isfile(p),
              (os.path.basename(p) or "(unset)") + note)

    # --- every pipeline module imports ------------------------------------
    # A clone missing one file otherwise fails only when the run reaches it,
    # which during development meant a crash after model load. Three files were
    # missing on the first attempt, all pulled in transitively.
    sys.path[:0] = [os.path.join(ROOT, "pipeline"), os.path.join(ROOT, "eval"),
                    os.path.join(ROOT, "src")]
    MODS = ["rap2_mapping", "rap2_eval", "tvlm_pseudo_subattr", "multiimg_eval",
            "age_eval", "phase1_build_all_fragments", "exp10_repr_infer",
            "exp11_model_infer", "exp20_unified_infer", "momentary_targeted_eval",
            "momentary_k1_control", "merge_deliverable", "full_grid",
            "momentary_deploy_grid", "k1_grid"]
    broken = []
    for m in MODS:
        try:
            importlib.import_module(m)
        except Exception as e:
            broken.append(f"{m} ({type(e).__name__}: {e})")
    check(f"all {len(MODS)} pipeline modules import", not broken,
          "" if not broken else "; ".join(broken[:2]))

    # --- SAM3, optional ---------------------------------------------------
    if a.stage1:
        try:
            importlib.import_module("sam3.model_builder")
            check("sam3.model_builder importable", True)
        except Exception as e:
            check("sam3.model_builder importable", False,
                  f"{e} — stage 1 only; see README. Stages 2-6 do not need it.")

    # --- what already exists ---------------------------------------------
    print("\nstage outputs" + "\n" + "-" * 66)
    for lab, p in (("1 tracks", os.path.join(OUTDIR, "track_sam3")),
                   ("2 fragments", os.path.join(OUTDIR, "phase1_fragments.json")),
                   ("3 gender", os.path.join(OUTDIR, "identity.json")),
                   ("4 age", os.path.join(OUTDIR, "age.json")),
                   ("5 momentary", os.path.join(OUTDIR, "momentary_exposed.json")),
                   ("6 labels", os.path.join(OUTDIR, "labels.json"))):
        print(f"  {lab:14s} {'present' if os.path.exists(p) else 'not yet'}  {p}")

    n_fail = results.count("FAIL")
    print("\n" + "=" * 66)
    if n_fail:
        print(f"{n_fail} required check(s) failed — fix before running")
        return 1
    print("environment looks runnable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
