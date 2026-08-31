#!/usr/bin/env python3
"""Rewrite the absolute paths baked into the pipeline so config/paths.sh drives them.

The scripts came out of a research tree where every path was a literal. This
replaces each one with `os.environ.get("VAR", <original literal>)`, so behaviour
on the original machine is unchanged and a new machine only edits paths.sh.

    python setup/patch_paths.py            # apply
    python setup/patch_paths.py --check    # report only, exit 1 if any remain

Idempotent: a literal already wrapped in os.environ.get is skipped.
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# literal -> environment variable. Order matters: longer paths first, so a
# prefix does not consume a more specific match.
SUBS = [
    ("/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/annotations/lotte_tta_sft.json",
     "LOTTE_ANNOT"),
    ("/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/lotte_tta.csv", "LOTTE_CSV"),
    ("/mnt/nvme0n1p1/pjh/datasets/lotte_cheonho/images", "LOTTE_IMAGES"),
    ("/mnt/nvme0n1p1/pjh/.cache/huggingface", "HF_HOME"),
    ("/mnt/sdb1/pjh/par_datasets/rap/extracted", "RAP_ROOT"),
    ("/mnt/sdb1/pjh/par_datasets/upar_data/annotations/phase1/", "UPAR_ANNOT"),
]

# Paths that point back into the ORIGINAL research tree. These cannot be made to
# work by substitution — the directory does not exist in a clone — so they are
# rewritten to this checkout's own out/ instead.
OUTDIR_LITERALS = [
    "/mnt/nvme0n1p1/pjh/Qwen-VL-Series-Finetune/scratch_sam2_poc/sam3_jobs/out/track_sam3",
    "/mnt/nvme0n1p1/pjh/Qwen-VL-Series-Finetune/scratch_sam2_poc/sam3_jobs",
]

OUTDIR_DECL = (
    'OUTDIR = os.environ.get("TRACKPAR_OUT",\n'
    '                        os.path.join(os.path.dirname(os.path.abspath(__file__)),\n'
    '                                     "..", "out"))\n')


def patch(path, check):
    src = open(path).read()
    orig = src

    def still_valid(text):
        try:
            ast.parse(text)
            return True
        except SyntaxError:
            return False

    for literal, var in SUBS:
        # skip anything already wrapped
        if f'os.environ.get("{var}"' in src:
            continue
        cand = src.replace(f'"{literal}"',
                           f'os.environ.get("{var}", "{literal}")')
        if cand == src:
            continue
        # A literal inside an implicit string concatenation ("a/" "b.csv")
        # cannot be swapped for a call - that is a syntax error, not a path
        # problem. Leave it and say so.
        if still_valid(cand):
            src = cand
        else:
            print(f"    skip {var} in {os.path.basename(path)}: literal is part "
                  f"of a string concatenation, left as-is")

    for literal in OUTDIR_LITERALS:
        if f'"{literal}/track_sam3"' in src or f'"{literal}"' in src:
            if "OUTDIR = os.environ.get" not in src:
                m = re.search(r"^import os$", src, re.M) or \
                    re.search(r"^import sys$", src, re.M)
                if m:
                    src = src[:m.end() + 1] + "\n" + OUTDIR_DECL + src[m.end() + 1:]
            src = src.replace(f'"{literal}/track_sam3"',
                              'os.path.join(OUTDIR, "track_sam3")')
            src = src.replace(f'"{literal}"', "OUTDIR")

    left = len(re.findall(r'"/(?:mnt|home)/[^"]*"', src))
    if src != orig and not check:
        ast.parse(src)          # never write something that will not import
        open(path, "w").write(src)
    return src != orig, left


def main():
    check = "--check" in sys.argv
    files = sorted(glob.glob(os.path.join(ROOT, "pipeline", "*.py")) +
                   glob.glob(os.path.join(ROOT, "eval", "*.py")))
    bad = 0
    print(f"{'file':38s} {'action':11s} absolute paths left")
    for p in files:
        changed, left = patch(p, check)
        act = "would edit" if (changed and check) else ("edited" if changed else "clean")
        flag = "" if left == 0 else f"  <-- {left}"
        print(f"{os.path.basename(p):38s} {act:11s} {left}{flag}")
        bad += left
    if bad:
        print(f"\n{bad} literal path(s) still present. Some are harmless (comments,\n"
              f"example strings); check them before shipping.")
    return 1 if (bad and check) else 0


if __name__ == "__main__":
    sys.exit(main())
