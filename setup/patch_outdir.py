#!/usr/bin/env python3
"""Route every script's out/ directory through $TRACKPAR_OUT.

Each script resolved out/ relative to its own file. That happened to make them
agree inside the original tree, but here it puts them in pipeline/out/ and
eval/out/ — two directories, neither of which is $TRACKPAR_OUT — so stage 3
cannot find what stage 2 wrote.

    python setup/patch_outdir.py          # apply
    python setup/patch_outdir.py --check  # report only

Idempotent: a file that already declares OUTDIR is left alone. That guard matters
because the declaration itself contains os.path.join(ANCHOR, "out"), which the
rewrite rules would otherwise collapse into OUTDIR = os.environ.get(..., OUTDIR).
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECL = 'OUTDIR = os.environ.get("TRACKPAR_OUT", os.path.join({a}, "out"))\n'


def patch(path, check):
    src = open(path).read()
    if "OUTDIR = " in src:
        return False, 0
    orig = src

    anchor = None
    for cand in ("JOBS", "BASE", "O"):
        # `BASE = os.path.dirname(...)` in the file that defines it, but
        # `BASE = E.BASE` in the two that re-export it from exp10.
        if re.search(rf"^{cand} = (os\.path\.dirname|[A-Z]\.[A-Z]+$)", src, re.M):
            anchor = cand
            break
    if anchor is None:
        return False, len(re.findall(r'["\{]out/', src))

    src = src.replace(f'os.path.join({anchor}, "out/', 'os.path.join(OUTDIR, "')
    src = src.replace(f'os.path.join({anchor}, f"out/', 'os.path.join(OUTDIR, f"')
    src = src.replace(f'os.path.join({anchor}, "out")', "OUTDIR")
    src = src.replace(f'f"{{{anchor}}}/out/', 'f"{OUTDIR}/')

    if src != orig:
        m = re.search(rf"^{anchor} = .*$", src, re.M)
        src = src[:m.end() + 1] + DECL.format(a=anchor) + src[m.end() + 1:]

    left = len(re.findall(r'["\{]out/', src))
    if src != orig and not check:
        ast.parse(src)
        open(path, "w").write(src)
    return src != orig, left


def main():
    check = "--check" in sys.argv
    files = sorted(glob.glob(os.path.join(ROOT, "pipeline", "*.py")) +
                   glob.glob(os.path.join(ROOT, "eval", "*.py")))
    bad = 0
    print(f"{'file':38s} {'action':11s} out/ refs left")
    for p in files:
        changed, left = patch(p, check)
        act = "would edit" if (changed and check) else ("edited" if changed else "clean")
        print(f"{os.path.basename(p):38s} {act:11s} {left}")
        bad += left
    if bad:
        print(f"\n{bad} unresolved out/ reference(s)")
    return 1 if (bad and check) else 0


if __name__ == "__main__":
    sys.exit(main())
