#!/bin/bash
# TrackPAR end to end: SAM3 tracking through to a labelled corpus.
#
#   source config/paths.sh
#   bash run_all.sh                    # every stage
#   bash run_all.sh --from fragments   # skip tracking, reuse existing tracks
#   bash run_all.sh --score            # and grade the result afterwards
#
# Seven stages. Each writes into $TRACKPAR_OUT and skips itself when its output
# is already there, so an interrupted run resumes instead of restarting.
#
# ---------------------------------------------------------------------------
# The shape of the pipeline
# ---------------------------------------------------------------------------
#
#   0  route          per attribute: identity or momentary?   (cached)
#      |
#      +-- identity ------------------------------+
#      |     1  SAM 3 tracking                    |  the track is what lets K
#      |     2  fragments                         |  frames of ONE person go
#      |     3  gender   K=4 frames, one call     |  into a single call
#      |     4  age      K=4 frames, one call     |
#      |                                          |
#      +-- momentary -----------------------------+
#            5  one call per FRAME (K=1)          |  no tracking: the answer is
#               boxes come straight from the      |  per frame, so identity
#               annotation file                   |  across frames is not needed
#
#   6  merge          one label file
#
# Stage 0 decides which branch an attribute takes. Routing is cached in
# out/attr_routing.json, so an attribute is sent to the model once and never
# again.
#
# Why momentary skips tracking: every prompt measured either prefers K=1 or is
# within noise of K=8 on exposed (5 of 7 prefer K=1; the two that do not are
# -0.014 and -0.019, far inside what 426 frames resolve: a bootstrap puts one
# arm's 95% interval at about +/-0.05), and
# on watched K=1 wins outright (svfd 0.667 against 0.000). Since momentary needs
# a per-frame answer anyway, the tracking stage buys nothing for it.
#
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/config/paths.sh"
mkdir -p "$TRACKPAR_OUT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/eval:$ROOT/src:${PYTHONPATH:-}"

FROM="route"; SCORE=0; ATTRS="exposed watched gender age"
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --score) SCORE=1; shift ;;
    --attrs) ATTRS="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

say() { echo "=== [$(date '+%H:%M:%S')] $*"; }
want() {
  local order=(route track fragments gender age momentary merge) w=-1 h=-1 i=0
  for s in "${order[@]}"; do
    [ "$s" = "$FROM" ] && w=$i
    [ "$s" = "$1" ] && h=$i
    i=$((i + 1))
  done
  [ "$h" -ge "$w" ]
}
IFS=',' read -r G0 G1 <<< "$TRACKPAR_GPUS"; G1="${G1:-$G0}"

say "pre-flight"
python -u "$ROOT/setup/check_env.py" || { say "check_env failed — stopping"; exit 1; }

# --------------------------------------------------------------- 0. route
# Decides, per attribute, whether the identity branch or the momentary branch
# runs. Cached in out/attr_routing.json, so an attribute is sent to the model
# once and never again.
if want route; then
  say "0/6 routing attributes: $ATTRS"
  CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/route_attributes.py" \
      --attrs $ATTRS || { say "routing failed — stopping"; exit 1; }
fi

# Tracking exists to put K frames of ONE person in a single call, which only the
# identity branch uses. If nothing routed to identity, stages 1-4 are skipped.
NEED_IDENTITY=$(python -c '
import json, sys
r = json.load(open(sys.argv[1]))
print("1" if any(r.get(a, {}).get("kind") == "identity" for a in sys.argv[2:]) else "0")
' "$TRACKPAR_OUT/attr_routing.json" $ATTRS)
say "identity branch needed: $NEED_IDENTITY"

# ------------------------------------------------------------ 1. tracking
if want track && [ "$NEED_IDENTITY" = "1" ] && [ ! -d "$TRACKPAR_OUT/track_sam3" ]; then
  say "1/6 SAM3 tracking  (needs the $SAM3_ENV environment, not this one)"
  CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/track_sam3_chunked.py"
else
  say "1/6 tracking skipped"
fi

# ----------------------------------------------------------- 2. fragments
if want fragments && [ "$NEED_IDENTITY" = "1" ] && [ ! -s "$TRACKPAR_OUT/phase1_fragments.json" ]; then
  say "2/6 tracks -> fragments"
  python -u "$ROOT/pipeline/phase1_build_all_fragments.py" \
      --out "$TRACKPAR_OUT/phase1_fragments.json"
else
  say "2/6 fragments skipped"
fi

# -------------------------------------------------------------- 3. gender
if want gender && [ ! -s "$TRACKPAR_OUT/identity.json" ]; then
  say "3/6 gender — K=4 multi-image, identity adapter"
  CUDA_VISIBLE_DEVICES="$TRACKPAR_GPUS" python -u "$ROOT/pipeline/multiimg_eval.py" \
      --arm run --adapter "$IDENTITY_ADAPTER" --model-id "$BASE_MODEL" \
      --frames 4 --holdout "" --out "$TRACKPAR_OUT/identity.json"
else
  say "3/6 gender skipped"
fi

# ----------------------------------------------------------------- 4. age
# A separate pass because the identity prompt asks for age_group (young/adult/
# old) and never an integer. 96.4% of this corpus is "adult", so that scale
# cannot discriminate: always answering "adult" scores 0.9645 while the best
# model scores 0.9320. The integer prompt reaches MAE 3.63 on held-out tracks,
# against a best-constant baseline of 10.46.
if want age && [ ! -s "$TRACKPAR_OUT/age.json" ]; then
  say "4/6 age — integer prompt, K=4 multi-image"
  CUDA_VISIBLE_DEVICES="$TRACKPAR_GPUS" python -u "$ROOT/pipeline/age_eval.py" \
      --arm run --adapter "$IDENTITY_ADAPTER" --model-id "$BASE_MODEL" \
      --frames 4 --holdout "" --out "$TRACKPAR_OUT/age.json"
else
  say "4/6 age skipped"
fi

# ----------------------------------------------------------- 5. momentary
# One call per FRAME, K=1. The prompt comes from config/prompt_registry.json when
# the attribute has one written for it; otherwise pipeline/make_prompt.py writes
# one from exemplars first.
#
# The prompt and the parser are chosen together. STYLE selects the parser: for
# `meta` the prompt text comes from a file, for a built-in style (svfd, plain,
# trueonly, padq) the runner builds it and reads its own schema back. Passing a
# prompt file to the wrong style still produces valid JSON, so nothing downstream
# would notice — hence the pairing lives here rather than being inferred.
prompt_args() {   # $1 style, $2 prompt file -> sets PROMPT_ARGS
  if [ "$1" = "meta" ]; then PROMPT_ARGS=(--prompt meta --prompt_file "$2")
  else PROMPT_ARGS=(--prompt "$1"); fi
}

if want momentary; then
  if ! ls "$TRACKPAR_OUT"/momentary_exposed*.json >/dev/null 2>&1; then
    prompt_args "$EXPOSED_STYLE" "$EXPOSED_PROMPT"; EX_ARGS=("${PROMPT_ARGS[@]}")
    say "5a/6 exposed — $(basename "$EXPOSED_PROMPT") / $EXPOSED_STYLE @ K=$EXPOSED_K, per frame"
    CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/momentary_k1_control.py" \
        "${EX_ARGS[@]}" --all-tracks --shard-idx 0 --n-shards 2 \
        --out "$TRACKPAR_OUT/momentary_exposed_sh0.json" &
    P0=$!
    CUDA_VISIBLE_DEVICES="$G1" python -u "$ROOT/pipeline/momentary_k1_control.py" \
        "${EX_ARGS[@]}" --all-tracks --shard-idx 1 --n-shards 2 \
        --out "$TRACKPAR_OUT/momentary_exposed_sh1.json" &
    P1=$!
    wait $P0 $P1
  else
    say "5a/6 exposed skipped"
  fi

  if ! ls "$TRACKPAR_OUT"/momentary_watched*.json >/dev/null 2>&1; then
    # ~4.4 s per frame, ~9,500 frames, about 6 h across two cards per attribute.
    prompt_args "$WATCHED_STYLE" "$WATCHED_PROMPT"; WA_ARGS=("${PROMPT_ARGS[@]}")
    say "5b/6 watched — $(basename "$WATCHED_PROMPT") / $WATCHED_STYLE @ K=$WATCHED_K, per frame"
    CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/momentary_k1_control.py" \
        "${WA_ARGS[@]}" --all-tracks --shard-idx 0 --n-shards 2 \
        --out "$TRACKPAR_OUT/momentary_watched_sh0.json" &
    P0=$!
    CUDA_VISIBLE_DEVICES="$G1" python -u "$ROOT/pipeline/momentary_k1_control.py" \
        "${WA_ARGS[@]}" --all-tracks --shard-idx 1 --n-shards 2 \
        --out "$TRACKPAR_OUT/momentary_watched_sh1.json" &
    P1=$!
    wait $P0 $P1
  else
    say "5b/6 watched skipped"
  fi
fi

# --------------------------------------------------------------- 6. merge
say "6/6 merging into one label file"
python -u "$ROOT/pipeline/merge_labels.py"

if [ "$SCORE" = "1" ]; then
  say "scoring — restricted to the annotated sessions, see docs/RESULTS.md"
  python -u "$ROOT/eval/full_grid.py" || true
  python -u "$ROOT/eval/momentary_deploy_grid.py" || true
fi

say "done — labels in $TRACKPAR_OUT/labels.json"
