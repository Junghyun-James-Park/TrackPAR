#!/bin/bash
# TrackPAR end to end: SAM3 tracking through to a labelled corpus.
#
#   source config/paths.sh
#   bash run_all.sh                    # every stage
#   bash run_all.sh --from fragments   # skip tracking, reuse existing tracks
#   bash run_all.sh --score            # and grade the result afterwards
#
# Six stages. Each writes into $TRACKPAR_OUT and skips itself when its output is
# already there, so an interrupted run resumes instead of restarting.
#
# ---------------------------------------------------------------------------
# The two design decisions this encodes
# ---------------------------------------------------------------------------
#
# 1. IDENTITY IS PER TRACK, MOMENTARY IS PER FRAME.
#    On tracks holding at least one positive, exposed changes between frames
#    71.8% of the time and watched 85.7%. A single value per track for those two
#    is the wrong shape of answer, not merely a coarse one. gender and age do not
#    have that problem, and asking once per track lets the model see K views of
#    the same person.
#
# 2. EXPOSED AND WATCHED USE DIFFERENT PROMPTS AT DIFFERENT K.
#    Measured on all 5,168 annotated instances with bootstrap intervals:
#
#      exposed   combined 0.697 [0.678,0.716]  ~  eyes 0.689  ~  subattr 0.677
#      watched   svfd     0.740 [0.689,0.784]  >  PADQ 0.585  >  eyes 0.508
#
#    exposed is a four-way tie; watched is svfd alone, separated from every other
#    arm. And svfd's watched only works at K=1 — packed into a K=8 call it applies
#    the prompt's stated rarity prior to the whole batch and answers no to
#    everything, reading F1 0.000. See docs/RESULTS.md.
#
# Identity runs through the fine-tuned adapter; momentary runs through the SAME
# base model with no adapter. Fine-tuning for identity destroys momentary
# prompt-following, so the two passes deliberately load different models.
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/config/paths.sh"
mkdir -p "$TRACKPAR_OUT"
export PYTHONPATH="$ROOT/pipeline:$ROOT/eval:$ROOT/src:${PYTHONPATH:-}"

FROM="track"; SCORE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --score) SCORE=1; shift ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

say() { echo "=== [$(date '+%H:%M:%S')] $*"; }
want() {
  local order=(track fragments gender age momentary merge) w=-1 h=-1 i=0
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

# ------------------------------------------------------------ 1. tracking
if want track && [ ! -d "$TRACKPAR_OUT/track_sam3" ]; then
  say "1/6 SAM3 tracking  (needs the $SAM3_ENV environment, not this one)"
  CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/track_sam3_chunked.py"
else
  say "1/6 tracking skipped"
fi

# ----------------------------------------------------------- 2. fragments
if want fragments && [ ! -s "$TRACKPAR_OUT/phase1_fragments.json" ]; then
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
# Two passes, because the two attributes want different prompts and different K.
if want momentary; then
  if ! ls "$TRACKPAR_OUT"/momentary_exposed*.json >/dev/null 2>&1; then
    say "5a/6 exposed — $(basename "$EXPOSED_PROMPT") @ K=$EXPOSED_K, sharded"
    CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/exp20_unified_infer.py" \
        --rep full_mask --K "$EXPOSED_K" --tag momentary_exposed --prompt meta \
        --prompt_file "$EXPOSED_PROMPT" --shard_idx 0 --n_shards 2 &
    P0=$!
    CUDA_VISIBLE_DEVICES="$G1" python -u "$ROOT/pipeline/exp20_unified_infer.py" \
        --rep full_mask --K "$EXPOSED_K" --tag momentary_exposed --prompt meta \
        --prompt_file "$EXPOSED_PROMPT" --shard_idx 1 --n_shards 2 &
    P1=$!
    wait $P0 $P1
  else
    say "5a/6 exposed skipped"
  fi

  if ! ls "$TRACKPAR_OUT"/momentary_watched*.json >/dev/null 2>&1; then
    # K=1 means one call per FRAME, so this is the expensive stage: ~4.4 s per
    # frame, ~18,000 frames over 2,438 tracks, about 11 h across two cards.
    say "5b/6 watched — $(basename "$WATCHED_PROMPT") @ K=$WATCHED_K, per frame"
    CUDA_VISIBLE_DEVICES="$G0" python -u "$ROOT/pipeline/momentary_k1_control.py" \
        --prompt svfd --all-tracks --shard-idx 0 --n-shards 2 \
        --out "$TRACKPAR_OUT/momentary_watched_sh0.json" &
    P0=$!
    CUDA_VISIBLE_DEVICES="$G1" python -u "$ROOT/pipeline/momentary_k1_control.py" \
        --prompt svfd --all-tracks --shard-idx 1 --n-shards 2 \
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
