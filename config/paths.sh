# Every absolute path TrackPAR needs, in one file.
#
#   source config/paths.sh
#
# The scripts were written inside a research tree with these baked in.
# setup/patch_paths.py rewrites each literal to read the environment variable
# below, keeping the original value as the fallback, so on a new machine you edit
# this file and nothing else.
#
# Anything marked REQUIRED must point somewhere real before a run starts;
# setup/check_env.py verifies them and refuses to continue otherwise.

# --- where this checkout lives ------------------------------------------
export TRACKPAR_ROOT="${TRACKPAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TRACKPAR_OUT="${TRACKPAR_OUT:-$TRACKPAR_ROOT/out}"

# --- the corpus (REQUIRED) ----------------------------------------------
# Frames, one directory per recording session. Not redistributable with this
# repo; see README "Data".
export LOTTE_IMAGES="${LOTTE_IMAGES:-/path/to/lotte_cheonho/images}"
# Per-frame person instances used to build tracks.
export LOTTE_ANNOT="${LOTTE_ANNOT:-/path/to/lotte_cheonho/annotations/lotte_tta_sft.json}"
# Ground truth. Only needed to SCORE; a labelling run does not read it.
export LOTTE_CSV="${LOTTE_CSV:-/path/to/lotte_cheonho/lotte_tta.csv}"

# --- model cache (REQUIRED) ----------------------------------------------
# NOT ${HF_HOME:-...}. A shell profile that exports HF_HOME to a directory this
# account cannot write makes every model load fail with a PermissionError that
# reads like a HuggingFace outage. If the inherited value is not writable,
# replace it.
_hf_default="$HOME/.cache/huggingface"
if [ -z "${HF_HOME:-}" ] || ! mkdir -p "${HF_HOME}" 2>/dev/null || [ ! -w "${HF_HOME}" ]; then
  [ -n "${HF_HOME:-}" ] && echo "paths.sh: HF_HOME=${HF_HOME} is not writable — using ${_hf_default}" >&2
  export HF_HOME="$_hf_default"
fi

# --- models ---------------------------------------------------------------
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-9B}"
# Identity LoRA adapter (gender + age). ~1.2 GB, fetched rather than committed;
# see setup/fetch_weights.sh. Leave empty to run identity on the base model.
export IDENTITY_ADAPTER="${IDENTITY_ADAPTER:-$TRACKPAR_ROOT/weights/identity_lora}"

# --- SAM3, stage 1 only ---------------------------------------------------
# The one dependency this repo cannot carry: `sam3.model_builder` is not on PyPI.
# See README "Stage 1 needs SAM3". Stages 2-6 do not import it, so if you already
# have tracks you can ignore all of this and start at stage 2.
export SAM3_ENV="${SAM3_ENV:-sam3}"
export SAM3_SRC="${SAM3_SRC:-/path/to/sam3}"

# --- which momentary setting ships ---------------------------------------
# Both matter, and K is easy to overlook: on the same frames the same prompt
# moves by up to 0.667 watched F1 between K=8 and K=1. See docs/RESULTS.md.
#
# exposed uses `eyes`, not `combined`. combined leads at K=1 (0.674 vs 0.714)
# but collapses at K=8 (0.540 vs 0.728), and K=8 is what this stage runs.
# Picking by the K=1 table would have shipped the weaker arm.
export EXPOSED_PROMPT="${EXPOSED_PROMPT:-$TRACKPAR_ROOT/prompts/crop/eyes.txt}"
export EXPOSED_K="${EXPOSED_K:-8}"
export WATCHED_PROMPT="${WATCHED_PROMPT:-$TRACKPAR_ROOT/prompts/crop/svfd.txt}"
export WATCHED_K="${WATCHED_K:-1}"

# --- GPUs -----------------------------------------------------------------
# Two cards is the assumption throughout; the momentary stage shards across them.
export TRACKPAR_GPUS="${TRACKPAR_GPUS:-0,1}"
