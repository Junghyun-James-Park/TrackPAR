#!/bin/bash
# Label one attribute on your own data.
#
#   bash label_attribute.sh --attr holding_item \
#       --definition "the person is holding a product in their hand" \
#       --images /data/my_frames
#
#   bash label_attribute.sh --attr wearing_helmet \
#       --definition "the person is wearing a safety helmet" \
#       --video site.mp4 --fps 1 --limit 100
#
# Writes <out>.json and <out>.csv, defaulting to $TRACKPAR_OUT/<attr>.*
# The output field is named after the attribute. Nothing is renamed by hand.
#
# Everything here is a thin wrapper over pipeline/label_attribute.py; pass
# --help through for the full list of options.
#
#   bash label_attribute.sh --self-test     # check the parser, no GPU
#
set -eu

ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/config/paths.sh"
export PYTHONPATH="$ROOT/pipeline:$ROOT/eval:$ROOT/src:${PYTHONPATH:-}"
mkdir -p "$TRACKPAR_OUT"

# Both cards are exposed even though the labelling model fits on one. The writer
# that produces rules and prompts is larger and needs the pair; the labelling
# model is pinned to the first card inside the runner so it does not spread and
# slow itself down. With a single card visible, loading the writer beside a
# resident labelling model offloads it to CPU and the run stalls without erroring.

# --self-test needs no GPU and no model, so it skips the pre-flight below.
for arg in "$@"; do
  if [ "$arg" = "--self-test" ]; then
    exec python -u "$ROOT/pipeline/label_attribute.py" "$@"
  fi
done

# Only the model cache has to be right for this entry point. The corpus paths
# that check_env.py insists on belong to the Lotte pipeline, and this script
# reads your data instead, so a full pre-flight would fail for the wrong reason.
python - <<'PY'
import os, sys
h = os.environ.get("HF_HOME", "")
if not h or not os.access(h, os.W_OK):
    sys.exit(f"HF_HOME is not writable: {h!r}. Fix it in config/paths.sh.")
print(f"  HF_HOME {h}")
PY

echo "=== [$(date '+%H:%M:%S')] labelling on GPU(s) $TRACKPAR_GPUS"
CUDA_VISIBLE_DEVICES="$TRACKPAR_GPUS" python -u "$ROOT/pipeline/label_attribute.py" "$@"
echo "=== [$(date '+%H:%M:%S')] done"
