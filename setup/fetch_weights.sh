#!/bin/bash
# Install the identity LoRA adapter (gender + age). ~1.2 GB, not committed.
#
#   bash setup/fetch_weights.sh /local/path/to/adapter_dir     # from a local copy
#   bash setup/fetch_weights.sh --gdrive <FILE_ID_or_URL>      # from Google Drive
#
# Two files must both arrive:
#
#   adapter_model.safetensors   ~346 MB   the LoRA weights
#   non_lora_state_dict.bin     ~912 MB   the trained vision tower + merger
#
# The second is the trap. This adapter trained the vision path as well, and
# PeftModel.from_pretrained loads only the first file — silently. A copy missing
# non_lora_state_dict.bin evaluates a base vision tower under a fine-tuned
# adapter, which is a model that never existed, and it scores plausibly enough
# that the run looks fine. setup/check_env.py fails if it is absent.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="${IDENTITY_ADAPTER:-$ROOT/weights/identity_lora}"

usage() {
  cat <<'EOF'
usage:
  bash setup/fetch_weights.sh /path/to/adapter_dir
  bash setup/fetch_weights.sh --gdrive <FILE_ID or share URL>

The adapter is not public. Obtain it from the authors as either a directory or a
Google Drive link to a .tar.gz of that directory.
EOF
}

[ $# -ge 1 ] || { usage; exit 1; }
mkdir -p "$DST"

if [ "$1" = "--gdrive" ]; then
  [ $# -ge 2 ] || { usage; exit 1; }
  ID="$2"
  # Accept either a bare file id or any of the share-URL shapes Drive produces.
  case "$ID" in
    *drive.google.com*)
      ID=$(printf '%s' "$ID" | sed -E 's#.*/file/d/([^/]+).*#\1#; s#.*[?&]id=([^&]+).*#\1#')
      ;;
  esac
  echo "Google Drive file id: $ID"

  if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown is not installed. Install it with:"
    echo "    pip install gdown"
    echo
    echo "Or download manually from"
    echo "    https://drive.google.com/file/d/$ID/view"
    echo "then re-run this script pointing at the extracted directory."
    exit 1
  fi

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "downloading (~1.2 GB) ..."
  # Files this size trip Drive's virus-scan interstitial; gdown handles the
  # confirmation token, a plain curl of the share URL returns an HTML page.
  gdown --id "$ID" -O "$TMP/adapter.tar.gz"

  case "$(file -b --mime-type "$TMP/adapter.tar.gz" 2>/dev/null || echo unknown)" in
    text/html)
      echo "Drive returned an HTML page, not the archive. The file is probably"
      echo "still restricted — set sharing to 'Anyone with the link' and retry."
      exit 1 ;;
  esac

  echo "extracting ..."
  tar -xzf "$TMP/adapter.tar.gz" -C "$TMP"
  SRC="$(find "$TMP" -name adapter_model.safetensors -printf '%h\n' | head -1)"
  [ -n "$SRC" ] || { echo "no adapter_model.safetensors inside the archive"; exit 1; }
else
  SRC="$1"
  [ -d "$SRC" ] || { echo "not a directory: $SRC"; exit 1; }
fi

for f in adapter_config.json adapter_model.safetensors non_lora_state_dict.bin \
         config.json processor_config.json tokenizer_config.json tokenizer.json \
         chat_template.jinja; do
  [ -f "$SRC/$f" ] && cp -n "$SRC/$f" "$DST/$f"
done
echo "adapter -> $DST"
du -sh "$DST"

python3 - "$DST" <<'PY'
import hashlib, os, sys
d = sys.argv[1]
ok = True
for f in ("adapter_model.safetensors", "non_lora_state_dict.bin"):
    p = os.path.join(d, f)
    if not os.path.exists(p):
        print(f"MISSING {f}  <-- check_env.py will refuse to run")
        ok = False
        continue
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    print(f"  {f:30s} {os.path.getsize(p)/1e6:8.1f} MB  sha256 {h.hexdigest()[:16]}")
print("\nCompare these hashes against the ones the authors published; a truncated"
      "\ndownload is otherwise indistinguishable from a complete one." if ok else "")
PY
