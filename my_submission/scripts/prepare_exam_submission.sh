#!/usr/bin/env bash
set -euo pipefail

# Copy a selected checkpoint into the exact path used by predict.py during exam grading.
# The checkpoint remains ignored by Git; this script only prepares the local submission.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_CHECKPOINT="${1:-}"
DESTINATION="${SUBMISSION_ROOT}/models/best.pth"

if [[ -z "$SOURCE_CHECKPOINT" ]]; then
  echo "Usage: $0 /path/to/final_checkpoint.pth" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $SOURCE_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$(dirname "$DESTINATION")"
tmp="${DESTINATION}.tmp"
cp --reflink=auto -- "$SOURCE_CHECKPOINT" "$tmp"
mv -f -- "$tmp" "$DESTINATION"
printf 'Prepared grading checkpoint: %s\n' "$DESTINATION"
printf 'Source: %s\n' "$(readlink -f "$SOURCE_CHECKPOINT")"
printf 'Size: %s bytes\n' "$(stat -c '%s' "$DESTINATION")"
