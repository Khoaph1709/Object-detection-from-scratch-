#!/usr/bin/env bash
set -euo pipefail

# Run the instructor-provided prediction container. Ground truth is intentionally
# never mounted here; hidden evaluation remains outside the submission container.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-object-detection-exam:2026}"
IMAGE_DIR="${1:-}"
OUTPUT_NAME="${2:-test_predictions.json}"
OUTPUT_DIR="${3:-$SUBMISSION_ROOT/grading_outputs}"

if [[ -z "$IMAGE_DIR" ]]; then
  echo "Usage: $0 /absolute/path/to/read-only/images [output.json] [output_dir]" >&2
  exit 2
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "ERROR: image directory not found: $IMAGE_DIR" >&2
  exit 1
fi
if [[ ! -f "$SUBMISSION_ROOT/predict.py" ]]; then
  echo "ERROR: predict.py not found under $SUBMISSION_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

# IMAGE_DIR is mounted read-only; only OUTPUT_DIR is writable by the container.
docker run --rm --gpus all \
  -v "$IMAGE_DIR:/exam/test_images:ro" \
  -v "$SUBMISSION_ROOT:/workspace" \
  -v "$OUTPUT_DIR:/exam/outputs" \
  "$IMAGE_NAME" \
  python predict.py \
    --image_dir /exam/test_images \
    --output "/exam/outputs/$OUTPUT_NAME"

printf 'Predictions written to %s\n' "$OUTPUT_DIR/$OUTPUT_NAME"
