#!/usr/bin/env bash
set -u -o pipefail

# One-shot local A100 launcher for targeted high-resolution FCOS training.
# Full-image training + sliced validation. Run inside tmux for SSH disconnects.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="${RUN_NAME:-targeted_highres_p2p3_a100}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/checkpoints/$RUN_NAME}"
CONFIG="${CONFIG:-$PROJECT_ROOT/my_submission/configs/train_targeted_highres_p2p3_a100.json}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$PROJECT_ROOT/checkpoints/run_a_small_object_chair_hem_l40s/best.pth}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/indoor5-v2-student}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-12}"
# Safe starting order for A100 40/80 GB. Override for known hardware:
# BATCH_CANDIDATES="12 10 8 6 4 2" ./my_submission/scripts/train_targeted_highres_a100.sh
BATCH_CANDIDATES="${BATCH_CANDIDATES:-10 8 6 4 2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -f "$DATA_ROOT/public/annotations/train.json" ]]; then
  DATA_ROOT="$DATA_ROOT/public"
fi

TRAIN_JSON="$DATA_ROOT/annotations/train.json"
VAL_JSON="$DATA_ROOT/annotations/val.json"
TRAIN_IMAGES="$DATA_ROOT/train/images"
VAL_IMAGES="$DATA_ROOT/val/images"
LOG_FILE="$RUN_DIR/train_a100.log"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "$CONFIG" ]] || fail "Config not found: $CONFIG"
[[ -f "$TRAIN_JSON" ]] || fail "Train annotation not found: $TRAIN_JSON"
[[ -f "$VAL_JSON" ]] || fail "Validation annotation not found: $VAL_JSON"
[[ -d "$TRAIN_IMAGES" ]] || fail "Train image directory not found: $TRAIN_IMAGES"
[[ -d "$VAL_IMAGES" ]] || fail "Validation image directory not found: $VAL_IMAGES"
mkdir -p "$RUN_DIR"

if [[ ! -f "$RUN_DIR/last.pth" && ! -f "$RUN_DIR/best.pth" ]]; then
  [[ -f "$BASELINE_CHECKPOINT" ]] || fail "Baseline checkpoint not found: $BASELINE_CHECKPOINT"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  fail "Python executable not found: $PYTHON_BIN"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== GPU ==="
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -Eqi 'A100'; then
    echo "WARNING: nvidia-smi did not identify an A100; batch candidates may be inappropriate."
  fi
else
  fail "nvidia-smi not found; refusing to launch GPU training without verification."
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cat <<EOF
=== A100 training setup ===
project       : $PROJECT_ROOT
config        : $CONFIG
data root     : $DATA_ROOT
run directory : $RUN_DIR
epochs        : $EPOCHS
batch trials  : $BATCH_CANDIDATES
amp           : enabled
EOF

for batch_size in $BATCH_CANDIDATES; do
  args=(
    "$PYTHON_BIN" "$PROJECT_ROOT/my_submission/train.py"
    --config "$CONFIG"
    --train_data "$TRAIN_JSON"
    --val_data "$VAL_JSON"
    --image_dir "$TRAIN_IMAGES"
    --val_image_dir "$VAL_IMAGES"
    --checkpoint_dir "$RUN_DIR"
    --device cuda
    --epochs "$EPOCHS"
    --batch_size "$batch_size"
    --num_workers "$NUM_WORKERS"
    --amp
  )

  # Prefer a complete last checkpoint after interruption. If no last exists,
  # warm-start the targeted architecture from the proven baseline.
  if [[ -f "$RUN_DIR/last.pth" ]]; then
    args+=(--auto_resume)
    echo "Resuming from $RUN_DIR/last.pth with batch_size=$batch_size"
  elif [[ -f "$RUN_DIR/best.pth" ]]; then
    args+=(--resume_model_only --resume "$RUN_DIR/best.pth")
    echo "Warm-starting from existing $RUN_DIR/best.pth with batch_size=$batch_size"
  else
    args+=(--resume_model_only --resume "$BASELINE_CHECKPOINT")
    echo "Warm-starting from $BASELINE_CHECKPOINT with batch_size=$batch_size"
  fi

  echo "Command: ${args[*]}"
  set +e
  "${args[@]}" 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "Training completed successfully. Check: $RUN_DIR/best.pth"
    exit 0
  fi

  if grep -Eqi "out of memory|cuda oom|cublas_status_alloc_failed|CUDA error: out of memory" "$LOG_FILE"; then
    echo "CUDA OOM with batch_size=$batch_size; trying the next candidate."
    continue
  fi

  echo "Training failed with exit code $status and does not look like CUDA OOM." >&2
  echo "Inspect $LOG_FILE before retrying; no automatic non-OOM retry was attempted." >&2
  exit "$status"
done

fail "All A100 batch candidates failed with CUDA OOM. Try BATCH_CANDIDATES=1 or lower short_size/max_size in a copied config."
