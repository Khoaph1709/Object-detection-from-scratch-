#!/usr/bin/env bash
set -u -o pipefail

# One-shot local V100 launcher. Run from any directory inside the repository.
# It trains full images and validates with sliced inference via the JSON config.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="${RUN_NAME:-targeted_highres_p2p3_v100}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/checkpoints/$RUN_NAME}"
CONFIG="${CONFIG:-$PROJECT_ROOT/my_submission/configs/train_targeted_highres_p2p3_l40s.json}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-$PROJECT_ROOT/checkpoints/run_a_small_object_chair_hem_l40s/best.pth}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/indoor5-v2-student}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-6}"
# V100 16 GB: try batch 2 first, then batch 1 only if CUDA OOM occurs.
# For a 32 GB V100, use BATCH_CANDIDATES="4 3 2 1".
BATCH_CANDIDATES="${BATCH_CANDIDATES:-2 1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

HF_AUTO_DOWNLOAD="${HF_AUTO_DOWNLOAD:-0}"
HF_DATA_REPO="${HF_DATA_REPO:-}"
HF_MODEL_REPO="${HF_MODEL_REPO:-}"
HF_DATA_REVISION="${HF_DATA_REVISION:-main}"
HF_MODEL_REVISION="${HF_MODEL_REVISION:-main}"
HF_BASELINE_RUN_NAME="${HF_BASELINE_RUN_NAME:-$(basename "$(dirname "$BASELINE_CHECKPOINT")")}"

if [[ "$HF_AUTO_DOWNLOAD" == "1" ]]; then
  if [[ -z "$HF_DATA_REPO" || -z "$HF_MODEL_REPO" ]]; then
    fail "HF_AUTO_DOWNLOAD=1 requires HF_DATA_REPO and HF_MODEL_REPO."
  fi
  if ! "$PYTHON_BIN" -c 'import huggingface_hub' >/dev/null 2>&1; then
    fail "huggingface_hub is missing. Install: python3 -m pip install -U 'huggingface_hub[hf_xet]'"
  fi
  echo "=== Hugging Face artifact sync ==="
  "$PYTHON_BIN" "$PROJECT_ROOT/my_submission/scripts/hf_sync_assets.py" \
    --project-root "$PROJECT_ROOT" \
    --dataset-repo "$HF_DATA_REPO" \
    --model-repo "$HF_MODEL_REPO" \
    --dataset-revision "$HF_DATA_REVISION" \
    --model-revision "$HF_MODEL_REVISION" \
    --model-run-name "$HF_BASELINE_RUN_NAME"
fi

if [[ -f "$DATA_ROOT/public/annotations/train.json" ]]; then
  DATA_ROOT="$DATA_ROOT/public"
fi

TRAIN_JSON="$DATA_ROOT/annotations/train.json"
VAL_JSON="$DATA_ROOT/annotations/val.json"
TRAIN_IMAGES="$DATA_ROOT/train/images"
VAL_IMAGES="$DATA_ROOT/val/images"
LOG_FILE="$RUN_DIR/train_v100.log"

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
fi

echo "=== V100 training setup ==="
echo "project       : $PROJECT_ROOT"
echo "config        : $CONFIG"
echo "data root     : $DATA_ROOT"
echo "run directory : $RUN_DIR"
echo "epochs        : $EPOCHS"
echo "batch trials  : $BATCH_CANDIDATES"
echo "amp           : enabled"
echo ""

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

  # Existing last.pth is authoritative after an interruption. Otherwise load
  # the proven chair-only HEM model and initialize a fresh optimizer.
  if [[ -f "$RUN_DIR/last.pth" ]]; then
    args+=(--auto_resume)
    echo "Resuming from $RUN_DIR/last.pth with batch_size=$batch_size"
  elif [[ -f "$RUN_DIR/best.pth" ]]; then
    args+=(--resume_model_only --resume "$RUN_DIR/best.pth")
    echo "Warm-starting from existing best checkpoint with batch_size=$batch_size"
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
    echo "CUDA OOM with batch_size=$batch_size. Trying the next smaller batch size."
    continue
  fi

  echo "Training failed with exit code $status and the log does not look like a CUDA OOM." >&2
  echo "Inspect $LOG_FILE before retrying; no automatic retry was attempted." >&2
  exit "$status"
done

fail "All V100 batch-size candidates failed because of CUDA OOM. Try BATCH_CANDIDATES=1 and/or reduce resolution in a copied config."
