# Modal Training Guide

This project can train on Modal with persistent checkpoints, automatic resume, downloadable outputs, and TensorBoard logs.

## 1. Install and Authenticate

Run this locally, not inside Colab:

```bash
pip install modal
modal setup
```

The Modal app is defined in:

```text
my_submission/modal_app.py
```

It uses one persistent Modal Volume:

```text
xla-fcos-volume
```

## 2. Upload Dataset Once

From the repository root:

```bash
modal run my_submission/modal_app.py --action upload --local-data-dir indoor5-v2-student
```

Check that the data is on the Volume:

```bash
modal volume ls xla-fcos-volume /indoor5-v2-student/public/annotations
```

Expected files:

```text
train.json
val.json
```

## 3. Train on Modal

Check the selected GPU spec:

```bash
modal run my_submission/modal_app.py --action gpu_status --gpu L40S
```

Smoke or baseline run:

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name fcos_modal_l40s \
  --epochs 20 \
  --batch-size 8 \
  --val-interval 2 \
  --score-threshold 0.12 \
  --pre-nms-topk 1000
```

Recommended small-object full retraining config (L40S):

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name fcos_modal_small_objects_l40s \
  --config-path /root/project/my_submission/configs/train_modal_small_objects_l40s.json
```

This recipe uses the stride-4 P2 feature level, 800/1280 multi-scale training, AMP, 50 epochs, batch size 4, and a larger pre-NMS candidate pool. Batch size 4 is the safe default for the P2 high-resolution model on a 48 GB L40S. If the GPU still runs out of memory, reduce `batch_size` to 2 before reducing image resolution; keep the P2 level because it is the main architectural improvement for small objects.

The training configuration files are:

```text
my_submission/configs/train_modal_small_objects_l40s.json
my_submission/configs/train_modal_l40s.json
```

Edit the small-object config for persistent hyperparameter changes. CLI arguments override the config for quick experiments.

Experimental ConvNeXt-Small + BiFPN run:

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name fcos_modal_convnext_small_bifpn_l40s \
  --config-path /root/project/my_submission/configs/train_modal_convnext_small_bifpn_l40s.json \
  --batch-size 2 \
  --amp
```

This experiment uses one BiFPN layer, `short_size=704`, `max_size=1056`, and batch size 2. It is intentionally separate from the stable P2-FPN run. Compare validation mAP and per-class recall before choosing it for the final submission; do not resume it from a checkpoint created by the Tiny+FPN architecture.

## 4. Resume Training

Training always passes `--auto_resume`, so rerun the same command:

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name fcos_modal_l40s
```

Checkpoints are stored at:

```text
/data/checkpoints/fcos_modal_l40s
```

inside the Modal Volume.

To upload a local Colab checkpoint folder before resuming:

```bash
modal volume put --force xla-fcos-volume my_submission/models/baseline_fcos_smoke_2ep /checkpoints/fcos_modal_l40s
modal volume ls xla-fcos-volume /checkpoints/fcos_modal_l40s
```

The most important file is:

```text
last.pth
```

Training logs now include GPU memory columns:

```text
gpu_allocated_mb
gpu_reserved_mb
gpu_max_allocated_mb
```

These are also written to TensorBoard under `gpu/*`.

## 5. TensorBoard

Start TensorBoard:

```bash
modal serve my_submission/modal_app.py
```

Modal will print a public URL for the TensorBoard web server. The server reads:

```text
/data/checkpoints/<active_run>/tensorboard
```

`@modal.web_server` functions cannot accept runtime parameters. This project stores an active run name on the Modal Volume and automatically updates it whenever you run actions with `--run-name` (train, predict, predict_val, pack, mine_hard_examples). TensorBoard reads that active run and points to:

```text
/data/checkpoints/<active_run>/tensorboard
```

If TensorBoard is already running, restart `modal serve my_submission/modal_app.py` to switch to the newly active run.

Refresh the page after new training logs are committed.

## 6. Download Outputs

Download individual files:

```bash
modal volume get xla-fcos-volume /checkpoints/fcos_modal_l40s/best.pth ./modal_outputs/fcos_modal_l40s/best.pth
modal volume get xla-fcos-volume /checkpoints/fcos_modal_l40s/last.pth ./modal_outputs/fcos_modal_l40s/last.pth
modal volume get xla-fcos-volume /checkpoints/fcos_modal_l40s/train_log.csv ./modal_outputs/fcos_modal_l40s/train_log.csv
modal volume get xla-fcos-volume /checkpoints/fcos_modal_l40s/val_history.jsonl ./modal_outputs/fcos_modal_l40s/val_history.jsonl
modal volume get xla-fcos-volume /checkpoints/fcos_modal_l40s/val_predictions.score.json ./modal_outputs/fcos_modal_l40s/val_predictions.score.json
```

Or pack everything into one zip on Modal:

```bash
modal run my_submission/modal_app.py --action pack --run-name fcos_modal_l40s
modal volume get xla-fcos-volume /exports/fcos_modal_l40s_outputs.zip .
```

## 7. Suggested GPU Choices

Use `L40S` first. It gives 48 GB VRAM and is a good balance for the free credit.

Use `A100-40GB` or `A100-80GB` only if L40S is unavailable or too slow.

Avoid `H100` for the first run because it will burn the $30 credit quickly.
