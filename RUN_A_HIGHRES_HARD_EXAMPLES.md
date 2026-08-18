# Run A: validation tuning and high-resolution hard-example fine-tuning

This runbook uses the epoch-4 Run A checkpoint as the immutable baseline and creates a separate fine-tuning run. The baseline should remain at `/data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth`.

## 1. Generate validation predictions and tune inference

```bash
modal run --detach my_submission/modal_app.py \
  --action predict_val \
  --gpu L40S \
  --run-name run_a_convnext_small_bifpn_l40s \
  --config-path /root/project/my_submission/configs/predict_run_a_highres.json \
  --checkpoint-name best.pth \
  --output-name val_predictions_tta.json
```

Then run the bounded global and class-specific sweep. The class-specific values are applied one class at a time to avoid an exponential grid.

```bash
modal run --detach my_submission/modal_app.py \
  --action tune_val \
  --run-name run_a_convnext_small_bifpn_l40s \
  --predictions-name val_predictions_tta.json \
  --tune-output-name threshold_tuning.json \
  --tune-thresholds 0.08,0.12,0.16,0.20,0.25,0.30 \
  --tune-limits 10,15,20,30,50 \
  --tune-class-thresholds 'chair=0.10|0.14|0.18|0.22,backpack=0.10|0.14|0.18|0.22' \
  --tune-class-limits 'chair=5|10|15,backpack=5|10|15'
```

The best result is written to:

```text
/checkpoints/run_a_convnext_small_bifpn_l40s/threshold_tuning.json
```

## 2. Mine chair and backpack examples on the training set

The miner must use training predictions, never hidden-test labels.

```bash
modal run --detach my_submission/modal_app.py \
  --action predict \
  --gpu L40S \
  --run-name run_a_convnext_small_bifpn_l40s \
  --config-path /root/project/my_submission/configs/predict_run_a_highres.json \
  --remote-image-dir /data/indoor5-v2-student/public/train/images \
  --checkpoint-name best.pth \
  --output-name train_predictions.json
```

After that completes, mine both classes:

```bash
modal run --detach my_submission/modal_app.py \
  --action mine_hard_examples \
  --run-name run_a_convnext_small_bifpn_l40s \
  --predictions-name train_predictions.json \
  --mining-output-name combined_hard_example_weights.json \
  --mining-classes chair,backpack \
  --mining-class-score-thresholds chair=0.20,backpack=0.20
```

The combined weights are written to:

```text
/checkpoints/run_a_convnext_small_bifpn_l40s/combined_hard_example_weights.json
```

The sampler combines chair and backpack difficulty per image using a capped maximum weight, preventing an image that is hard for both classes from being sampled excessively.

## 3. High-resolution fine-tune from Run A best.pth

The new config uses `short_size=800`, `max_size=1200`, batch size 2, low differential learning rates, class-specific chair/backpack loss weights, and the mined image sampler. The new run writes to a different checkpoint directory.

```bash
modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_a_highres_hard_examples_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_highres_hard_examples_l40s.json \
  --resume-model-only \
  --resume-checkpoint-path /data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth \
  --amp
```

This is intentionally model-only resume: the input resolution, optimizer learning rate, sampler and scheduler are changed, so the old optimizer/scheduler state must not be reused. The baseline checkpoint is not overwritten.

## 4. Compare checkpoints

Use validation mAP, chair AP, backpack AP, recall, and prediction count. Select the fine-tuned checkpoint only if total mAP improves without a severe recall collapse.

```bash
modal volume ls xla-fcos-volume \
  /checkpoints/run_a_highres_hard_examples_l40s
```

The primary baseline remains:

```text
/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth
```

Do not start another train job with the same run name while an existing detached App is active.
