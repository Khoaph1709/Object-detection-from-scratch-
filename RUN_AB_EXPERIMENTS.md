# Controlled Run A / Run B

Hai run này được thiết kế để so sánh công bằng trên cùng dataset, seed, resolution, augmentation, optimizer family và validation evaluator.

## Run A: ConvNeXt-Small + BiFPN

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_a_convnext_small_bifpn_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_convnext_small_bifpn_l40s.json \
  --epochs 20 \
  --batch-size 4 \
  --amp
```

## Run B: ConvNeXt-Tiny + P2-P7 FPN

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_b_convnext_tiny_p2fpn_l40s \
  --config-path /root/project/my_submission/configs/train_run_b_convnext_tiny_p2fpn_l40s.json \
  --epochs 20 \
  --batch-size 4 \
  --amp
```

Both configs use `scheduler_epochs=20`, `val_interval=2`, `early_stopping_patience=3`, `early_stopping_min_delta=0.001`, `seed=42`, one frozen-backbone epoch, differential learning rates (`1e-5` for backbone and `6e-5` for head), AMP, and `auto_resume=false`. Use unique run names and do not mix checkpoints between the two runs.

## Selection rule

Select the checkpoint with the highest validation `mAP@0.5`, not the final epoch. Also record per-class AP, recall, prediction count and the epoch at which the best checkpoint was saved. If one run stops early, that is evidence of validation plateau under this controlled schedule.

## Artifacts to download

```bash
modal volume get xla-fcos-volume /checkpoints/run_a_convnext_small_bifpn_l40s/val_history.jsonl ./run_a_val_history.jsonl
modal volume get xla-fcos-volume /checkpoints/run_b_convnext_tiny_p2fpn_l40s/val_history.jsonl ./run_b_val_history.jsonl
```

Do not compare only training loss. The primary decision metric is validation mAP under the same evaluator and prediction configuration.
