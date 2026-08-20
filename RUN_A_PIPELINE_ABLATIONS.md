# Run A pipeline ablations

This runbook evaluates improvements against the validated Run A baseline without changing more than one major factor at a time. The baseline is ConvNeXt-Small + BiFPN, trained at 704/1056, with `best.pth` from epoch 8 and validation mAP@0.5 approximately 0.659358.

## Experiment policy

Use a new Modal `run_name` for every ablation. Start each ablation from the same baseline checkpoint with `--resume-model-only`, so optimizer and scheduler are reset while the model weights remain comparable. Run experiments sequentially, not concurrently, because each experiment uses an L40S and a separate checkpoint directory.

Each ablation is intentionally short: four epochs, validation after every epoch, and early stopping after one validation interval without improvement. The fast-search configs use `batch_size=8` and `warmup_steps=300` at 704/1056 to reduce wall-clock time. If L40S memory exceeds the safe limit or an OOM occurs, change only `batch_size` from 8 to 6, then 4; keep the same config and run name only after the failed job is stopped. The experiment is retained only if total validation mAP improves over 0.659358 and the chair/backpack regression is acceptable.

## 0. Baseline checkpoint

The baseline checkpoint must exist at:

```text
/data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth
```

Do not overwrite this directory. It is the rollback checkpoint for every experiment.

## 1. Assignment ablation

This changes only `center_sampling_radius` from 1.5 to 2.0. The fast-search config uses batch 8; this is intended for checkpoint discovery rather than a strict one-variable scientific ablation. Run:

```bash
modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_a_assignment_radius2_ablation_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_assignment_radius2_ablation_l40s.json \
  --resume-model-only \
  --resume-checkpoint-path /data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth \
  --amp
```

Validate after the run:

```bash
modal run --detach my_submission/modal_app.py \
  --action predict_val \
  --gpu L40S \
  --run-name run_a_assignment_radius2_ablation_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_assignment_radius2_ablation_l40s.json \
  --checkpoint-name best.pth \
  --output-name val_predictions.json
```

## 2. Quality-aware classification ablation

This enables Quality Focal-style classification targets. For each positive location, the target is a detached combination of predicted IoU and FCOS centerness. Negative locations retain target zero. It changes classification score calibration while keeping the detector architecture unchanged. The fast-search config uses batch 8 and warmup 300.

Run:

```bash
modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_a_quality_focal_ablation_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_quality_focal_ablation_l40s.json \
  --resume-model-only \
  --resume-checkpoint-path /data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth \
  --amp
```

Validate:

```bash
modal run --detach my_submission/modal_app.py \
  --action predict_val \
  --gpu L40S \
  --run-name run_a_quality_focal_ablation_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_quality_focal_ablation_l40s.json \
  --checkpoint-name best.pth \
  --output-name val_predictions.json
```

## 3. Mine hard examples from the baseline

Run mining on the train set using the original baseline checkpoint. Do not mine from hidden test images.

```bash
modal run --detach my_submission/modal_app.py \
  --action predict \
  --gpu L40S \
  --run-name run_a_convnext_small_bifpn_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_convnext_small_bifpn_l40s.json \
  --remote-image-dir /data/indoor5-v2-student/public/train/images \
  --checkpoint-name best.pth \
  --output-name train_predictions.json
```

Then mine both chair and backpack:

```bash
modal run --detach my_submission/modal_app.py \
  --action mine_hard_examples \
  --run-name run_a_convnext_small_bifpn_l40s \
  --predictions-name train_predictions.json \
  --mining-output-name combined_hard_example_weights.json \
  --mining-classes chair,backpack \
  --mining-class-score-thresholds chair=0.20,backpack=0.20
```

Verify the file before starting the combined run:

```bash
modal volume ls xla-fcos-volume \
  /checkpoints/run_a_convnext_small_bifpn_l40s
```

The expected file is:

```text
combined_hard_example_weights.json
```

Inspect mined examples before training. Any image containing a real chair/backpack that is missing from the annotations must be removed from the hard-negative set.

## 4. Combined quality-aware + mined-sampler ablation

This combines the quality-aware loss, radius 2.0 assignment, light chair/backpack class weights, and the mined image sampler. It should be run only after the two simpler ablations have been evaluated and the mined annotations have been checked.

```bash
modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_a_quality_mined_combined_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_quality_mined_combined_l40s.json \
  --resume-model-only \
  --resume-checkpoint-path /data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth \
  --amp
```

Validate:

```bash
modal run --detach my_submission/modal_app.py \
  --action predict_val \
  --gpu L40S \
  --run-name run_a_quality_mined_combined_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_quality_mined_combined_l40s.json \
  --checkpoint-name best.pth \
  --output-name val_predictions.json
```

## Selection criteria

Select a new checkpoint only if all of the following are true:

```text
mAP@0.5 > 0.659358
chair AP >= 0.509
backpack AP >= 0.483
chair/backpack recall does not decrease by more than 0.03
```

If a run improves chair AP but lowers total mAP, keep the original baseline. If no ablation exceeds the baseline after four epochs, stop the experiment suite and use the original Run A checkpoint. Do not extend a failed ablation to 20 epochs without a validation signal.

## Detached execution

To survive closing the local terminal, launch the Modal command with `nohup` and `disown`:

```bash
nohup modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_a_quality_focal_ablation_l40s \
  --config-path /root/project/my_submission/configs/train_run_a_quality_focal_ablation_l40s.json \
  --resume-model-only \
  --resume-checkpoint-path /data/checkpoints/run_a_convnext_small_bifpn_l40s/best.pth \
  --amp > quality_focal_modal.log 2>&1 &

echo $! > quality_focal_modal.pid
disown
```

After the log prints `The detached App will keep running`, do not send Ctrl+C to the original `modal run` process. It is safe to stop a separate `tail -f quality_focal_modal.log` viewer.
