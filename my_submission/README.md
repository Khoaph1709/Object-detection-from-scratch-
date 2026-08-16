# Custom Object Detection Submission

This submission implements a custom object detector for the five-class final assignment:

- `person`
- `car`
- `dog`
- `cat`
- `chair`

The first checkpoint contains the project structure, dataset reader, resize and horizontal flip augmentation, visualization tools, ConvNeXt-Tiny feature extraction, and an FPN module that produces P3-P7 features.

## Install

```bash
pip install -r requirements.txt
```

## Visualize Augmentations

From the repository root:

```bash
PYTHONPATH=. python my_submission/scripts/visualize_augmented_samples.py \
  --annotation indoor5-v2-student/public/annotations/train.json \
  --image_root indoor5-v2-student/public/train/images \
  --output my_submission/artifacts/augmented_samples/train_aug_grid.jpg \
  --num_images 30
```

## Check Feature Shapes

```bash
PYTHONPATH=. python my_submission/scripts/check_feature_shapes.py \
  --image_size 640 \
  --batch_size 2
```

Expected ConvNeXt-Tiny feature names:

```text
C3
C4
C5
```

Expected FPN feature names:

```text
P3
P4
P5
P6
P7
```

## Visualize FCOS Positive Targets

```bash
PYTHONPATH=. python my_submission/scripts/visualize_positive_targets.py \
  --annotation indoor5-v2-student/public/annotations/train.json \
  --image_root indoor5-v2-student/public/train/images \
  --output my_submission/artifacts/positive_targets/positive_targets.jpg \
  --index 0
```

## Debug One-Epoch Mini Overfit

This command verifies the detector, target assignment, loss, backward pass, checkpoint saving, validation prediction export, and public evaluator integration on a tiny subset:

```bash
PYTHONPATH=. python my_submission/train.py \
  --train_data indoor5-v2-student/public/annotations/train.json \
  --val_data indoor5-v2-student/public/annotations/val.json \
  --image_dir indoor5-v2-student/public/train/images \
  --val_image_dir indoor5-v2-student/public/val/images \
  --checkpoint_dir my_submission/models/debug \
  --epochs 1 \
  --batch_size 2 \
  --num_workers 0 \
  --short_size 256 \
  --max_size 384 \
  --max_steps 2 \
  --overfit_images 8 \
  --no_pretrained_backbone \
  --score_threshold 0.9
```

## Full Baseline Training

Run this on a GPU machine:

```bash
cd my_submission
python train.py --config configs/train_competitive_l40s.json
```

Edit the JSON config to change hyperparameters. CLI arguments still override config values, for example:

```bash
python train.py --config configs/train_competitive_l40s.json --epochs 5 --batch_size 4
```

The default training mode uses full precision for stability. Set `"amp": true` or add `--amp` only after a stable run has been confirmed and you want faster training on a CUDA GPU.

## Modal Training

For Modal GPU training with persistent checkpoints, resume, TensorBoard, and output download, see:

```text
../MODAL_TRAINING.md
```

The best checkpoint is saved to:

```text
models/best.pth
```

The latest checkpoint is saved after every epoch:

```text
models/last.pth
```

Training also writes lightweight monitoring logs inside `checkpoint_dir`:

```text
train_log.csv
val_history.jsonl
val_predictions.score.json
tensorboard/
```

In Colab, monitor the TensorBoard dashboard with:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/XLA/checkpoints/baseline_fcos_stable/tensorboard
```

You can also monitor raw files with:

```bash
!tail -n 20 /content/drive/MyDrive/XLA/checkpoints/baseline_fcos_stable/train_log.csv
!tail -n 5 /content/drive/MyDrive/XLA/checkpoints/baseline_fcos_stable/val_history.jsonl
!cat /content/drive/MyDrive/XLA/checkpoints/baseline_fcos_stable/val_predictions.score.json
```

If training is interrupted, run the same command with:

```bash
--auto_resume
```

or explicitly resume from a checkpoint:

```bash
--resume ./models/last.pth
```

## Prediction

From inside `my_submission/`:

```bash
python predict.py \
  --config configs/predict_val.json \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint_url https://your-storage.example.com/best.pth \
  --checkpoint_sha256 <optional_sha256>
```

Do not commit `.pth` files to GitHub. Keep `checkpoint` as a local target path (for example `models/best.pth`) and provide `checkpoint_url` so the script downloads weights automatically when the local file is missing.
