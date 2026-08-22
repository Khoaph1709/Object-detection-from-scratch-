# Custom Object Detector for the Final Assignment

This submission implements a custom **anchor-free FCOS-style object detector** for the final assignment. The implementation uses PyTorch and basic neural-network layers; it does not use a complete detector such as YOLO, Detectron2, torchvision Faster R-CNN, or torchvision SSD.

The class order is taken from the final assignment PDF and is therefore the single source of truth:

```text
bottle, cup, chair, laptop, backpack
```

The detector predicts a bounding box, a class label, and a confidence score for every retained detection. It supports multiple objects per image and images with no detections.

## Environment

From the repository root, install the dependencies with:

```bash
pip install -r my_submission/requirements.txt
```

The stable baseline uses a ConvNeXt-Tiny feature extractor through `timm`, followed by a custom FPN and a custom FCOS head. This branch additionally supports the experimental `ConvNeXt-Small + BiFPN` configuration. Both pyramids produce `P2`–`P7`; the stride-4 `P2` branch is included specifically to preserve detail for small objects. BiFPN uses custom normalized learnable fusion weights and depthwise-separable convolutions. The ConvNeXt-Small experiment is intentionally separate because its higher capacity and bidirectional fusion require more VRAM and must be validated against the stable P2-FPN baseline. If the course requires every parameter to be randomly initialized, pass `--no_pretrained_backbone`. The assignment instructor has allowed pretrained backbones, so the default uses pretrained ConvNeXt weights to improve convergence and accuracy; the detector head, pyramid, target assignment, loss, NMS, and inference pipeline remain custom.

## Dataset Layout

Place the supplied dataset at `public/` in the execution environment:

```text
public/
├── classes.json
├── train/images/
├── val/images/
├── annotations/train.json
├── annotations/val.json
└── tools/evaluate_predictions.py
```

The annotation files must contain the five PDF classes and use pixel-coordinate boxes in the format `[xmin, ymin, xmax, ymax]`.

## Training

The required training command from the PDF is supported directly. Run it from inside `my_submission/`:

```bash
cd my_submission
python train.py \
  --train_data ../public/annotations/train.json \
  --val_data ../public/annotations/val.json \
  --image_dir ../public/train/images \
  --val_image_dir ../public/val/images \
  --checkpoint_dir ./models/
```

Alternatively, run the stable high-resolution Modal configuration prepared for small objects:

```bash
cd my_submission
python train.py --config configs/train_modal_small_objects_l40s.json
```

The experimental ConvNeXt-Small + BiFPN configuration is:

```bash
cd my_submission
python train.py --config configs/train_modal_convnext_small_bifpn_l40s.json
```

For the Modal full-train wrapper, use a distinct run name for this experiment:

```bash
modal run my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name fcos_modal_convnext_small_bifpn_l40s \
  --config-path /root/project/my_submission/configs/train_modal_convnext_small_bifpn_l40s.json \
  --batch-size 2 \
  --amp
```

Training writes the best validation checkpoint to `./models/best.pth` and the latest checkpoint to `./models/last.pth`. It also writes `train_log.csv`, `val_history.jsonl`, validation predictions, and optional TensorBoard logs in the checkpoint directory. The validation metric is the evaluator supplied with the assignment and is reported as `mAP@0.5`.

For a quick local smoke test, use a tiny subset and disable pretrained weights:

```bash
cd my_submission
PYTHONPATH=.. python train.py \
  --train_data ../public/annotations/train.json \
  --val_data ../public/annotations/val.json \
  --image_dir ../public/train/images \
  --val_image_dir ../public/val/images \
  --checkpoint_dir ./models/smoke \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --short_size 256 \
  --max_size 384 \
  --max_steps 2 \
  --overfit_images 4 \
  --no_pretrained_backbone \
  --no_tensorboard
```

## Inference

The mandatory inference interface is:

```bash
cd my_submission
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

If `./models/best.pth` is not present, `predict.py` attempts to download it when a checkpoint URL is supplied:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --checkpoint_url https://your-public-storage.example/best.pth \
  --checkpoint_sha256 <optional_sha256>
```

The URL is intentionally provided by the model owner rather than hard-coded into the submission. The script downloads the checkpoint into `models/downloads/`, optionally verifies the SHA-256 digest, loads it, and uses the saved `class_names` metadata. Do not commit `best.pth`, `last.pth`, or any other model weights to the submission archive.

Horizontal-flip test-time augmentation can be enabled when it improves validation performance:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json \
  --tta_flip \
  --tta_merge_strategy nms
```

The output is a JSON array. Every input image appears exactly once, including images whose `boxes` list is empty. Each box uses the required format:

```json
[
  {
    "image_id": "img_7fd91a4c2e30.jpg",
    "boxes": [
      {
        "class": "chair",
        "confidence": 0.91,
        "bbox": [48.0, 72.0, 210.0, 356.0]
      }
    ]
  }
]
```

Coordinates are clipped and converted back to the original image size. NMS is applied independently for each class.

## Evaluation

Evaluate validation predictions with the official evaluator supplied in `public/tools/`:

```bash
python ../public/tools/evaluate_predictions.py \
  --ground_truth ../public/annotations/val.json \
  --predictions predictions.json \
  --output val_score.json
```

The evaluator checks the JSON schema, valid classes, box coordinates, IoU, precision, recall, and `mAP@0.5`. The hidden test set is evaluated by the course system and is not included in the repository.

## Data-quality audit

The repository includes a read-only annotation audit for the supplied train split. It checks the expected class order, missing or malformed image records, invalid or out-of-bounds boxes, boxes smaller than two pixels, duplicate same-class boxes, suspicious high-IoU boxes from different classes, object-size distributions, and image groups for manual review. It never edits the annotation JSON.

Run it only against the training annotation:

```bash
python scripts/audit_annotations.py \
  --annotations /path/to/indoor5-v2-student/public/annotations/train.json \
  --image-dir /path/to/indoor5-v2-student/public/train/images \
  --output-dir ./audit_train_quality
```

To rank likely chair/backpack false-positive images, optionally add predictions generated from the train split:

```bash
python scripts/audit_annotations.py \
  --annotations /path/to/indoor5-v2-student/public/annotations/train.json \
  --image-dir /path/to/indoor5-v2-student/public/train/images \
  --predictions /path/to/train_predictions.json \
  --output-dir ./audit_train_quality
```

The audit writes `summary.json`, `issues.json`, `boxes.csv`, and `review_manifest.json`. When `--image-dir` is supplied, it also renders review images under `review_images/`. The groups `chair_false_positive_high` and `backpack_false_positive_high` require train predictions; `tiny_objects`, `no_chair_or_backpack`, and `duplicate_or_overlap_suspect` are created from annotations alone. High overlap is a review signal rather than an automatic deletion rule, because legitimate occlusion can produce overlapping boxes. Do not pass hidden-test annotations or hidden-test predictions to this tool.

## Overload and false-positive analysis

After generating predictions on the train split, the overload analyzer ranks the images with the most retained boxes and classifies each prediction using one-to-one matching against the train ground truth. It reports true positives, same-class duplicates, class confusion, and background false positives. It does not modify annotations or predictions.

Run it with train annotation, train images, and predictions generated for the same train images:

```bash
python3 my_submission/scripts/analyze_overload_images.py \
  --annotations /path/to/indoor5-v2-student/public/annotations/train.json \
  --predictions /path/to/train_predictions.json \
  --image-dir /path/to/indoor5-v2-student/public/train/images \
  --output-dir ./overload_analysis \
  --topk 30
```

The output contains `overload_summary.json`, `overload_diagnostics.csv`, `all_prediction_diagnostics.csv`, `overload_predictions.json`, and rendered images under `overload_top30/`. The rendered colors are green for ground truth/true-positive context, orange for duplicate predictions, red for background false positives, and purple for class confusion. The duplicate/background split is a diagnostic heuristic; use it to choose between stricter NMS and hard-negative training, not to edit labels automatically. Use train data only for this analysis and never pass hidden-test labels.

## Submission Checklist

Submit the `my_submission/` directory without model weights. It contains `models/`, `utils/`, `train.py`, `predict.py`, `README.md`, and `requirements.txt`. The public dataset is supplied separately by the course environment. Before creating the archive, remove all `.pth` files and verify that the required train and predict commands work with the public dataset paths.

## Inference-only postprocessing sweep

To compare score thresholds, class-aware NMS thresholds, and per-image detection caps without retraining, run `scripts/sweep_inference_postprocess.py` on a validation prediction JSON. The input predictions must be generated for the same validation split and checkpoint; this script cannot recover boxes removed by the original detector before the JSON was written.

```bash
python3 my_submission/scripts/sweep_inference_postprocess.py \
  --predictions ./val_predictions_sliced.json \
  --ground-truth ./indoor5-v2-student/public/annotations/val.json \
  --evaluator ./indoor5-v2-student/public/tools/evaluate_predictions.py \
  --output-dir ./inference_sweep \
  --thresholds 0.08,0.10,0.12 \
  --nms-thresholds 0.45,0.50,0.55 \
  --limits 30,50,70,100
```

The best configuration and all scores are saved in `inference_sweep/sweep_summary.json`. Select a configuration using validation mAP and per-class recall, not by prediction count alone. Do not use hidden-test annotations for this sweep.

## Targeted high-resolution P2/P3 rewrite

The `targeted_highres` architecture keeps the ConvNeXt + FCOS detector and adds a gated P2/P3 refinement neck plus a zero-initialized small-object residual head. The new branches are initialized close to identity so a baseline checkpoint can be warm-started safely. P1 is mutually exclusive with this branch.

The recommended training configuration is:

```text
configs/train_targeted_highres_p2p3_l40s.json
```

It keeps full-image training, the proven radius-2.0 assignment, small-object sampling/cropping, and sliced validation. It does not enable P1, tile training, DIoU, or quality-aware classification.
