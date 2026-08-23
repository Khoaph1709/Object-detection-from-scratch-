# Indoor5 Object Detection

## Overview

This project implements a custom object detector for the Indoor5 dataset. The detector recognizes five classes:

```text
bottle, cup, chair, laptop, backpack
```

The final pipeline is an **anchor-free FCOS detector with hard-example mining (HEM)**. It is designed for images containing many small objects and follows the course requirement to implement the main detection pipeline independently.

## From-Scratch Requirement

The project does not use a complete detector implementation such as YOLOv5/YOLOv8, Detectron2, MMDetection, Faster R-CNN, or torchvision SSD. PyTorch and standard neural-network operators are used to implement the model, training procedure, losses, decoding, confidence filtering, and non-maximum suppression.

A pretrained feature extractor is used as permitted by the assignment. The detection architecture itself is custom.

## Final Architecture

```text
Input image
    ↓
Resize, normalization and training augmentation
    ↓
ConvNeXt-Small backbone
    ↓
BiFPN feature pyramid
    ↓
P2–P7 feature levels
    ↓
Custom FCOS prediction head
    ├── Classification
    ├── Bounding-box regression
    └── Centerness
    ↓
Decode, confidence filtering and per-class NMS
    ↓
Bounding boxes in original-image coordinates
```

The final model uses a P2-aware feature pyramid, center-based assignment with radius `2.0`, controlled small-object sampling/cropping, and HEM-oriented image sampling. The regression loss is GIoU, classification uses focal loss, and the centerness branch provides objectness quality.

## Data Pipeline

The expected dataset layout is:

```text
public/
├── annotations/
│   ├── train.json
│   └── val.json
├── train/images/
├── val/images/
└── tools/evaluate_predictions.py
```

## Environment Setup

From the `my_submission` directory, install the dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The package requires Python 3.10 or newer, PyTorch, torchvision, `timm`, Pillow, NumPy, and tqdm. A CUDA-enabled PyTorch installation is recommended for training, while CPU execution is also supported for small inference checks.

The instructor-provided grading environment may supply its own pinned dependencies. In that environment, run the commands below from the directory containing `train.py` and `predict.py`.

## Data Pipeline

The data loader supports multiple objects per image and the required bounding-box format:

```text
[xmin, ymin, xmax, ymax]
```

Training includes image resizing, normalization, horizontal flipping, color variation, and controlled crops for small objects. Bounding boxes are transformed together with their corresponding images.

## Training

From the `my_submission` directory, the main HEM configuration is:

```text
configs/train_hem_l40s.json
```

Run training with:

```bash
python train.py \
  --config configs/train_hem_l40s.json \
  --train_data ./public/annotations/train.json \
  --val_data ./public/annotations/val.json \
  --image_dir ./public/train/images \
  --val_image_dir ./public/val/images \
  --checkpoint_dir ./models/hem_run \
  --device cuda \
  --amp
```

Training writes checkpoints to the selected checkpoint directory, including `best.pth` and `last.pth`.

## Inference

The required inference command is:

```bash
python predict.py \
  --image_dir /path/to/images \
  --output predictions.json
```

Inference automatically enables the validated tiled settings for small-object detection. These include a resize short side of `704`, maximum size `1056`, tile size `640`, tile overlap `0.20`, confidence threshold `0.08`, pre-NMS top-k `1600`, NMS threshold `0.55`, and a maximum of `100` detections per image. Test-time flipping, weighted box fusion, and score fusion are disabled by default.

The output is a JSON array. Every image must produce an entry, including images with no detections:

```json
[
  {
    "image_id": "image.jpg",
    "boxes": [
      {
        "class": "chair",
        "confidence": 0.91,
        "bbox": [48, 72, 210, 356]
      }
    ]
  }
]
```

The class must be one of the five dataset classes, confidence must be in `[0, 1]`, and bounding boxes must use original-image pixel coordinates.

## Checkpoint Download

The assignment does not allow model-weight files inside the submitted archive. If `models/best.pth` is absent, `predict.py` automatically downloads the final HEM checkpoint from the public GitHub Release asset:

```text
https://github.com/Khoaph1709/Object-detection-from-scratch-/releases/download/hem-final/best.pth
```

The checkpoint is downloaded atomically and then loaded by the inference program. The release asset is distributed separately from the submission archive [1].

## Submission Package

Submit a compressed archive containing `my_submission/` with the source code and configuration files. Do not include the dataset, hidden annotations, runtime logs, prediction artifacts, or any `.pth`, `.pt`, or `.ckpt` file.

The core submission structure is:

```text
my_submission/
├── models/
├── utils/
├── configs/
├── scripts/
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

The main implementation files are `models/`, `utils/`, `train.py`, and `predict.py`. The repository also contains concise utilities for checkpoint preparation, GitHub Release upload, dataset merging, and Docker-based local verification.

## Evaluation

Public validation can be evaluated with the supplied evaluator:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions predictions.json \
  --output score.json
```

The official hidden evaluation supplies only the hidden image directory to `predict.py`. Hidden annotations remain private and must not be included in the submission or mounted into the inference container.

[1]: https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository "GitHub Releases documentation"
