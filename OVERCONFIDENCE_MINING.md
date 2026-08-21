# Overconfidence Sample Mining

`my_submission/scripts/mine_overconfident_samples.py` finds high-confidence predictions that are likely to be harmful to AP. It compares prediction boxes with same-class ground-truth boxes and categorizes errors as:

| Reason | Meaning |
|---|---|
| `background_false_positive` | The predicted class has no ground-truth box in the image, or no same-class overlap exists. |
| `localization_error` | A same-class ground-truth box exists, but the best IoU is below the matching threshold. |
| `duplicate_prediction` | A high-confidence prediction matches a ground-truth box already claimed by a higher-confidence prediction. |
| `true_positive` | A prediction reaches the matching IoU threshold; it is not included in the mined error set. |

The script processes predictions in descending confidence order. For each error it computes an `overconfidence_score`. High-confidence background false positives receive a score close to their confidence; localization errors receive `confidence × (1 - best_same_class_iou)`. Images are ranked by the sum of their top error scores.

## Local command

From the repository root:

```bash
python3 my_submission/scripts/mine_overconfident_samples.py \
  --ground_truth indoor5-v2-student/public/annotations/val.json \
  --predictions path/to/val_predictions.json \
  --image_dir indoor5-v2-student/public/val/images \
  --output_dir overconfidence_val \
  --score_threshold 0.30 \
  --match_iou_threshold 0.50 \
  --max_images 100 \
  --max_errors_per_image 8
```

For training-set mining, replace the annotation and image paths with the train split and use the train prediction JSON.

## Outputs

The command creates:

```text
overconfidence_val/
├── overconfidence_report.json
├── overconfidence_errors.csv
├── summary.json
└── visualizations/
    ├── 0001_<image>.jpg
    ├── 0002_<image>.jpg
    └── ...
```

Green boxes are ground-truth annotations. Red boxes are high-confidence error predictions. Each red label includes class, confidence, same-class IoU, and error reason.

Use `overconfidence_errors.csv` for sorting by class/reason and `visualizations/` for visual inspection. For the current project, pay special attention to chair background false positives and duplicate predictions, and to backpack localization errors with confidence above 0.30.

The tool is diagnostic only. It does not change model weights, labels, or predictions, and it does not access hidden test labels.
