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
It runs for 15 epochs with a 15-epoch cosine schedule. Early stopping has patience 5 but is blocked until epoch 10, because the newly added high-resolution branches are identity/zero initialized and need several epochs to adapt after warm-start. It keeps full-image training, the proven radius-2.0 assignment, small-object sampling/cropping, and sliced validation. It does not enable P1, tile training, DIoU, or quality-aware classification.

## Durable Modal training when the local machine is offline

The training wrapper uses a detached-compatible spawned function, `modal.Retries`, single-use retry containers, and checkpoint auto-resume. Training checkpoints are written atomically as a temporary file followed by a replace, which avoids leaving a partially written `last.pth` after interruption.

For long training, start the command with `modal run --detach`. The command can remain attached while the machine is online; if the terminal or local machine disappears after the detached app has been submitted, Modal can keep the remote function alive and retry container failures. The training loop writes `last.pth` and `best.pth` to the mounted Volume, and a retry continues from `last.pth` when it exists.

Always use a new `run_name` for a new experiment. Do not pass an old warm-start checkpoint when intentionally resuming an existing run directory; the wrapper detects the existing `last.pth` and switches to `--auto_resume`.

## One-shot local V100 launcher

Use `scripts/train_targeted_highres_v100.sh` for a local server with a V100. The launcher uses the canonical dataset root `indoor5-v2-student/public/`, stores outputs under `checkpoints/<run_name>`, enables AMP, tries batch sizes `2` then `1` by default, and resumes from `last.pth` after interruption.

Example:

```bash
cd /path/to/Object-detection-from-scratch-
chmod +x my_submission/scripts/train_targeted_highres_v100.sh
RUN_NAME=targeted_highres_p2p3_v100 \
DATA_ROOT="$PWD/indoor5-v2-student/public" \
BASELINE_CHECKPOINT="$PWD/checkpoints/run_a_small_object_chair_hem_l40s/best.pth" \
my_submission/scripts/train_targeted_highres_v100.sh
```

For a 32 GB V100, try `BATCH_CANDIDATES="4 3 2 1"`; for a 16 GB V100, keep the default `"2 1"`. Run it inside `tmux` if the SSH connection may close. On a later restart, reuse the same `RUN_NAME`; the script will prefer `checkpoints/<run_name>/last.pth` and continue the optimizer/scheduler state.

## One-shot local A100 launcher

Use `scripts/train_targeted_highres_a100.sh` on a local A100 server. The launcher detects the A100 with `nvidia-smi`, uses the canonical dataset root `indoor5-v2-student/public/`, writes to `checkpoints/<run_name>`, enables AMP, and tries batch sizes `10, 8, 6, 4, 2` until one fits. Override `BATCH_CANDIDATES` for a known 40 GB or 80 GB card.

Example:

```bash
cd /path/to/Object-detection-from-scratch-
chmod +x my_submission/scripts/train_targeted_highres_a100.sh
RUN_NAME=targeted_highres_p2p3_a100 \
DATA_ROOT="$PWD/indoor5-v2-student/public" \
BASELINE_CHECKPOINT="$PWD/checkpoints/run_a_small_object_chair_hem_l40s/best.pth" \
BATCH_CANDIDATES="10 8 6 4 2" \
my_submission/scripts/train_targeted_highres_a100.sh
```

Run it in `tmux` when SSH may disconnect. Reusing the same `RUN_NAME` makes the launcher prefer `checkpoints/<run_name>/last.pth` and resume the optimizer/scheduler state.

## Hugging Face Hub: upload một lần, tự tải về trước khi train

Workflow này tách thành **hai repository private** để không trộn dữ liệu khóa học với checkpoint: một dataset repository chứa `indoor5-v2-student/public/...` và một model repository chứa `checkpoints/<run_name>/best.pth` hoặc `last.pth`. Hugging Face hỗ trợ upload thư mục lớn theo kiểu resumable; `snapshot_download` giữ nguyên cấu trúc file dưới `local_dir`, còn `hf_hub_download` tải từng checkpoint theo đúng đường dẫn tương đối [1] [2]. Không commit token vào repository. Trên máy upload hoặc server, đăng nhập bằng `hf auth login`, hoặc truyền token qua secret environment `HF_TOKEN`.

> **Lưu ý quyền dữ liệu:** chỉ upload dataset lên repository private nếu bạn có quyền lưu trữ và đồng bộ bộ dữ liệu của môn học. Không chuyển repository sang public nếu đề bài hoặc giấy phép dữ liệu không cho phép.

### Chuẩn bị xác thực

```bash
cd /path/to/Object-detection-from-scratch-
python3 -m pip install -U "huggingface_hub[hf_xet]"
hf auth login
```

Lệnh `hf auth login` sẽ hỏi token trong terminal và lưu xác thực cục bộ; token không nằm trong code. Nếu server dùng secret manager thay vì file login, có thể export `HF_TOKEN` trong session của server, nhưng không ghi giá trị thật vào shell script, Git, log hoặc lệnh chia sẻ công khai.

### Upload dataset và baseline checkpoint

Trước khi upload, đặt checkpoint đã lấy từ Modal vào đúng vị trí local:

```text
indoor5-v2-student/public/annotations/train.json
indoor5-v2-student/public/annotations/val.json
indoor5-v2-student/public/train/images/...
indoor5-v2-student/public/val/images/...
checkpoints/run_a_small_object_chair_hem_l40s/best.pth
```

Tạo hai repo private trên Hub và upload bằng script trong branch này. Thay `YOUR_HF_USER` bằng namespace Hugging Face của bạn; không thay token vào lệnh.

```bash
cd /path/to/Object-detection-from-scratch-
python3 my_submission/scripts/hf_upload_assets.py \
  --project-root "$PWD" \
  --dataset-repo "YOUR_HF_USER/indoor5-v2-student-private" \
  --dataset-dir "$PWD/indoor5-v2-student/public" \
  --model-repo "YOUR_HF_USER/fcos-indoor5-checkpoints-private" \
  --run-name "run_a_small_object_chair_hem_l40s" \
  --upload-last
```

Script tự tạo repository ở chế độ private nếu repository chưa tồn tại. Dataset repository sẽ có `indoor5-v2-student/public/...`; model repository sẽ có `checkpoints/run_a_small_object_chair_hem_l40s/best.pth` và, khi dùng `--upload-last` và file tồn tại, `last.pth`. Nếu upload bị gián đoạn, chạy lại cùng lệnh; Hub sẽ bỏ qua phần đã có và tiếp tục file còn thiếu [1].

### Server tự download trước khi chạy launcher

Clone đúng branch targeted rewrite trên server, cài dependency và đăng nhập Hub bằng tài khoản có quyền đọc hai repository private:

```bash
cd /path/to/Object-detection-from-scratch-
python3 -m pip install -r my_submission/requirements.txt
hf auth login
```

Sau đó dùng launcher A100 hoặc V100 với `HF_AUTO_DOWNLOAD=1`. Khi dataset marker hoặc baseline checkpoint chưa tồn tại, launcher gọi `hf_sync_assets.py`; khi chúng đã tồn tại, nó không tải lại. Dataset được đặt tại `indoor5-v2-student/public/...`, còn baseline được đặt tại `checkpoints/run_a_small_object_chair_hem_l40s/best.pth`. Checkpoint tải xuống được ghi qua thư mục tạm và atomic replace để file dở dang không bị dùng cho warm-start.

Ví dụ cho A100:

```bash
HF_AUTO_DOWNLOAD=1 \
HF_DATA_REPO="YOUR_HF_USER/indoor5-v2-student-private" \
HF_MODEL_REPO="YOUR_HF_USER/fcos-indoor5-checkpoints-private" \
HF_BASELINE_RUN_NAME="run_a_small_object_chair_hem_l40s" \
RUN_NAME="targeted_highres_p2p3_a100" \
DATA_ROOT="$PWD/indoor5-v2-student/public" \
BASELINE_CHECKPOINT="$PWD/checkpoints/run_a_small_object_chair_hem_l40s/best.pth" \
my_submission/scripts/train_targeted_highres_a100.sh
```

Ví dụ cho V100:

```bash
HF_AUTO_DOWNLOAD=1 \
HF_DATA_REPO="YOUR_HF_USER/indoor5-v2-student-private" \
HF_MODEL_REPO="YOUR_HF_USER/fcos-indoor5-checkpoints-private" \
HF_BASELINE_RUN_NAME="run_a_small_object_chair_hem_l40s" \
RUN_NAME="targeted_highres_p2p3_v100" \
DATA_ROOT="$PWD/indoor5-v2-student/public" \
BASELINE_CHECKPOINT="$PWD/checkpoints/run_a_small_object_chair_hem_l40s/best.pth" \
my_submission/scripts/train_targeted_highres_v100.sh
```

Sau lần đầu, có thể tắt `HF_AUTO_DOWNLOAD` hoặc giữ nguyên; launcher sẽ kiểm tra marker và không download lại artifact đang có. Nếu muốn buộc đồng bộ lại artifact mới, chạy trực tiếp `hf_sync_assets.py --force`; không dùng `--force` trong khi một job train đang đọc cùng checkpoint. Không xóa `last.pth` của run đang train: khi khởi động lại cùng `RUN_NAME`, launcher ưu tiên checkpoint đầy đủ này và tiếp tục optimizer/scheduler state.

### Pin phiên bản artifact để tái lập

Sau lần upload đầu tiên, có thể dùng commit SHA đầy đủ thay cho `main`:

```bash
HF_DATA_REVISION="FULL_DATASET_COMMIT_SHA" \
HF_MODEL_REVISION="FULL_MODEL_COMMIT_SHA" \
HF_AUTO_DOWNLOAD=1 \
HF_DATA_REPO="YOUR_HF_USER/indoor5-v2-student-private" \
HF_MODEL_REPO="YOUR_HF_USER/fcos-indoor5-checkpoints-private" \
my_submission/scripts/train_targeted_highres_a100.sh
```

Pin revision giúp một lần train luôn lấy đúng phiên bản dataset và checkpoint đã chọn [2].

[1]: https://huggingface.co/docs/huggingface_hub/en/guides/upload "Hugging Face Hub upload guide"
[2]: https://huggingface.co/docs/huggingface_hub/en/guides/download "Hugging Face Hub download guide"

## Final train+validation fine-tune and exam grading

After the model and hyperparameters are frozen, an optional final fine-tune can use the labeled train and validation images together. This is not an independent validation run; keep the previously selected checkpoint as a backup. The merge script preserves `train/images/...` and `val/images/...` in the merged JSON and uses `indoor5-v2-student/public/` as the common image root.

On Modal L40S, first create the merged annotation inside the existing Volume:

```bash
modal run --detach my_submission/modal_app.py \
  --action merge_train_val
```

Then launch the final two-epoch fine-tune. Replace the resume path only if a different final checkpoint was selected:

```bash
modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_final_trainval_finetune_l40s \
  --config-path /root/project/my_submission/configs/train_final_trainval_finetune_l40s.json \
  --epochs 2 \
  --batch-size 6 \
  --amp \
  --resume-model-only \
  --resume-checkpoint-path /data/checkpoints/run_a_small_object_chair_hem_l40s/best.pth \
  --train-annotation-name train_val_merged.json \
  --val-annotation-name train_val_merged.json \
  --train-image-dir-name . \
  --val-image-dir-name .
```

The final fine-tune intentionally does not use the merged set as an independent validation set. It saves the final state at `/data/checkpoints/run_final_trainval_finetune_l40s/last.pth`. Download that file, then prepare the grading checkpoint without committing the `.pth` file:

```bash
modal volume get xla-fcos-volume \
  /checkpoints/run_final_trainval_finetune_l40s/last.pth \
  ./final_trainval_last.pth
bash my_submission/scripts/prepare_exam_submission.sh ./final_trainval_last.pth
```

The instructor README defines the Docker contract. Build the instructor image from the directory containing its Dockerfile, then run predictions with only the read-only test image mount. Never mount hidden annotations into the submission container:

```bash
bash my_submission/scripts/run_exam_docker.sh \
  /absolute/path/to/hidden/test/images \
  hidden_predictions.json \
  ./grading_outputs
```

The hidden evaluator remains outside the container. The final hidden score can only be produced on the instructor/grading machine that has the hidden image mount and evaluator; this repository does not contain hidden labels or the instructor Dockerfile.
