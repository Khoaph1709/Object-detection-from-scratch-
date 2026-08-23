# Indoor5 Object Detection — HEM Final Submission

## 1. Mục tiêu và phạm vi

Đây là bài nộp cho bài toán phát hiện năm lớp đối tượng trong ảnh tự nhiên:

```text
bottle, cup, chair, laptop, backpack
```

Pipeline cuối cùng được chốt là **custom anchor-free FCOS với HEM (hard-example mining)**. Mô hình sử dụng ConvNeXt-Small làm backbone trích xuất đặc trưng, BiFPN một tầng để kết hợp đặc trưng đa tỉ lệ, các mức P2–P7 để giữ thông tin đối tượng nhỏ, và FCOS head tự cài đặt để dự đoán phân lớp, khoảng cách tới bounding box và centerness.

Mã nguồn không sử dụng YOLOv5/YOLOv8, Detectron2, MMDetection, Faster R-CNN hoặc SSD có sẵn. Các thành phần chính gồm data loader, augmentation, FCOS assignment, GIoU loss, focal classification loss, centerness loss, decoding, per-class NMS và chuyển bounding box về tọa độ ảnh gốc.

## 2. Cấu trúc dữ liệu bắt buộc

Trong môi trường chạy, thư mục dữ liệu phải có dạng:

```text
public/
├── classes.json
├── annotations/
│   ├── train.json
│   └── val.json
├── train/
│   └── images/
├── val/
│   └── images/
└── tools/
    └── evaluate_predictions.py
```

Trong repository phát triển local, layout tương ứng là:

```text
indoor5-v2-student/public/
├── annotations/train.json
├── annotations/val.json
├── train/images/
└── val/images/
```

Annotation sử dụng format:

```json
{
  "classes": ["bottle", "cup", "chair", "laptop", "backpack"],
  "images": [
    {
      "id": "img_example.jpg",
      "file_name": "train/images/img_example.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "image_id": "img_example.jpg",
      "class": "chair",
      "bbox": [48, 72, 210, 356]
    }
  ]
}
```

Bounding box luôn có dạng `[xmin, ymin, xmax, ymax]` và được tính theo pixel của ảnh gốc. Class order phải đúng chính xác: `bottle`, `cup`, `chair`, `laptop`, `backpack`.

## 3. Các file chính trong submission

```text
my_submission/
├── models/
│   └── best.pth                 # không commit; tự tải khi predict nếu cần
├── utils/
│   ├── assigner.py              # FCOS center/range assignment
│   ├── augmentations.py         # resize, normalize, flip, small-object crop
│   ├── box_ops.py               # IoU, GIoU và box operations
│   ├── checkpoint.py             # tải checkpoint qua URL và kiểm tra SHA256
│   ├── classes.py
│   ├── config.py
│   ├── dataset.py                # Dataset và batch collation
│   ├── locations.py
│   ├── losses.py                 # focal, GIoU và centerness losses
│   ├── postprocess.py            # decode và per-class NMS
│   └── tiling.py                 # sliced inference tùy chọn
├── models/
│   ├── backbone.py               # ConvNeXt backbone
│   ├── bifpn.py                  # BiFPN
│   ├── detector.py               # HEM-only FCOS detector
│   ├── fpn.py
│   └── head.py
├── scripts/
│   ├── hf_upload_assets.py       # upload private dataset/checkpoint
│   ├── hf_sync_assets.py         # tự tải artifact từ Hugging Face
│   ├── merge_train_val_annotations.py
│   ├── prepare_exam_submission.sh
│   └── run_exam_docker.sh
├── configs/
│   ├── train_hem_l40s.json
│   ├── predict_hem_l40s.json
│   └── train_final_trainval_finetune_hem_l40s.json
├── train.py
├── predict.py
├── modal_app.py
├── requirements.txt
└── README.md
```

Các config và script cho P1, targeted P2/P3, DIoU, quality-aware classification, WBF, overload mining, threshold sweep và các ablation không còn được giữ trong bản HEM-only này để tránh chạy nhầm mô hình.

## 4. Cài đặt môi trường

Trong thư mục `my_submission`, cài các dependency:

```bash
python3 -m pip install -r requirements.txt
```

Nếu chỉ chạy theo Docker grading, giảng viên cung cấp image với dependency cố định; `requirements.txt` là tài liệu môi trường phát triển và không thay thế instructor image.

## 5. Huấn luyện HEM

Config chính thức của HEM là:

```text
configs/train_hem_l40s.json
```

Các thiết lập quan trọng đã được chốt từ thí nghiệm trước gồm ConvNeXt-Small, BiFPN, P2–P7, center sampling radius 2.0, P2/P3 range overlap 8.0, small-object image sampling/crop, chair hard-example mining và full-image training.

Lệnh chạy local từ thư mục `my_submission`:

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

Lệnh train phải tạo:

```text
models/hem_run/best.pth
models/hem_run/last.pth
```

Không commit các file `.pth` vào bài nộp. Checkpoint được dùng khi grading phải được đặt ở `models/best.pth` trên máy grading hoặc được `predict.py` tự tải qua `--checkpoint_url`.

## 6. Inference

Lệnh bắt buộc theo đề bài vẫn hoạt động:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json
```

Nếu `models/best.pth` chưa tồn tại, có thể chỉ rõ nguồn checkpoint:

```bash
python predict.py \
  --image_dir ./public/val/images \
  --output predictions.json \
  --checkpoint models/best.pth \
  --checkpoint_url https://example.org/hem-best.pth \
  --checkpoint_sha256 SHA256_OF_THE_FILE
```

Trong bài nộp thật, thay URL mẫu bằng URL checkpoint riêng có quyền truy cập phù hợp. `checkpoint.py` tải qua file tạm rồi đổi tên atomic; nếu cung cấp SHA256, file chỉ được dùng khi hash khớp.

Để tái hiện sliced inference đã dùng khi chọn baseline HEM, dùng config:

```bash
python predict.py \
  --config configs/predict_hem_l40s.json \
  --image_dir ./public/val/images \
  --output val_predictions_sliced.json \
  --checkpoint /path/to/hem/best.pth \
  --tile_inference
```

Kết quả phải là một JSON array. Mỗi phần tử có dạng:

```json
{
  "image_id": "img_example.jpg",
  "boxes": [
    {
      "class": "chair",
      "confidence": 0.91,
      "bbox": [48, 72, 210, 356]
    }
  ]
}
```

Ảnh không có detection vẫn phải xuất `"boxes": []`. Confidence phải nằm trong `[0, 1]`, class phải thuộc năm lớp quy định, và box phải ở tọa độ ảnh gốc.

## 7. Kiểm tra public validation

Từ thư mục chứa `public`:

```bash
python public/tools/evaluate_predictions.py \
  --ground_truth public/annotations/val.json \
  --predictions val_predictions_sliced.json \
  --output val_score.json
```

Validation chỉ được dùng để chọn checkpoint trước khi chốt. Sau khi gộp train + validation cho final fine-tune, merged set không còn là validation độc lập và không được dùng để báo cáo một mAP tổng quát hóa độc lập.

## 8. Final fine-tune trên train + validation bằng Modal L40S

Chỉ thực hiện bước này **sau khi đã chốt HEM checkpoint**. Bước này không dùng targeted P2/P3.

### 8.1. Chuẩn bị branch và Volume

```bash
git checkout experiment/targeted-highres-p2p3-rewrite
git pull --ff-only origin experiment/targeted-highres-p2p3-rewrite
```

Modal Volume phải có:

```text
/data/indoor5-v2-student/public/annotations/train.json
/data/indoor5-v2-student/public/annotations/val.json
/data/indoor5-v2-student/public/train/images/
/data/indoor5-v2-student/public/val/images/
/data/checkpoints/run_a_small_object_chair_hem_l40s/best.pth
```

Nếu cần upload dataset từ local, truyền trực tiếp thư mục `public`:

```bash
modal run my_submission/modal_app.py \
  --action upload \
  --local-data-dir ./indoor5-v2-student/public
```

### 8.2. Gộp annotation

```bash
modal run my_submission/modal_app.py \
  --action merge_train_val
```

File kết quả là:

```text
/data/indoor5-v2-student/public/annotations/train_val_merged.json
```

### 8.3. Chạy HEM final fine-tune

```bash
modal run --detach my_submission/modal_app.py \
  --action train \
  --gpu L40S \
  --run-name run_final_trainval_finetune_hem_l40s \
  --config-path /root/project/my_submission/configs/train_final_trainval_finetune_hem_l40s.json \
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

Config `train_final_trainval_finetune_hem_l40s.json` đặt `targeted_highres=false`, giữ `hard_negative_sampling=true`, small-object crop/sampling, radius 2.0, GIoU và HEM settings. Vì merged annotation được truyền vào cả train và validation path chỉ để đáp ứng interface hiện tại, validation bị trì hoãn trong config và không được dùng để chọn `best.pth`.

Checkpoint cuối cần lấy là:

```text
/data/checkpoints/run_final_trainval_finetune_hem_l40s/last.pth
```

Tải về và đặt vào path grading:

```bash
modal volume get xla-fcos-volume \
  /checkpoints/run_final_trainval_finetune_hem_l40s/last.pth \
  ./final_trainval_hem_last.pth

bash my_submission/scripts/prepare_exam_submission.sh ./final_trainval_hem_last.pth
```

Sau đó checkpoint grading nằm tại:

```text
my_submission/models/best.pth
```

Giữ lại bản HEM gốc làm backup. Không ghi đè `run_a_small_object_chair_hem_l40s/best.pth`.

## 9. Docker grading

Đề bài yêu cầu nộp thư mục `my_submission/`, không nộp lại `public/` và không đưa bất kỳ `.pth` nào vào file zip. Trên máy grading, instructor image được build từ thư mục chứa Dockerfile:

```bash
docker build -t object-detection-exam:2026 .
```

Kiểm tra public validation trước:

```bash
bash my_submission/scripts/run_exam_docker.sh \
  ./indoor5-v2-student/public/val/images \
  val_predictions.json \
  ./my_submission/grading_outputs
```

Sau đó đánh giá ở ngoài container:

```bash
python3 indoor5-v2-student/public/tools/evaluate_predictions.py \
  --ground_truth indoor5-v2-student/public/annotations/val.json \
  --predictions my_submission/grading_outputs/val_predictions.json \
  --output my_submission/grading_outputs/val_score.json
```

Khi instructor cung cấp thư mục ảnh hidden, chỉ thay image mount:

```bash
bash my_submission/scripts/run_exam_docker.sh \
  /absolute/path/to/hidden/test/images \
  hidden_predictions.json \
  ./my_submission/grading_outputs
```

Script chỉ mount hidden images read-only, workspace submission và output directory. **Không mount hidden annotations vào container.** Hidden evaluator chạy ở ngoài container theo đúng README grading của giảng viên.

## 10. Hugging Face artifact workflow

Nếu dùng Hugging Face để phân phối artifact, giữ hai repository private:

```text
Khoaph/indoor5-v2-student-private
Khoaph/fcos-indoor5-checkpoints-private
```

Upload dataset đúng từ thư mục `public`:

```bash
python3 my_submission/scripts/hf_upload_assets.py \
  --project-root . \
  --dataset-repo Khoaph/indoor5-v2-student-private \
  --dataset-dir ./indoor5-v2-student/public \
  --model-repo Khoaph/fcos-indoor5-checkpoints-private \
  --run-name run_a_small_object_chair_hem_l40s \
  --upload-last
```

Repository dataset sau upload phải giữ path:

```text
indoor5-v2-student/public/annotations/train.json
indoor5-v2-student/public/annotations/val.json
indoor5-v2-student/public/train/images/...
indoor5-v2-student/public/val/images/...
```

Không commit HF token. Dùng `hf auth login` hoặc secret environment `HF_TOKEN`. Trên server, `hf_sync_assets.py` tải dataset về `indoor5-v2-student/public/` và checkpoint về `checkpoints/run_a_small_object_chair_hem_l40s/best.pth` nếu artifact chưa có [1].

## 11. HEM architecture summary

```text
Input image
    ↓
Resize + normalize + horizontal flip + controlled small-object crop
    ↓
ConvNeXt-Small backbone
    ↓
BiFPN, one layer
    ↓
P2, P3, P4, P5, P6, P7 feature levels
    ↓
FCOS head
    ├── focal classification loss
    ├── GIoU box loss
    └── centerness loss
    ↓
Center sampling radius = 2.0
P2/P3 range overlap = 8.0
    ↓
Decode → confidence threshold → per-class NMS
    ↓
Original-image bounding boxes
```

The detector implementation is HEM-only. P1 and targeted high-resolution P2/P3 branches are intentionally not part of the final runtime, so a stale experiment cannot be selected accidentally through the cleaned repository.

## 12. Academic and data-use notes

The dataset contains course material and should remain private unless the instructor or dataset license explicitly permits redistribution. The data-quality and error-analysis artifacts used during development are not part of the final submission. Hidden annotations must not be copied, inspected or mounted into the submission container before official grading.

The final HEM checkpoint previously achieved a sliced validation mAP@0.5 of approximately `0.693145`. This is a validation result, not a guarantee of hidden-test performance. The optional final train+validation fine-tune is an independent experiment and should not overwrite the original HEM checkpoint.

[1]: https://huggingface.co/docs/huggingface_hub/en/guides/upload "Hugging Face Hub upload guide"

## References

[1]: https://huggingface.co/docs/huggingface_hub/en/guides/upload "Hugging Face Hub upload guide"
