# C2 — SigLIP ONNX trực tiếp

Ảnh → **SigLIP vision encoder + attribute head đã có trong bundle** → 21 nhóm
thuộc tính. Không gọi V-LLM, không cần OAuth/API key, không thêm prompt hay judge.
Đây là attribute head đã huấn luyện của bundle, không phải zero-shot text matching.

Runner chỉ sử dụng public API `VisionEncoder`; không thay đổi mã nguồn,
preprocessing, taxonomy hoặc trọng số trong `siglip_infer`.

## Cài đặt

Từ thư mục gốc repository, dùng Python 3.11+:

```bash
source golden_dataset_harness/.venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install -e ./siglip_infer
```

Nếu chưa có virtualenv: `python3 -m venv golden_dataset_harness/.venv` trước khi
activate. Cài editable bundle để các đường dẫn assets/models vẫn trỏ tới bundle
cục bộ. Không cần download/copy model nếu file ONNX đã có.

Các artifact dùng cho C2:

```text
siglip_infer/models/siglip_vitb_attr.onnx
siglip_infer/models/siglip_manifest.json
siglip_infer/assets/attribute_schema.json
siglip_infer/assets/attribute_taxonomy.json
```

Runner kiểm tra checksum vision/schema theo manifest, provenance của taxonomy
và thứ tự/mã lớp so với harness. Không load hoặc yêu cầu file ONNX text tower để
trích xuất thuộc tính. Giữ nguyên bộ dependency Python của bundle.

## Chạy

```bash
python -m c2_siglip \
  --image ./person.jpg \
  --device cpu \
  --output-dir ./output/c2_single

python -m c2_siglip \
  --input-dir ./sample_images \
  --device cpu \
  --threshold 0.2 \
  --output-dir ./output/c2_batch
```

Mặc định CPU để dễ chạy trên máy hiện tại. `--device auto` dùng lựa chọn provider
của bundle; `--device cuda` yêu cầu ONNX Runtime có CUDA, không tự âm thầm đổi CPU.
GPU cần runtime phù hợp theo README của `siglip_infer`; không cài đồng thời
`onnxruntime` và `onnxruntime-gpu`.

Ảnh nên là person crop/một người, cùng ảnh với C1/harness. Không có person detector.
Chạy tuần tự, khởi tạo model một lần cho toàn bộ thư mục. Preprocessing của bundle
resize về 384×128, normalize và tạo tensor FP16. Không tạo embedding output riêng.

## Giải mã và kết quả

- 17 nhóm một nhãn: softmax trên slice logits rồi argmax.
- 4 nhóm nhiều nhãn: sigmoid cho từng lớp; chọn lớp có điểm `>= --threshold`.
- Threshold mặc định **0.2**, theo bundle; thay đổi ngưỡng được ghi trong summary.
- Xuất đủ 21 nhóm kể cả các nhóm có display tier `hidden`.
- Tập nhiều nhãn không chọn được lớp nào là `[]`; không tự thêm `unknown`.
- Giữ hành vi argmax của model, không thêm quy tắc từ chối ảnh che khuất.

Output `predictions.jsonl` và `summary.json` dùng [contract chung với C1](../c1_vllm_medium/README.md#kết-quả-chung-của-c1-và-c2).
`attributes` là mã lớp chuẩn, `scores` là map thuộc tính → mã lớp → điểm,
ví dụ `{"bag_type":{"backpack":0.82},"gender":{"female":0.91}}`.
Nhóm nhiều nhãn rỗng có scores `{}`. Chỉ lưu điểm lớp được chọn, không lưu 200 logits.

Summary còn có model/manifest/taxonomy hashes, threshold, ONNX providers thực tế,
preprocessing và phiên bản thư viện. `http_attempts=0`; `model_calls` là số lần
gọi encoder. Các scores chưa được hiệu chỉnh và không thể so trực tiếp với
`confidence` tổng hợp của harness.

Output folder đã tồn tại cần `--overwrite`. Ảnh lỗi vẫn được ghi thành một hàng
error; không thay bằng nhãn mock. Thời gian load/kiểm tra model tách khỏi mỗi ảnh.

## Kiểm thử

```bash
python -m pytest tests/test_baseline_io.py tests/test_c2.py -m 'not siglip_model' -q
python -m pytest tests/test_c2.py -m siglip_model -q
```

Lệnh thứ hai chạy ONNX thật khi runtime/artifact có sẵn. Test dùng ảnh tổng hợp
để kiểm tra khả năng thực thi và schema, không kiểm chứng chất lượng thuộc tính.
Khi có ground truth mới đo accuracy/F1 và chọn ngưỡng trên tập validation riêng.
