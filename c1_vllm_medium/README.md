# C1 — V-LLM Medium + prompt trực tiếp

Ảnh → **một lần trích xuất thuộc tính** bằng V-LLM Medium → JSON đủ 21 nhóm.
Không chạy caption, consensus, grounding hoặc judge. Dùng cùng prompt/schema và
adapter OAuth của harness để dễ so sánh. HTTP có thể retry; báo cáo đếm riêng
`model_calls` (extraction logic) và `http_attempts` (chat requests thực tế, không gồm OAuth).

Chạy các lệnh bên dưới từ **thư mục gốc repository**.

## Cài đặt

```bash
python3 -m venv golden_dataset_harness/.venv
source golden_dataset_harness/.venv/bin/activate
python -m pip install -e ".[dev]"
```

Nếu virtualenv đã có, chỉ cần activate và cập nhật cài đặt. C1 không cần cài SigLIP.

Điền host, client ID, client secret và project ID hợp lệ trong
`golden_dataset_harness/.env` theo `.env.example`. **Không tự động dùng
`VLLM_MODEL` hoặc mặc định Large của harness.** Chỉ định model ID Medium thật qua
`--model`, hoặc biến môi trường shell `C1_VLLM_MODEL`.

`MODEL_ID_MEDIUM` trong ví dụ là chỗ thay bằng ID do nền tảng cung cấp; runner sẽ
từ chối chuỗi placeholder này. Chưa xác minh ID Medium nên không đoán tên.
`--model` ưu tiên hơn `C1_VLLM_MODEL`; biến `C1_VLLM_MODEL` được đọc từ shell,
không phải từ file `.env` của adapter.

## Chạy

```bash
python -m c1_vllm_medium \
  --image ./person.jpg \
  --model MODEL_ID_MEDIUM \
  --output-dir ./output/c1_single

python -m c1_vllm_medium \
  --input-dir ./sample_images \
  --model MODEL_ID_MEDIUM \
  --output-dir ./output/c1_batch
```

Ảnh đầu vào nên là một người/person crop, giống input của C2 và harness. Thư mục
được duyệt không đệ quy, theo thứ tự filename, hỗ trợ JPG/JPEG/PNG/BMP/WebP.
Chạy tuần tự, reuse HTTP client và OAuth token. Muốn chạy lại vào cùng folder phải
thêm `--overwrite`; chỉ thay `predictions.jsonl` và `summary.json` của baseline.

## Kết quả chung của C1 và C2

- `predictions.jsonl`: một dòng cho **mọi ảnh**, kể cả ảnh lỗi.
- `summary.json`: số thành công/lỗi, thời gian, số lời gọi, config và hashes.

Mỗi dòng có `schema_version=baseline-v1`, `attribute_schema_version=siglip-v1`,
`run_id`, `image_id` (filename gồm extension), `image_path`, `image_sha256`,
`method`, `model_id`, `status`, `attributes`, `scores`, `elapsed_ms`,
`model_calls`, `http_attempts`, `error`.

`attributes` thành công có đủ 21 nhóm với mã lớp giống harness:

- Một nhãn: chuỗi hoặc `null`. Nhiều nhãn: mảng chuỗi hoặc `null`.
- `null`: chưa gán được nhãn; `[]`: tập rỗng đã được quan sát.
- `unknown` chỉ hợp lệ khi có trong taxonomy; `none` chỉ là mã của nhóm cho phép.
- Lỗi: `attributes=null`, `scores=null`, có `error.code` và thông báo an toàn.

C1 luôn có `scores=null`: không tạo confidence giả. C2 có điểm softmax/sigmoid;
chúng chưa được hiệu chỉnh thành xác suất đúng. Không có caption hoặc auto-accept.

`elapsed_ms` gồm đọc/kiểm tra ảnh, preprocessing, inference, validation và retry;
không gồm setup model. `model_load_ms` trong summary gồm setup/kiểm tra artifact.
Latency success chỉ tính ảnh thành công; báo cáo vẫn giữ số lỗi riêng.

Summary lưu prompt hash, schema/taxonomy hash, temperature, giới hạn ảnh/token,
retry/timeout và phiên bản thư viện; không chứa credential, token hoặc ảnh base64.
Không có ground truth nên chưa tính accuracy/F1 hoặc xếp hạng chất lượng.

Exit code: `0` tất cả thành công; `1` có lỗi inference/runtime; `2` sai input/config;
`130` bị ngắt. Khi bị ngắt, JSONL giữ các hàng hoàn tất; có thể chưa có summary.

## Kiểm thử

```bash
python -m pytest tests/test_baseline_io.py tests/test_c1.py -q
```

Test C1 dùng HTTP giả lập cho OAuth, structured extraction, retry và lỗi; không
gọi model thật hay chứng minh độ chính xác trên ảnh. Chạy API thật cần credential
và model ID Medium hợp lệ.

## So sánh với harness

So phần `attributes` trên cùng ảnh/crop. Baseline giữ extension trong image_id,
còn batch harness hiện dùng stem; ghép theo đường dẫn ảnh hoặc nội dung ảnh khi
xây evaluator, không join mù theo image_id. C1 Medium so với harness Large là so
hai cấu hình hệ thống; muốn đo riêng tác dụng orchestration phải giữ cùng model.
