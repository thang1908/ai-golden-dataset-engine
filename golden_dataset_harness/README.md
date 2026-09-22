# Golden Dataset Harness

Hai baseline để chuẩn bị so sánh thuộc tính:

| Phương pháp | Cách chạy | Hướng dẫn |
|---|---|---|
| Harness hiện tại | Nhiều caption, consensus, grounding, judge | Tài liệu bên dưới |
| C1 | Một extraction V-LLM Medium + prompt/schema | [c1_vllm_medium](../c1_vllm_medium/README.md) |
| C2 | SigLIP ONNX vision + attribute head cục bộ | [c2_siglip](../c2_siglip/README.md) |

Hai baseline xuất JSONL và summary cùng schema thuộc tính 21 nhóm. Đây là báo cáo
thực thi; chưa có ground truth để tính accuracy/F1. Không thay mặc định model của
harness khi chạy C1 hoặc C2.

Pipeline tạo golden dataset từ ảnh người. Hệ thống gửi ảnh tới dịch vụ vision model qua API OpenAI-compatible của vLLM, tự lấy OAuth access token và trả về caption, thuộc tính, grounding, quality score và trạng thái review.

## Luồng xử lý

```text
Ảnh → Caption + Attribute Extraction → Consensus → Grounding
    → Quality Judge → Confidence → Auto Accept / Human Review
```

Model được gọi từ xa qua vLLM. Provider `mock` chỉ phục vụ test tự động.

## 1. Cài đặt

Yêu cầu Python 3.11 trở lên. Chạy từ thư mục gốc repository:

```bash
python3 -m venv golden_dataset_harness/.venv
source golden_dataset_harness/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Nếu virtual environment đã tồn tại, chỉ cần `source golden_dataset_harness/.venv/bin/activate`.

## 2. Tạo cấu hình môi trường

```bash
cp golden_dataset_harness/.env.example golden_dataset_harness/.env
```

Điền thông tin do nền tảng cung cấp vào `golden_dataset_harness/.env`:

```dotenv
VLLM_BASE_URL=https://your-real-host
VLLM_CLIENT_ID=your-client-id
VLLM_CLIENT_SECRET=your-client-secret
VLLM_PROJECT_ID=your-project-id
VLLM_MODEL=your-model-name

VLLM_TIMEOUT_SECONDS=120
VLLM_MAX_RETRIES=3
VLLM_MAX_TOKENS=1024
VLLM_MAX_IMAGE_BYTES=60000
VLLM_ENABLE_THINKING=false
VLLM_GUARDRAIL=off
VLLM_GUARD_OUTPUT_MODE=refuse
VLLM_TOKEN_REFRESH_SKEW_SECONDS=60
```

`VLLM_BASE_URL` có thể là `https://host` hoặc `https://host/v1`; code sẽ chuẩn hoá URL. Không điền `/chat/completions` vào biến này.

Credential chỉ nằm trong `.env`. File này đã được loại khỏi Git và Docker build context.

Kiểm tra cấu hình mà không in secret:

```bash
python -c "from golden_dataset_harness.models.provider_settings import ProviderSettings; s=ProviderSettings(); print(s.oauth_url, s.openai_base_url, s.vllm_model)"
```

## 3. Chạy API

```bash
uvicorn golden_dataset_harness.api.main:app --reload --port 8000
```

Mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), chọn `POST /annotate`, rồi upload một ảnh. Có thể gọi bằng curl:

```bash
curl -X POST http://127.0.0.1:8000/annotate \
  -H "accept: application/json" \
  -F "file=@/duong-dan/toi/person.jpg"
```

Khi có request đầu tiên, client gọi `POST /oauth/token`, cache access token theo `expires_in`, sau đó gọi `POST /v1/chat/completions`. Token được làm mới trước khi hết hạn và được lấy lại nếu model API trả HTTP 401.

Các endpoint:

| Method | Path | Chức năng |
|---|---|---|
| `POST` | `/annotate` | Xử lý một ảnh |
| `POST` | `/annotate/batch` | Xử lý nhiều ảnh |
| `GET` | `/annotations/{image_id}` | Lấy annotation theo ID |
| `GET` | `/annotations?status=pending_review` | Lọc theo trạng thái |
| `GET` | `/export/label-studio` | Export dữ liệu review |
| `POST` | `/review/{image_id}` | Gửi kết quả human review |
| `GET` | `/stats` | Thống kê pipeline |
| `GET` | `/taxonomy` | 21 nhóm thuộc tính và 200 lớp SigLIP |
| `GET` | `/export/siglip` | Các bản ghi đã accept/approve, dạng ô nhãn SigLIP |
| `GET` | `/export/label-studio/config` | XML cấu hình các trường review trong Label Studio |

## 4. Chạy batch

```bash
python -m golden_dataset_harness.workflow.runner \
  --input-dir ./sample_images \
  --output-dir ./output \
  --concurrency 2
```

Kết quả được ghi vào `output/dataset.jsonl`. Một ảnh tạo nhiều lời gọi model, vì vậy nên bắt đầu với concurrency 1 hoặc 2 rồi tăng theo rate limit của dịch vụ.

## 5. Chạy bằng Docker

```bash
docker compose -f golden_dataset_harness/docker-compose.yml up --build
```

Compose tự đọc `golden_dataset_harness/.env` và chạy API ở cổng 8000.

## Cấu hình pipeline

`configs/settings.yaml` chọn provider và các tham số của pipeline. Host, credential và tên model luôn lấy từ `.env`.

```yaml
model:
  provider: "openai-compatible"
```

Taxonomy thuộc tính nằm tại `configs/attributes.yaml`, đồng bộ tên, thứ tự, loại nhãn và
mã lớp từ `siglip_infer/assets/attribute_schema.json` (v1). Pipeline đóng gói taxonomy
riêng nên không cần cài SigLIP hay chạy ONNX. Confidence threshold, trọng số và số
caption candidates nằm tại `configs/settings.yaml`.

### Thuộc tính SigLIP

| Nhóm | Số lớp | Kiểu |
|---|---:|---|
| `age` | 6 | Một nhãn |
| `gender` | 2 | Một nhãn |
| `body_build` | 3 | Một nhãn |
| `upper_clothing_type` | 25 | Một nhãn |
| `upper_clothing_color` | 15 | Một nhãn |
| `lower_clothing_type` | 16 | Một nhãn |
| `lower_clothing_color` | 15 | Một nhãn |
| `clothing_style` | 5 | Một nhãn |
| `upper_pattern` | 7 | Một nhãn |
| `footwear_type` | 13 | Một nhãn |
| `bag_type` | 11 | Nhiều nhãn |
| `bag_color` | 15 | Nhiều nhãn |
| `headwear` | 8 | Một nhãn |
| `eyewear` | 4 | Một nhãn |
| `face_mask` | 5 | Một nhãn |
| `other_accessories` | 8 | Nhiều nhãn |
| `hair_length` | 6 | Một nhãn |
| `hair_texture` | 3 | Một nhãn |
| `hairstyle` | 5 | Một nhãn |
| `hair_color` | 12 | Một nhãn |
| `carried_objects` | 16 | Nhiều nhãn |

JSON trả về đủ 21 trường. Trường một nhãn là chuỗi; trường nhiều nhãn là mảng chuỗi
không trùng. Các trạng thái được giữ riêng:

- `null`: chưa gán được nhãn, ví dụ bị che khuất và taxonomy không có lớp `unknown`.
  Bản ghi này luôn chuyển sang human review, kể cả confidence cao.
- `[]`: đã quan sát và xác nhận không có vật/phụ kiện thuộc nhóm nhiều nhãn.
- `"unknown"`: câu trả lời hợp lệ chỉ trong nhóm có lớp này; với `bag_color` dùng
  `["unknown"]`. Không thêm lớp này vào `gender`, `body_build` hoặc các nhóm khác.
- `"none"`: mã lớp một nhãn của `headwear`, `eyewear`, `face_mask`.

Grounding kiểm tra riêng từng nhãn, kể cả khẳng định không có đồ vật/phụ kiện.
Nhãn sai, trùng, trường lạ hoặc sai kiểu bị từ chối. VLM phải trả đủ các trường.

Đây là thay đổi schema so với bản 11 thuộc tính cũ. Các tên như `age_group`,
`upper_color`, `bag`, `hair`, `hat` không còn hợp lệ. Dữ liệu cũ cần được gán nhãn lại
hoặc chuyển đổi có review vì các lớp mới chi tiết hơn; không tự đổi `t-shirt` thành
một kiểu tay áo khi chưa có bằng chứng. Mỗi bản ghi mới có
`attribute_schema_version: "siglip-v1"`.

### Review và export

`POST /review/{image_id}` nhận các thay đổi từng trường; trường không gửi lên được giữ nguyên:

```json
{
  "status": "human_approved",
  "corrected_caption": "A person carrying a backpack and a handbag.",
  "corrected_attributes": {
    "bag_type": ["backpack", "handbag"],
    "carried_objects": [],
    "headwear": "none"
  }
}
```

Label Studio: lấy XML từ `/export/label-studio/config` để cấu hình project trước khi
import `/export/label-studio`. Mỗi thuộc tính có control riêng; nhóm nhiều nhãn có
lựa chọn `none` để đánh dấu tập rỗng. Không chọn `none` cùng nhãn khác. Hàm
`parse_label_studio_export` kiểm tra và chuyển kết quả về cùng schema của API.

`GET /export/siglip` trả JSON có danh sách `rows` cho bản ghi `auto_accepted` hoặc
`human_approved`. Trong mỗi ô nhãn: nhiều nhãn nối bằng `|`, tập rỗng ghi `none`,
chưa gán nhãn ghi chuỗi rỗng. Ví dụ `backpack|handbag`. Đây là định dạng ô của schema
SigLIP, không phải logit hay embedding. Batch runner vẫn ghi `dataset.jsonl` với
thuộc tính dạng JSON đầy đủ, gồm cả các bản ghi cần review.

## Structured output

Các tác vụ attribute, grounding và judge sử dụng `response_format.type=json_schema`. Adapter kiểm tra JSON trả về và từ chối giá trị attribute ngoài taxonomy. Ảnh được chuyển sang JPEG và nén trước khi Base64 để giữ request dưới giới hạn message content của API.

Ví dụ annotation:

```json
{
  "image_id": "person_001",
  "caption": "A woman wearing a black jacket and carrying a backpack.",
  "attributes": {
    "age": "adult",
    "gender": "female",
    "body_build": "average",
    "upper_clothing_type": "jacket",
    "upper_clothing_color": "black",
    "lower_clothing_type": "jeans",
    "lower_clothing_color": "blue",
    "clothing_style": "casual_style",
    "upper_pattern": "solid",
    "footwear_type": "casual_sneakers",
    "bag_type": ["backpack"],
    "bag_color": ["black"],
    "headwear": "none",
    "eyewear": "none",
    "face_mask": "none",
    "other_accessories": [],
    "hair_length": "long_hair",
    "hair_texture": "straight_hair",
    "hairstyle": "loose_hair",
    "hair_color": "black",
    "carried_objects": []
  },
  "confidence": 0.87,
  "consensus_score": 0.85,
  "grounding_score": 0.90,
  "judge_score": 0.88,
  "review_status": "auto_accepted",
  "attribute_schema_version": "siglip-v1",
  "model_version": "your-model-name"
}
```

## Test

Test dùng mock HTTP transport, không gọi dịch vụ thật và không cần credential thật:

```bash
python -m pytest tests -q
```

Các test bao phủ OAuth token cache/refresh, retry khi HTTP 401, payload vision Base64, JSON Schema, kiểm tra taxonomy và pipeline end-to-end.

## Các file chính

```text
.
├── pyproject.toml
├── tests/
└── golden_dataset_harness/
    ├── .env.example
    ├── agents/
    ├── api/main.py
    ├── configs/
    │   ├── attributes.yaml
    │   └── settings.yaml
    ├── models/
    │   ├── base.py
    │   ├── factory.py
    │   ├── mock_vlm.py
    │   ├── oauth.py
    │   ├── openai_compatible.py
    │   └── provider_settings.py
    ├── workflow/
    │   ├── graph.py
    │   └── runner.py
    ├── docker-compose.yml
    └── Dockerfile
```
