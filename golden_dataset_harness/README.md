# Golden Dataset Harness

Pipeline tạo golden dataset từ ảnh người. Hệ thống gửi ảnh tới dịch vụ vision model qua API OpenAI-compatible của vLLM, tự lấy OAuth access token và trả về caption, thuộc tính, grounding, quality score và trạng thái review.

## Luồng xử lý

```text
Ảnh → Caption + Attribute Extraction → Consensus → Grounding
    → Quality Judge → Confidence → Auto Accept / Human Review
```

Model được gọi từ xa. Repository không tải hoặc chạy PhoBERT, Qwen hay InternVL cục bộ. Provider `mock` chỉ phục vụ test tự động.

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

## vLLM API Gateway

Chạy gateway riêng nếu bạn muốn gọi từng API vLLM qua FastAPI. Gateway lấy OAuth token ở server; client không cần và không nhận access token.

```bash
uvicorn golden_dataset_harness.api.vllm_gateway:app --reload --port 8001
```

Mở [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs). Các route chính là `GET /vllm/models`, `POST /vllm/chat`, `POST /vllm/chat/stream`, `POST /vllm/responses`, `POST /vllm/embeddings`, `POST /vllm/rerank`, nhóm `/vllm/files`, `/vllm/batches` và `POST /vllm/audio/transcriptions`.

Ví dụ gọi chat qua gateway:

```bash
curl -X POST http://127.0.0.1:8001/vllm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Xin chào!"}]}'
```

Route `/vllm/chat` hỗ trợ luôn vision, JSON Schema, thinking và guardrail bằng payload tương thích OpenAI. Nếu dùng kiểu SDK, trường `extra_body` được gateway tự chuyển thành các trường JSON cấp cao nhất trước khi gửi tới vLLM.

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

Taxonomy thuộc tính nằm tại `configs/attributes.yaml`. Confidence threshold, trọng số và số caption candidates nằm tại `configs/settings.yaml`.

## Structured output

Các tác vụ attribute, grounding và judge sử dụng `response_format.type=json_schema`. Adapter kiểm tra JSON trả về và từ chối giá trị attribute ngoài taxonomy. Ảnh được chuyển sang JPEG và nén trước khi Base64 để giữ request dưới giới hạn message content của API.

Ví dụ annotation:

```json
{
  "image_id": "person_001",
  "caption": "A woman wearing a black jacket and carrying a backpack.",
  "attributes": {
    "gender": "female",
    "upper_clothing": "jacket",
    "upper_color": "black",
    "bag": "backpack",
    "hair": "long_hair"
  },
  "confidence": 0.87,
  "consensus_score": 0.85,
  "grounding_score": 0.90,
  "judge_score": 0.88,
  "review_status": "auto_accepted",
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
