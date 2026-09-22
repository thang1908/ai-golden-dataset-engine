# C1 — V-LLM Medium, trực tiếp và độc lập

C1 chỉ gửi ảnh tới API V-LLM Medium một lần và nhận JSON gồm caption cùng đúng 21
thuộc tính. Folder này không import hoặc dùng `golden_dataset_harness`.

Code được tách theo trách nhiệm: `config.py` đọc config, `auth.py` quản lý OAuth
token, `image.py` chuẩn hóa ảnh, `taxonomy.py` giữ contract 21 thuộc tính, `api.py`
gọi Chat Completions và kiểm tra response, `output.py` ghi file atomically, còn
`cli.py` chỉ phối hợp các phần trên.

Luồng theo tài liệu API:

```text
POST /oauth/token → access_token
POST /v1/chat/completions (ảnh + prompt + JSON Schema) → caption + 21 attributes
```

Model mặc định là `v-llm-v1-medium`, theo mẫu bạn gửi. Request có
`response_format.type = json_schema`; schema bắt buộc `caption`, tất cả 21 trường
trong `attributes` và không cho phép trường lạ. C1 kiểm tra lại response trước khi
in ra.

## Cài đặt

```bash
python3 -m venv c1_vllm_medium/.venv
source c1_vllm_medium/.venv/bin/activate
python -m pip install -r c1_vllm_medium/requirements.txt
```

C1 mặc định đọc `.env` chung ở thư mục gốc repository và nhận các biến
`VLLM_BASE_URL`, `VLLM_CLIENT_ID`, `VLLM_CLIENT_SECRET`, `VLLM_PROJECT_ID`,
`VLLM_TIMEOUT_SECONDS`, `VLLM_MAX_TOKENS`, `VLLM_MAX_RETRIES`. Vì `.env` chung
đang có `VLLM_MODEL=v-llm-v1-large`, C1 **không dùng biến này** và vẫn gọi
`v-llm-v1-medium`. Chỉ đặt `C1_MODEL` khi cần model Medium khác.

Environment của process ghi đè file. Có thể dùng `--dotenv` để chỉ định file khác.
Secret không có CLI argument nên không xuất hiện trong shell history hoặc process list.

```bash
# Nếu chưa có .env chung, tạo từ mẫu:
cp c1_vllm_medium/.env.example .env
# Đổi các khóa C1_* trong .env, hoặc dùng các khóa VLLM_* chung.
```

## Chạy

Từ thư mục gốc repository:

```bash
python -m c1_vllm_medium --image ./person.jpg
```

Lưu JSON vào file nếu cần:

```bash
python -m c1_vllm_medium --image ./person.jpg --output ./attributes.json
```

Có thể ghi đè các giá trị không nhạy cảm trực tiếp:

```bash
python -m c1_vllm_medium \
  --image ./person.jpg \
  --base-url https://your-vllm-host \
  --client-id YOUR_CLIENT_ID \
  --project-id YOUR_PROJECT_ID \
  --model v-llm-v1-medium
```

Ảnh được chuẩn hóa JPEG, giới hạn cạnh 1280px và nén tối đa 60 KB trước khi Base64.
Token được cache theo `expires_in`; 401 làm mới token. Timeout/network failure,
408, 409, 429 và 5xx được retry bounded exponential backoff; số lần thử mặc định
là 3, cấu hình qua `C1_MAX_RETRIES`. File `--output` được ghi qua file tạm và rename.

## Output

Output là annotation object gồm `caption` và `attributes`; attributes dùng mã lớp trong
[`attribute_schema.json`](attribute_schema.json):

```json
{
  "caption": "A person wearing a dark jacket and carrying a backpack.",
  "attributes": {
    "age": "adult",
    "gender": "female",
    "upper_clothing_type": "jacket",
    "bag_type": ["backpack"],
    "carried_objects": []
  }
}
```

Caption là một câu ngắn mô tả các chi tiết nhìn thấy được, không suy luận danh tính,
mối quan hệ, ý định hoặc chi tiết bị che khuất. Thực tế `attributes` có đủ 21 trường.
Single-label là chuỗi hoặc `null`; multi-label là mảng mã không trùng hoặc `null`.
`[]` nghĩa là đã quan sát và không có nhãn. `unknown` và `none` chỉ dùng khi schema
cho phép. C1 không tạo score, confidence, consensus, judge, batch output hoặc dữ liệu
đánh giá.
