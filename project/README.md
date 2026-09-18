# Camera AI Person Query Understanding

Bộ phân tích câu tiếng Việt dùng **Transformer encoder + multi-task classification**, không dùng LLM. Luồng duy nhất: câu truy vấn người dùng → bóc tách thuộc tính → JSON. Backbone mặc định là `vinai/phobert-base`; Underthesea tự tách từ tiếng Việt trước tokenization, đồng nhất khi train và inference.

## Giải thích từng file

| File | Chức năng |
|---|---|
| `phobert.json` | Cấu hình PhoBERT sẵn dùng, bật word segmentation |
| `config.py` | Dataclass cấu hình model/training, loss weights, threshold; kiểm tra cấu hình |
| `labels.json` | Thứ tự nhãn dùng thống nhất từ annotation đến inference |
| `model.py` | `PersonQueryParser(nn.Module)`, AutoModel, shared CLS/dropout, các linear heads và weighted loss |
| `dataset.py` | HF Datasets đọc JSON, kiểm tra annotation, preprocessing/tokenizer, PyTorch Dataset và padding collator |
| `checkpoint.py` | Ghi bundle và chỉ thay champion khi validation tốt hơn |
| `tracking.py` | MLflow params/metrics/metadata, không lưu bản sao weights |
| `migrate_checkpoint.py` | Nhập lịch sử epoch cũ vào MLflow, giữ champion và xóa các epoch/state dư |
| `metrics.py` | Decode và gender accuracy, F1, exact match |
| `train.py` | AdamW, warmup/linear scheduler, Accelerate, gradient accumulation/clipping, validation, MLflow và champion-only checkpoint |
| `inference.py` | Load bundle offline, batch predict, JSON decode và CLI |
| `api.py` | FastAPI lifespan, `/parse_query`, `/health`, request validation và lỗi JSON |
| `train.json`, `validation.json` | 12/4 câu minh họa để chạy thử; **không phải dataset production** |
| `requirements.txt` | Dependency ranges; phiên bản đã kiểm thử được ghi riêng trong `requirements-tested.txt` |
| `tests/` | Kiểm thử offline bằng encoder RoBERTa rất nhỏ cùng kiến trúc PhoBERT, gồm training, tracking và chọn champion |

## Cài đặt

Python 3.10 trở lên. Chạy các lệnh từ thư mục **cha của `project/`**:

```bash
python3 -m venv .venv-query
source .venv-query/bin/activate
pip install -r project/requirements.txt
python -m pytest project/tests -q
```

Test không tải pretrained model, chỉ xác minh kỹ thuật bằng encoder nhỏ khởi tạo ngẫu nhiên. Training với backbone mặc định tải weights/tokenizer từ Hugging Face lần đầu. CPU chạy được nhưng full PhoBERT train sẽ chậm; CUDA GPU phù hợp hơn.

## Kiến trúc và loss

```text
Text → AutoTokenizer → AutoModel → last_hidden_state[:, 0, :] → Dropout
                                      ├─ gender: 3 classes
                                      ├─ clothing_type: 7 classes
                                      ├─ clothing_color: 9 classes
                                      ├─ hair_color: 4 classes
                                      ├─ hair_length: 3 classes
                                      └─ accessories: 5 independent labels
```

`forward()` trả `gender_logits`, `clothing_type_logits`, `clothing_color_logits`, `hair_logits={color, length}`, `accessory_logits`. Hair cần hai tensor vì số lớp khác nhau.

Categorical heads dùng CrossEntropyLoss trên raw logits và target long. Accessory dùng BCEWithLogitsLoss trên raw logits và target float multi-hot.

```text
hair_loss = CE(hair_color) + CE(hair_length)
total_loss = w_gender * CE(gender)
           + w_clothing_type * CE(clothing_type)
           + w_clothing_color * CE(clothing_color)
           + w_hair * hair_loss
           + w_accessory * BCE(accessories)
```

Weights mặc định đều 1; hair mặc định cộng hai CE. Cấu hình lưu cùng checkpoint. Đổi backbone bằng `backbone`, `revision`, `segmentation`; hidden size lấy tự động từ encoder.

## Dataset

File JSON **array**, mỗi record phải có đủ trường:

```json
[
  {
    "text": "cô gái mặc áo sơ mi trắng tóc đen mang túi đỏ",
    "gender": "female",
    "clothing_type": "shirt",
    "clothing_color": "white",
    "hair_color": "black",
    "hair_length": "unknown",
    "accessories": ["bag"]
  }
]
```

Danh sách giá trị hợp lệ nằm trong `labels.json`. `unknown` nghĩa câu không xác định thuộc tính; missing annotation là lỗi, không tự đổi thành unknown. Accessory không nhắc đến có nhãn 0, **không** tạo điều kiện loại người có phụ kiện đó. Màu phụ kiện chưa có head/annotation: JSON chỉ xuất loại; màu phụ kiện chưa được bóc tách.

Input được chuẩn hóa Unicode NFC và khoảng trắng. Input vượt 2000 ký tự hoặc `max_length` token bị từ chối thay vì cắt mất thuộc tính. Không tự suy gender từ áo/tóc.

Chuẩn bị dữ liệu thực có paraphrase, lỗi chính tả, câu thiếu thuộc tính, câu ngoài miền. Tách train/val/test theo nhóm template/paraphrase để tránh leakage. Trainer chặn câu trùng sau normalize/casefold giữa train/val; near-duplicate cần xử lý ở bước chuẩn bị dữ liệu. Schema v1 chỉ mô tả một người, một áo chính; không biểu diễn OR, điều kiện phủ định, hoặc hai túi khác màu.

## Training

Tạo config để chỉnh hyperparameters:

```bash
python -c 'from project.config import Config; Config().save("project/config.local.json")'
python -m project.train \
  --train project/train.json \
  --validation project/validation.json \
  --config project/config.local.json \
  --output project/checkpoints/run-1
```

Mặc định precision `no`. Với CUDA, đổi `mixed_precision` thành `fp16` hoặc `bf16` nếu phần cứng hỗ trợ. CPU/MPS dùng `no`; trên máy Mac muốn ép CPU có thể đặt `ACCELERATE_USE_CPU=true`. Chạy distributed qua `accelerate launch -m project.train` với các tham số tương tự.

Ví dụ loss weights trong config:

```json
{"gender": 1.0, "clothing_type": 1.0, "clothing_color": 1.0, "hair": 0.5, "accessory": 1.5}
```

Đặt object trên vào field `loss_weights`, không thay toàn bộ config. Các cấu hình batch size, LR, epochs, accumulation, seed, warmup, dropout và threshold đều trong `Config`.

PhoBERT đã là mặc định; có thể chạy training mà không truyền `--config`. Underthesea được cài cùng requirements và tự tách từ, bạn chỉ nhập câu tiếng Việt bình thường.

Nếu đã tạo `config.local.json` từ phiên bản XLM-R trước đây, hãy tạo cấu hình mới để dùng PhoBERT:

```bash
python -c 'from project.config import Config; Config().save("project/phobert.json")'
```

Sau đó truyền `--config project/phobert.json` và chọn output mới, chẳng hạn `project/checkpoints/phobert-run-1`. Mỗi checkpoint dùng backbone/preprocessing đã lưu trong bundle.

Word segmentation chạy giống nhau khi train và inference. Underthesea là lựa chọn của pipeline này; tác giả PhoBERT khuyến nghị VnCoreNLP/RDRSegmenter giống pretraining. Chất lượng cần được đánh giá trên tập truy vấn thực tế.

## MLflow và checkpoint champion

Mỗi lần training tạo một MLflow run. `--output` phải là thư mục mới hoặc rỗng; mỗi output chỉ giữ **một champion của run đó**. Không tự so sánh các run dùng tập validation khác nhau.

```text
project/
├── mlflow.db                   # Params, tags, toàn bộ lịch sử metrics
├── mlartifacts/                # Config, dataset hashes, champion metadata; không weights
└── checkpoints/
    └── phobert-run-2/
        └── champion/
            ├── model.safetensors
            ├── encoder/
            ├── tokenizer/
            ├── labels.json
            ├── config.json
            └── manifest.json  # Epoch tốt nhất và MLflow run ID
```

Champion được chọn theo `exact_match_accuracy` lớn nhất; nếu bằng nhau, chọn `validation_loss` nhỏ hơn. Nếu cả hai bằng nhau thì giữ champion cũ. Epoch kém hơn không ghi weights. Candidate được ghi hoàn chỉnh trước khi thay champion; có rollback nếu đổi thư mục thất bại. Trong lúc thay thế có thể cần dung lượng tạm cho hai bundle. API nạp model một lần khi startup; khởi động API sau khi training xong.

MLflow ghi config, loss weights, hash train/validation, world size, từng loss train/validation, accuracy/F1/exact match, learning rate, global step, epoch được promote. Không dùng autolog/log_model để tránh tạo thêm bản sao model. Chỉ main process ghi tracking/checkpoint. `champion.json` ở MLflow chỉ là metadata của champion.

Không lưu `epoch-N/`, `state-N/`, best/last pointers hoặc optimizer/RNG state trong run mới. Đã bỏ `--resume`; checkpoint champion dùng cho inference, không khôi phục chính xác training đang dở. MLflow không thay thế optimizer state.

Ví dụ training mới trong môi trường `.venv` hiện có:

```bash
.venv/bin/python -m pip install -r project/requirements.txt
ACCELERATE_USE_CPU=true .venv/bin/python -m project.train \
  --config project/phobert.json \
  --train project/train.json \
  --validation project/validation.json \
  --output project/checkpoints/phobert-run-2 \
  --run-name phobert-run-2
```

Mặc định SQLite nằm tại `project/mlflow.db`, không cần bật server khi train. Xem tracking từ thư mục cha `project/`:

```bash
.venv/bin/python -m mlflow ui \
  --backend-store-uri sqlite:///project/mlflow.db \
  --host 127.0.0.1 --port 5000
```

Mở **http://127.0.0.1:5000**, chọn experiment `person-query-parser`. Có thể dùng `--tracking-uri`, biến `MLFLOW_TRACKING_URI`, `--experiment`, `--run-name` để cấu hình khác. Server từ xa tự quản artifact store.

Chuyển một run kiểu cũ sang champion-only (lệnh này **xóa epoch/state dư** sau khi xác minh champion nạp được và metrics được ghi vào MLflow):

```bash
.venv/bin/python -m project.migrate_checkpoint \
  --run project/checkpoints/phobert-run-1
```

Không chạy lại migration khi thư mục đã có `champion/`. Chỉ nhận đúng cấu trúc legacy của trainer này; gặp entry lạ hoặc best.txt không khớp metrics thì dừng trước khi xóa. Model được chuyển bằng rename, không retrain. Lịch sử cũ chỉ nhập các metrics đã lưu, không dựng lại learning rate chưa ghi trước đó.

## Metrics

Validation mỗi epoch báo:

- `gender_accuracy`: gồm cả unknown.
- Macro F1 từng categorical head; accessory micro/macro F1.
- `attribute_micro_f1`, `attribute_macro_f1`: one-hot của năm categorical heads cộng năm accessory bits.
- `known_attribute_*`: loại cột unknown khỏi F1 để thấy ảnh hưởng mất cân bằng.
- `exact_match_accuracy`: cả năm categorical labels và toàn bộ accessory bits đúng trên cùng câu.

Macro F1 tính trên vocabulary cố định; lớp không có TP/FP/FN nhận F1=0. Threshold mặc định categorical=0 (argmax), accessory=0.5; tune trên validation, đánh giá cuối trên test riêng. Không dùng test để chọn checkpoint. Không có mức accuracy production được cam kết.

## Inference

Truyền thư mục run; chương trình tự nạp `champion/`. Run legacy chưa chuyển đổi vẫn được đọc qua `best.txt`. Không cần biến `$BEST_EPOCH`:

```bash
python -m project.inference \
  --checkpoint project/checkpoints/run-1 \
  --text 'Tìm cô gái áo trắng tóc đen'
```

Thay `run-1` bằng tên run đã train (ví dụ `phobert-run-1`). Có thể trỏ trực tiếp vào `champion/`; epoch bundle cũ vẫn tương thích. Python API:

```python
from project.inference import QueryParser

parser = QueryParser.from_checkpoint("project/checkpoints/run-1")
attributes = parser.parse("Tìm cô gái áo trắng tóc đen")
predictions = parser.predict(["Tìm cô gái áo trắng tóc đen", "người đeo kính"])
# predict() thêm confidence của từng thuộc tính; parse() chỉ trả attributes.
```

Đầu ra **mong muốn sau khi train đủ dữ liệu**, không phải kết quả bảo đảm từ dataset demo:

```json
{
  "gender": "female",
  "upper_clothing": {"type": "unknown", "color": "white"},
  "hair": {"color": "black"},
  "accessories": []
}
```

hair.length xuất khi short/long, bỏ khi unknown. Không có checkpoint hợp lệ thì báo lỗi, không khởi tạo head ngẫu nhiên để phục vụ.

## FastAPI

```bash
PERSON_QUERY_CHECKPOINT=project/checkpoints/run-1 \
  python -m uvicorn project.api:app --host 127.0.0.1 --port 8000 --limit-concurrency 8
curl -X POST http://127.0.0.1:8000/parse_query \
  -H 'Content-Type: application/json' \
  -d '{"text":"người đàn ông áo khoác đen đeo kính"}'
```

`PERSON_QUERY_DEVICE` mặc định cpu, có thể đặt cuda. Model được load một lần trong lifespan. GET `/health`: 200 ready, 503 unavailable. POST trả 422 cho input sai/quá context, 503 khi thiếu model, 500 khi inference thất bại. Error envelope có `error.code/message/details`; không echo text. API demo localhost chưa có authentication; production cần gateway/auth và capacity phù hợp. Mỗi worker nạp bản model riêng, cân nhắc RAM/VRAM trước tăng workers.

## Kiểm thử và nguồn kỹ thuật

`python -m pytest project/tests -q` kiểm tra loss/gradient, data validation, metrics tính tay, API errors, checkpoint parity, unknown/multi-accessory, hai epoch training với MLflow, champion selection và migration run cũ. Kết quả ở `TESTING.md` ghi phạm vi thực tế; không thay thế đánh giá dữ liệu thật.

Tài liệu chính thức: [PhoBERT](https://huggingface.co/docs/transformers/model_doc/phobert), [XLM-R](https://huggingface.co/docs/transformers/model_doc/xlm-roberta), [Accelerate](https://huggingface.co/docs/accelerate/package_reference/accelerator).

## Xử lý lỗi đường dẫn và API 503

Chạy lệnh từ thư mục cha của `project/`. Nếu checkpoint báo thiếu `manifest.json` ngay trong thư mục run, hãy cập nhật code: loader tự tìm `champion/` hoặc `best.txt` của run legacy. Không cần ghép `$BEST_EPOCH` (biến này có thể rỗng khi mở Terminal mới).

API cần `PERSON_QUERY_CHECKPOINT` được đặt trong cùng shell hoặc ngay trước lệnh khởi động như ví dụ trên. `/docs` mở được chưa có nghĩa model đã nạp; kiểm tra `/health` phải trả 200 ready. `/` trả 404 là bình thường vì không định nghĩa trang chủ. Xem log startup nếu `/parse_query` trả 503. Không cần train lại khi checkpoint đã tồn tại và chỉ sai đường dẫn.

Nguồn API tracking: [MLflow Tracking APIs](https://mlflow.org/docs/latest/ml/tracking/tracking-api/).
