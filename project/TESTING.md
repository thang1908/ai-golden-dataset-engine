# Kết quả kiểm thử hiện tại

Môi trường `.venv` của workspace, Python 3.11.9, macOS arm64, CPU. Phiên bản dependency trực tiếp ghi trong `requirements-tested.txt`.

## MLflow và champion-only

Lệnh: `HF_HOME=/private/tmp/person-query-hf HF_HUB_OFFLINE=1 ACCELERATE_USE_CPU=true .venv/bin/python -m pytest project/tests -q`

Kết quả: **19 passed**, 10.66 giây. Hai deprecation warnings từ Starlette/AnyIO, không có failure.

- Training hai epoch bằng encoder RoBERTa nhỏ, Underthesea thực và SQLite MLflow thật. Metrics/loss mỗi epoch có history, run kết thúc FINISHED, params/signature đúng.
- Output chỉ có `champion/` và một file weights. Không có epoch/state/best/last snapshots, không có model weights trong artifact MLflow.
- Chọn champion theo exact-match rồi loss; epoch thấp điểm hoặc bằng hoàn toàn không gọi serializer. Lỗi ghi candidate hoặc lỗi swap giữ nguyên champion cũ.
- Migration run legacy giữ model tốt nhất, nhập đầy đủ metrics rồi xóa epoch/state dư.
- Các test model/gradient, dataset, segmentation, Unicode, metrics, API và checkpoint reload tiếp tục qua.

## Run PhoBERT thực của người dùng

Đã chuyển `project/checkpoints/phobert-run-1` sang `champion/`, giữ weights của epoch 5. Dung lượng từ khoảng 10 GB còn **517 MB**. Không retrain hoặc thay đổi weights.

MLflow run `2249bfd8ae814df098d7b607be511d7e` trong `project/mlflow.db` có trạng thái FINISHED; lịch sử validation_loss gồm các bước 1, 2, 3, 4, 5. Artifacts chỉ gồm config.json, dataset_signature.json, champion.json. Metadata champion tham chiếu đúng run ID.

FastAPI TestClient với checkpoint thật và PERSON_QUERY_CHECKPOINT trỏ thư mục run: GET /health trả 200 ready, POST /parse_query trả 200 JSON.

## Giới hạn

Checkpoint thật được train trên dữ liệu mẫu, chưa chứng minh độ chính xác NLP thực tế. Chưa kiểm thử CUDA mixed precision, multi-GPU hoặc MLflow remote server. Test lần này chạy Python 3.11; Python 3.10+ là mục tiêu dependency. Không lưu optimizer/RNG state và không hỗ trợ --resume theo chính sách champion-only. MLflow UI CLI đã kiểm tra tùy chọn; UI không được để chạy nền.
