# Hướng dẫn đánh giá attribute

Tài liệu này mô tả cách đánh giá 21 attributes do G1–G5 sinh ra. Nó tách biệt với
[đánh giá caption](caption-evaluation-guideline.md): caption được review theo các
claim tự do, còn attribute được so sánh theo từng field có cấu trúc.

## 1. Nguồn sự thật và phạm vi

Nguồn tham chiếu mặc định là dòng có cùng `person_id` trong `sample/test/attributes.tsv`.
Mỗi prediction được ghép bằng:

```text
prediction.sample_id = attributes.person_id = identities.person_id
```

Ảnh query phải luôn được xem khi adjudicate mismatch. `attributes.tsv` là human
label tham chiếu, nhưng reviewer không nên kết luận model sai chỉ vì value khác
nếu attribute không nhìn thấy, ảnh query mơ hồ, hoặc label tham chiếu không còn
phù hợp với crop query cụ thể.

Không đánh giá các ảnh gallery khác trong cùng folder identity. Chỉ dùng ảnh
query được chỉ ra bởi `image_path`/`image_id` của sample đang review.

## 2. Kết quả cho từng attribute

Mỗi trong 21 field nhận đúng một kết quả sau:

| Kết quả | Ý nghĩa | Khi dùng |
|---|---|---|
| `correct` | Prediction khớp label tham chiếu và được ảnh hỗ trợ. | Ví dụ `gender: female` ở cả prediction và golden. |
| `incorrect` | Prediction mâu thuẫn rõ ràng với label tham chiếu và ảnh query. | Ví dụ áo blue trong ảnh, prediction `upper_clothing_color: red`. |
| `not_visible` | Ảnh query không đủ bằng chứng để xác nhận field. | Ví dụ crop không thấy giày nhưng model dự đoán footwear. |
| `golden_needs_review` | Golden label mâu thuẫn hoặc không áp dụng rõ ràng cho ảnh query. | Ví dụ golden có bag nhưng ảnh query không có phần thân dưới hoặc không thấy túi. |
| `not_applicable` | Field thực sự không áp dụng và prediction dùng đúng biểu diễn taxonomy. | Ví dụ không có kính và taxonomy cho phép `eyewear: none`. |

`null`, `none`, `unknown`, và `[]` là **giá trị taxonomy**, không phải kết quả
đánh giá. Reviewer cần đánh giá đúng theo nghĩa của từng field:

- `null`: không xác định được.
- `none`: xác định rõ là không có đối tượng đó.
- `unknown`: taxonomy có class unknown cho field đó.
- `[]`: không có mục nào trong multi-label field.

Ví dụ: ảnh không thấy chân, golden `footwear_type: unknown`, prediction
`footwear_type: sandals`. Kết quả nên là `not_visible`, không vội gán
`incorrect`.

## 3. Quy tắc so sánh

### So sánh tự động ban đầu

Với mỗi field, tạo `auto_match` bằng phép so sánh chính xác giữa
`prediction.attributes[field]` và `attributes.tsv[field]` sau khi chuẩn hoá:

- giữ nguyên class code taxonomy, không so sánh bản dịch hiển thị;
- coi thứ tự phần tử trong multi-label arrays là không quan trọng;
- phân biệt `null`, `none`, `unknown`, và `[]`;
- không tự quy đổi màu gần nhau, loại áo gần nhau, hoặc sandals/slippers thành
  match.

`auto_match=false` chỉ là danh sách case cần review; nó chưa khẳng định model sai.

### Adjudication bằng ảnh

Với `auto_match=false`, reviewer kiểm tra ảnh query và chọn một kết quả ở phần 2.

- Dùng `incorrect` khi prediction sai rõ ràng trên ảnh.
- Dùng `not_visible` khi không thể đánh giá đáng tin cậy từ crop.
- Dùng `golden_needs_review` khi prediction hợp ảnh hơn golden, hoặc golden không
  áp dụng cho crop query.
- Không tự thay golden trong lúc review; ghi issue để cập nhật dữ liệu trong một
  bước curate riêng.

## 4. Nhóm ưu tiên review

Mọi field được đếm riêng, nhưng các mismatch dưới đây cần review trước vì ảnh
hưởng nhiều đến mô tả người:

| Ưu tiên | Fields |
|---|---|
| Critical | `gender`, `upper_clothing_type`, `lower_clothing_type`, `upper_clothing_color`, `lower_clothing_color` |
| Important | `age`, `body_build`, `hair_length`, `hair_texture`, `hairstyle`, `hair_color` |
| Detail | `footwear_type`, `clothing_style`, `upper_pattern`, `bag_type`, `bag_color`, `headwear`, `eyewear`, `face_mask`, `other_accessories`, `carried_objects` |

Độ ưu tiên này dùng để sắp xếp review và phân tích lỗi. Nó không thay đổi kết quả
field-level; một `incorrect` vẫn là `incorrect` ở mọi nhóm.

## 5. Schema output đánh giá

JSONL là source of truth. Mỗi dòng đánh giá một `(sample_id, method)` và giữ cả
prediction/golden để review có thể audit lại:

```json
{
  "sample_id": "3",
  "image_id": "private:020799.../first_observation.jpg",
  "filepath": "images/020799.../first_observation.jpg",
  "method": "g4_critic_verifier",
  "attribute_evaluation": {
    "gender": {
      "prediction": "male",
      "golden": "female",
      "auto_match": false,
      "result": "incorrect",
      "note": "The query image visibly shows a female."
    },
    "upper_clothing_color": {
      "prediction": "blue",
      "golden": "blue",
      "auto_match": true,
      "result": "correct",
      "note": ""
    }
  },
  "attribute_summary": {
    "correct": 18,
    "incorrect": 2,
    "not_visible": 1,
    "golden_needs_review": 0,
    "not_applicable": 0
  },
  "reviewer": "human",
  "reviewed_at": "2026-09-23T00:00:00Z"
}
```

`attribute_evaluation` luôn phải có cả 21 fields. `attribute_summary` phải bằng
tổng của 21 kết quả. `image_id` và `filepath` phải được giữ để reviewer mở đúng
crop query.

## 6. Tổng hợp kết quả

Khi đã có review, báo cáo theo từng method và field:

```text
accuracy(field) = (correct + not_applicable) / (correct + incorrect + not_applicable)
coverage(field) = (correct + incorrect + not_applicable) / 200
needs_review(field) = not_visible + golden_needs_review
```

`not_applicable` là một prediction đúng nhưng được giữ riêng để biết field đó
không áp dụng. Không đưa `not_visible` hoặc `golden_needs_review` vào mẫu số
accuracy. Báo cáo hai số này riêng, để tránh một flow có nhiều ảnh không quan sát
được trông có accuracy cao giả tạo.

## 7. Quy trình review

1. Tạo comparison theo `sample_id`; giữ `image_id` và `filepath`.
2. Chạy auto-match 21 fields để lọc mismatch.
3. Reviewer mở ảnh query và adjudicate các mismatch theo bảng kết quả.
4. Ghi JSONL review trước, sau đó mới export Excel/CSV nếu cần nhập hoặc xem tay.
5. Tách các `golden_needs_review` thành danh sách curate; không sửa ngầm
   `attributes.tsv` hoặc overwrite prediction gốc.
