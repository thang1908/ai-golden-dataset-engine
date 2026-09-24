# Hướng dẫn đánh giá caption

Tài liệu này là rubric cho **human review** caption sinh bởi G1–G5. Đây chưa phải
là AI evaluation, score tự động, hay một thay đổi vào luồng generation.

## 1. Nguyên tắc chung

Đánh giá caption theo những gì caption **khẳng định**, không theo số lượng trường
taxonomy mà caption đã đề cập.

- Một claim phải quan sát được và đúng với ảnh query.
- Không phạt caption vì thiếu một chi tiết nhìn thấy, chẳng hạn không nêu giày,
  tóc, túi hoặc phụ kiện.
- Phạt thông tin bịa đặt (hallucination): caption khẳng định người, trang phục,
  vật thể, hành động hoặc bối cảnh không có trong ảnh.
- Dùng ảnh query có `is_query=1` làm nguồn sự thật. Không dùng các ảnh khác cùng
  `person_id`, và không coi golden caption là bằng chứng thay thế ảnh.
- Dùng dòng cùng `person_id` trong `attributes.tsv` làm structured reference để
  kiểm tra nhanh các claim về người, trang phục, màu sắc, tóc, giày dép, phụ kiện
  và đồ vật mang theo. Ảnh query vẫn quyết định cuối cùng khi label và crop mâu
  thuẫn hoặc chi tiết không nhìn rõ.
- Khi chi tiết không nhìn rõ hoặc không thể xác định đáng tin cậy, không chấm lỗi
  caption vì không nêu chi tiết đó. Nếu caption vẫn khẳng định một chi tiết trái
  với phần quan sát rõ ràng của ảnh, đó là lỗi.

Các loại claim cần kiểm tra gồm: số người, giới tính, tuổi tương đối, vóc dáng,
trang phục, màu sắc, tóc, giày dép, phụ kiện và đồ vật mang theo.

## 2. Đánh giá caption có hỗ trợ attribute

Mỗi caption được đánh giá với ba input cùng một `sample_id`:

```text
Ảnh query + attributes.tsv (golden attributes) + caption cần đánh giá
```

Quy trình:

```text
Golden attributes → structured facts tham chiếu
Caption → các claim quan sát được
Claim + facts + ảnh query → verdict cho từng claim → Level 0–3
```

Ví dụ golden attributes có `gender: female`, `upper_clothing_color: blue`,
`lower_clothing_color: black`. Caption là `A man wearing a blue shirt and black
trousers.`

- `man` mâu thuẫn với `gender: female`; kiểm tra ảnh query. Nếu ảnh là nữ, đây là
  Major Error.
- `blue shirt` và `black trousers` khớp facts và ảnh query.
- Kết quả caption là Level 1 vì có một Major Error, dù các claim khác đúng.

Attribute label không được thay thế việc xem ảnh. Nếu golden ghi `bag_type` nhưng
crop query không thấy túi, reviewer dùng ảnh để adjudicate, không tự động kết luận
caption sai. Các case này nên được ghi là `ambiguous` hoặc chuyển sang
`golden_needs_review` trong đánh giá attribute.

## 3. Mức độ nghiêm trọng theo attribute

| Nhóm | Attribute / nội dung | Mặc định khi sai | Khi nào tăng thành Major |
|---|---|---:|---|
| A — Critical | `gender`; số người | Major | Luôn Major nếu có bằng chứng rõ ràng. |
| A — Critical | `upper_clothing_type`, `lower_clothing_type` | Major | Luôn Major nếu làm sai trang phục chính. |
| A — Critical | `upper_clothing_color`, `lower_clothing_color` | Major | Màu chính khác rõ rệt, ví dụ blue → red. |
| B — Important | `age` | Minor | Sai cực đoan, ví dụ elderly → child. |
| B — Important | `body_build` | Minor | Làm thay đổi nhận dạng hoặc bị sai cực đoan. |
| B — Important | `hair_length`, `hair_texture`, `hairstyle`, `hair_color` | Minor | Claim tóc là đặc điểm nhận dạng và sai rõ rệt, ví dụ black → blonde. |
| C — Detail | `footwear_type`, `upper_pattern`, `clothing_style` | Minor | Không tự động tăng mức nếu không làm sai nội dung chính. |
| C — Detail | `bag_type`, `bag_color`, `headwear`, `eyewear`, `face_mask`, `other_accessories` | Minor | Major nếu bịa một vật thể lớn hoặc nổi bật không tồn tại, ví dụ backpack. |
| C — Detail | `carried_objects` | Minor | Major nếu bịa một đối tượng lớn hoặc làm thay đổi đáng kể nội dung ảnh, ví dụ holding a baby. |

Ví dụ: sandals → slippers thường là Minor; một chiếc áo blue bị mô tả thành red
hoặc một T-shirt bị mô tả thành dress là Major.

## 4. Quy tắc hallucination

Hallucination là claim không được ảnh hỗ trợ. Ví dụ ảnh chỉ có một người đàn ông
đứng trên đường, nhưng caption nói anh ấy cầm điện thoại và đeo ba lô khi hai vật
đó không thấy trong ảnh.

- Hallucination nhỏ, như một chiếc đồng hồ hoặc màu túi sai, là Minor.
- Hallucination lớn, như thêm ba lô, xe đạp, em bé, một người khác, hoặc hành động
  không có trong ảnh, là Major.
- Không suy diễn identity, quan hệ, ý định, địa điểm cụ thể hoặc chi tiết bị che
  khuất chỉ vì chúng có vẻ hợp lý.

## 5. Mức kết quả cuối cùng

| Level | Tên | Điều kiện |
|---:|---|---|
| 3 | Correct | Mọi claim đều đúng hoặc được ảnh hỗ trợ; không có hallucination. Thiếu chi tiết vẫn là Correct. |
| 2 | Minor Error | Nội dung chính đúng; chỉ có lỗi chi tiết nhỏ hoặc claim khó phân biệt. Không có Major Error. |
| 1 | Major Error | Có ít nhất một Major Error, nhưng caption vẫn còn mô tả đúng một phần đáng kể của ảnh. |
| 0 | Invalid | Caption về bản chất mô tả sai ảnh: sai người/số người/nội dung chính, hoặc có nhiều Major Error làm caption không còn hữu ích. |

Khi caption có nhiều lỗi, chọn **mức thấp nhất phù hợp với toàn bộ caption**. Một
lỗi Minor không được làm giảm Level 3 xuống Level 1. Một Major Error tối thiểu là
Level 1; chỉ dùng Level 0 khi caption sai toàn diện, không chỉ vì đếm số lỗi.

## 6. Ví dụ

Ảnh: một người nam, áo xanh, quần short đen, đi sandals.

| Caption | Kết quả | Lý do |
|---|---:|---|
| `A man wearing a blue shirt and black shorts.` | 3 | Các claim đúng. Việc không nói sandals không phải lỗi. |
| `A man wearing a blue shirt, black shorts, and slippers.` | 2 | Sai chi tiết footwear. |
| `A woman wearing a blue shirt and black shorts.` | 1 | Sai gender, nhưng các phần chính khác vẫn đúng. |
| `Two children playing football with a dog in a park.` | 0 | Sai hoàn toàn số người, tuổi, hành động, vật thể và bối cảnh. |

## 7. Cách dùng khi review

Với từng dòng trong `captions_merged.xlsx`:

1. Ghép caption, ảnh query và dòng `attributes.tsv` bằng `sample_id`/`person_id`.
2. Mở ảnh ở cột `image_path` và chỉ xét ảnh query đó.
3. Dùng attributes làm facts tham chiếu, sau đó đọc caption của từng flow độc lập
   với golden caption.
4. Liệt kê các claim sai hoặc hallucination ngắn gọn.
5. Chọn một Level từ 0 đến 3 theo bảng trên.
6. Nếu ảnh quá mờ/che khuất, ghi chú `ambiguous` thay vì suy đoán lỗi.

Golden caption là tham chiếu hữu ích để đọc nhanh, nhưng ảnh query luôn là nguồn
quyết định cuối cùng.
