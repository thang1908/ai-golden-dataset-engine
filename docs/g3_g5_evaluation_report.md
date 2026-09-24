# Báo cáo thử nghiệm các phương pháp sinh caption (G3–G5)

Thời điểm tạo: 2026-09-24 07:51 UTC

## 1. Mục tiêu

So sánh ba cách sinh caption từ ảnh để chọn luồng phù hợp cho bước gán nhãn dữ liệu tự động. Mỗi caption được Gemini kiểm tra lại xem có mô tả sai hoặc bịa thêm chi tiết so với ảnh hay không.

## 2. Kết luận nhanh

- Đã đánh giá **583** mẫu; **100.0%** hoàn thành evaluation.
- **G4 là lựa chọn tốt nhất ở batch hiện tại**: 80.2% caption không có chi tiết sai theo ảnh.
- **G5 là phương án thứ hai**: 72.9% caption đúng theo ảnh và không có lỗi evaluation trong batch này.
- G3 hiện chưa phù hợp làm luồng ưu tiên vì tỷ lệ caption có chi tiết sai cao hơn đáng kể.
- Khuyến nghị: tiếp tục dùng **G4** làm baseline để mở rộng thử nghiệm; lấy thêm một mẫu ngẫu nhiên để người review xác nhận kết quả của LLM-as-judge trước khi đưa vào production.

## 3. Ba phương pháp đã thử

Các workflow G3–G5 là **thiết kế/điều chỉnh cho project này**, lấy cảm hứng từ các hướng nghiên cứu bên dưới; không phải triển khai nguyên văn của một paper.

### G3 – đồng thuận nhiều lượt (multi-run consensus)

1. Cùng một ảnh được VLM quan sát độc lập qua nhiều lượt (Run 1…Run N). Mỗi lượt trả về caption và thuộc tính có cấu trúc.
2. Node **Consensus** đối chiếu các kết quả và chọn các thuộc tính xuất hiện nhất quán; các chi tiết mâu thuẫn hoặc thiếu bằng chứng có thể bị bỏ qua.
3. Caption Generator viết caption cuối từ bộ thuộc tính đã đồng thuận, thay vì viết trực tiếp từ một lần quan sát duy nhất.
4. **Nguồn cảm hứng:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models — Wang et al., 2022](https://arxiv.org/abs/2203.11171). Paper này lấy nhiều đường suy luận rồi chọn câu trả lời nhất quán nhất; ở đây ý tưởng được chuyển thành nhiều lượt quan sát ảnh, không phải dùng nguyên phương pháp Chain-of-Thought của paper.

### G4 – tạo, phản biện, kiểm tra (generate → critic → verifier)

1. **Generator** đọc ảnh và tạo `draft caption + attributes`.
2. **Critic** đọc lại ảnh cùng bản nháp, nêu ra chi tiết có thể sai, bịa thêm hoặc còn thiếu bằng chứng.
3. **Verifier** nhận ảnh, bản nháp và nhận xét của Critic, rồi quyết định `Accept` hoặc `Reject`.
4. Nếu bị `Reject`, phản hồi được đưa vào lượt **Regenerate** để Generator tạo bản mới. Luồng lặp đến khi được chấp nhận hoặc đạt số lần thử tối đa.
5. **Nguồn cảm hứng:** [Self-Refine: Iterative Refinement with Self-Feedback — Madaan et al., 2023](https://arxiv.org/abs/2303.17651). Paper đề xuất sinh bản nháp, tạo feedback và tinh chỉnh lặp; project này điều chỉnh thành hai vai trò riêng Critic/Verifier có kiểm tra trực tiếp ảnh.

### G5 – hỏi đáp để tinh chỉnh (question-guided refinement)

1. Generator tạo `draft caption + attributes` từ ảnh.
2. **Question Generator** biến các claim trong bản nháp thành câu hỏi kiểm chứng, ví dụ: “Người đó có tóc búi không?”, “Áo có màu trắng không?”.
3. **Image Answerer** chỉ nhìn ảnh để trả lời từng câu hỏi; không lấy câu trả lời từ caption nháp.
4. **Compare/Router** so sánh claim của caption với câu trả lời từ ảnh: khớp thì `Accept`, mâu thuẫn/không đủ bằng chứng thì gửi lý do về Generator để `Refine`.
5. **Nguồn cảm hứng:** [QACE: Asking Questions to Evaluate an Image Caption — Lee et al., 2021](https://aclanthology.org/2021.findings-emnlp.395/). QACE dùng Question Generation + Question Answering để kiểm tra nội dung caption với ảnh; project này điều chỉnh cơ chế đó từ **evaluation** sang một vòng **tinh chỉnh caption khi sinh**.

## 4. Cách đánh giá — một ví dụ từ đầu đến cuối

Phần này minh hoạ **một case giả định** để người đọc thấy rõ hệ thống đánh giá gì và kết quả đi vào Excel như thế nào.

### Case minh hoạ

Giả sử ảnh có một phụ nữ trẻ, tóc đen búi, mặc áo trắng tay ngắn và váy trắng dài. Golden dataset của ảnh này đã có 21 nhãn, trong đó có `gender = female`, `upper_clothing_color = white`, `hairstyle = bun`, `footwear_type = sandals`.

G4 hoặc G5 sinh caption sau:

`A young adult female with dark hair in a bun wears a white short-sleeve top and a long white skirt.`

Hệ thống không hỏi một câu chung chung là “caption có tốt không?”. Nó chấm **hai việc khác nhau**, rồi lưu cả hai để người review xem được lý do.

### Bước 1 — Caption có nói sai về ảnh không?

**Đưa vào Gemini:** ảnh gốc + caption vừa sinh. Không đưa golden attributes vào bước này.

Gemini kiểm tra từng claim mà caption đã nói: “female”, “young adult”, “dark hair”, “bun”, “white top”, “long white skirt”. Nếu các claim này phù hợp với ảnh, kết quả là:

```json
{"caption_is_correct": true, "note": "Caption mô tả đúng các chi tiết quan sát được trong ảnh."}
```

Điểm quan trọng: caption **không cần kể hết ảnh** mới được đúng. Ví dụ caption rút gọn `A woman wears a white top.` vẫn được `true` nếu ảnh đúng như vậy; hệ thống không phạt vì caption chưa nói đến tóc, váy hay giày.

Ngược lại, nếu model sinh `... She wears black boots.`, Gemini đối chiếu trực tiếp với ảnh. Khi ảnh không có/không cho thấy giày đen, kết quả Bước 1 là:

```json
{"caption_is_correct": false, "note": "Caption thêm chi tiết giày đen không được ảnh hỗ trợ."}
```

Nói ngắn gọn: **Bước 1 trả lời câu hỏi “model có bịa hoặc nói sai điều gì không?”**

### Bước 2 — Những thuộc tính caption đã nói có khớp bộ nhãn chuẩn không?

**Đưa vào Gemini:** caption + 21 golden attributes của đúng ảnh đó. Bước này không đưa ảnh vào, vì mục tiêu là đối chiếu cách caption diễn đạt với bộ nhãn chuẩn đã có.

Với từng thuộc tính, hệ thống luôn trả hai thông tin: `mentioned` (caption có nói đến không) và `is_correct` (nếu có nói thì có khớp không). Ví dụ từ caption ở trên:

| Thuộc tính | Golden label | Caption có nhắc? | Kết quả | Lý do |
| --- | --- | --- | --- | --- |
| Giới tính | female | Có | `true` | Caption nói `female`, khớp nhãn. |
| Màu áo | white | Có | `true` | Caption nói `white ... top`, khớp nhãn. |
| Kiểu tóc | bun | Có | `true` | Caption nói `hair in a bun`, khớp nhãn. |
| Loại giày / dép | sandals | Không | `null` | Caption không nói về giày; đây là trung tính, **không phải lỗi**. |
| Màu túi | null | Không | `null` | Caption không nói về túi; không có gì để chấm đúng/sai. |

Nếu caption sai thêm `black boots`, riêng hàng `Loại giày / dép` sẽ thành: `mentioned = true`, `is_correct = false`, vì caption đã khẳng định một thông tin khác golden label `sandals`.

Nói ngắn gọn: **Bước 2 trả lời câu hỏi “khi caption có nêu một thuộc tính, nó có khớp nhãn chuẩn không?”**. Giá trị `null` chỉ nghĩa là caption không nhắc tới thuộc tính đó; nó không làm giảm điểm caption ở Bước 1.

### Bước 3 — Người review nhìn thấy gì?

Sau khi hai bước hoàn thành, hệ thống ghi ngay một dòng JSONL rồi xuất sang Excel/Dashboard. Một người review sẽ thấy: ảnh, caption tiếng Anh/Việt, verdict tổng `caption_is_correct`, note tiếng Việt và 21 hàng thuộc tính như bảng trên.

Ví dụ dòng tóm tắt trong sheet **Caption**: `Case 123 | G4 | caption_is_correct = true | 3 thuộc tính khớp | 0 thuộc tính sai | 18 không nhắc`.
Trong sheet **Thuộc tính**, người review có thể mở riêng hàng `Loại giày / dép` để thấy `mentioned = false`, `is_correct = null`; đây là lý do số lượng `null` cao không đồng nghĩa caption sai.

Khi người review không đồng ý với AI judge, họ sửa verdict trên Dashboard/Excel. Human override được lưu riêng, không ghi đè kết quả gốc của Gemini; vì vậy sau này vẫn audit được AI đã chấm gì và người review đã sửa gì.

### Quy ước khi đọc các bảng kết quả

- **Caption đúng theo ảnh**: tỷ lệ `caption_is_correct = true` ở Bước 1.
- **Thuộc tính khớp nhãn**: chỉ tính trong những thuộc tính caption có nhắc; mẫu số không bao gồm `null`.
- **Null / không nhắc**: caption không nói đến thuộc tính đó hoặc chưa thể kết luận; không phải một lỗi mô tả.
- **Thời gian đánh giá**: thời gian Gemini thực hiện việc kiểm tra, không phải tổng thời gian sinh caption.
- Lưu ý: đây là kết quả **LLM-as-judge**. Chưa có chỉnh sửa của người review trong batch này, vì vậy chưa nên gọi là độ chính xác ground truth.

## 5. Kết quả chính

| Luồng | Số mẫu | Evaluation thành công | Lỗi | Caption đúng theo ảnh | Thuộc tính được nêu khớp nhãn | P50 thời gian chấm |
| --- | --- | --- | --- | --- | --- | --- |
| G3 | 199 | 100.0% | 0 | 22.6% | 64.4% | 6,592 ms |
| G4 | 192 | 100.0% | 0 | 80.2% | 60.6% | 4,893 ms |
| G5 | 192 | 100.0% | 0 | 72.9% | 61.7% | 4,708 ms |

## 6. Chi tiết kết quả caption

| Luồng | Caption đúng theo ảnh | Caption có chi tiết sai |
| --- | --- | --- |
| G3 | 45 | 154 |
| G4 | 154 | 38 |
| G5 | 140 | 52 |

## 7. Chi tiết thuộc tính caption

| Luồng | Khớp nhãn | Không khớp nhãn | Không nhắc (null) |
| --- | --- | --- | --- |
| G3 | 1719 | 949 | 1511 |
| G4 | 970 | 630 | 2432 |
| G5 | 1160 | 720 | 2152 |

## 8. Phân tích chi tiết 21 thuộc tính

Bảng dưới đây cho biết caption đã nhắc tới từng thuộc tính bao nhiêu lần. Tỷ lệ khớp chỉ tính các lần caption có nêu thuộc tính đó; “không nhắc” không phải lỗi.

### G3

| Thuộc tính | Caption có nhắc | Khớp nhãn | Không khớp | Không nhắc | Tỷ lệ khớp khi có nhắc |
| --- | --- | --- | --- | --- | --- |
| Độ tuổi | 199 | 116 | 83 | 0 | 58.3% |
| Giới tính | 199 | 171 | 28 | 0 | 85.9% |
| Dáng người | 194 | 87 | 107 | 5 | 44.8% |
| Loại áo | 198 | 124 | 74 | 1 | 62.6% |
| Màu áo | 197 | 147 | 50 | 2 | 74.6% |
| Loại quần / váy | 171 | 21 | 150 | 28 | 12.3% |
| Màu quần / váy | 169 | 124 | 45 | 30 | 73.4% |
| Phong cách trang phục | 124 | 119 | 5 | 75 | 96.0% |
| Họa tiết áo | 195 | 178 | 17 | 4 | 91.3% |
| Loại giày / dép | 150 | 22 | 128 | 49 | 14.7% |
| Loại túi | 21 | 15 | 6 | 178 | 71.4% |
| Màu túi | 21 | 17 | 4 | 178 | 81.0% |
| Mũ / nón | 14 | 9 | 5 | 185 | 64.3% |
| Kính | 11 | 3 | 8 | 188 | 27.3% |
| Khẩu trang | 6 | 2 | 4 | 193 | 33.3% |
| Phụ kiện khác | 14 | 6 | 8 | 185 | 42.9% |
| Độ dài tóc | 197 | 145 | 52 | 2 | 73.6% |
| Chất tóc | 197 | 189 | 8 | 2 | 95.9% |
| Kiểu tóc | 160 | 29 | 131 | 39 | 18.1% |
| Màu tóc | 197 | 185 | 12 | 2 | 93.9% |
| Vật mang theo | 34 | 10 | 24 | 165 | 29.4% |

### G4

| Thuộc tính | Caption có nhắc | Khớp nhãn | Không khớp | Không nhắc | Tỷ lệ khớp khi có nhắc |
| --- | --- | --- | --- | --- | --- |
| Độ tuổi | 81 | 50 | 31 | 111 | 61.7% |
| Giới tính | 90 | 85 | 5 | 102 | 94.4% |
| Dáng người | 16 | 7 | 9 | 176 | 43.8% |
| Loại áo | 190 | 110 | 80 | 2 | 57.9% |
| Màu áo | 181 | 122 | 59 | 11 | 67.4% |
| Loại quần / váy | 164 | 65 | 99 | 28 | 39.6% |
| Màu quần / váy | 162 | 112 | 50 | 30 | 69.1% |
| Phong cách trang phục | 13 | 12 | 1 | 179 | 92.3% |
| Họa tiết áo | 78 | 65 | 13 | 114 | 83.3% |
| Loại giày / dép | 127 | 17 | 110 | 65 | 13.4% |
| Loại túi | 34 | 14 | 20 | 158 | 41.2% |
| Màu túi | 26 | 17 | 9 | 166 | 65.4% |
| Mũ / nón | 13 | 6 | 7 | 179 | 46.2% |
| Kính | 7 | 4 | 3 | 185 | 57.1% |
| Khẩu trang | 6 | 0 | 6 | 186 | 0.0% |
| Phụ kiện khác | 8 | 5 | 3 | 184 | 62.5% |
| Độ dài tóc | 137 | 105 | 32 | 55 | 76.6% |
| Chất tóc | 5 | 1 | 4 | 187 | 20.0% |
| Kiểu tóc | 55 | 22 | 33 | 137 | 40.0% |
| Màu tóc | 169 | 144 | 25 | 23 | 85.2% |
| Vật mang theo | 38 | 7 | 31 | 154 | 18.4% |

### G5

| Thuộc tính | Caption có nhắc | Khớp nhãn | Không khớp | Không nhắc | Tỷ lệ khớp khi có nhắc |
| --- | --- | --- | --- | --- | --- |
| Độ tuổi | 144 | 89 | 55 | 48 | 61.8% |
| Giới tính | 140 | 124 | 16 | 52 | 88.6% |
| Dáng người | 17 | 12 | 5 | 175 | 70.6% |
| Loại áo | 190 | 116 | 74 | 2 | 61.1% |
| Màu áo | 182 | 127 | 55 | 10 | 69.8% |
| Loại quần / váy | 157 | 78 | 79 | 35 | 49.7% |
| Màu quần / váy | 154 | 103 | 51 | 38 | 66.9% |
| Phong cách trang phục | 63 | 55 | 8 | 129 | 87.3% |
| Họa tiết áo | 97 | 81 | 16 | 95 | 83.5% |
| Loại giày / dép | 130 | 14 | 116 | 62 | 10.8% |
| Loại túi | 66 | 15 | 51 | 126 | 22.7% |
| Màu túi | 49 | 20 | 29 | 143 | 40.8% |
| Mũ / nón | 14 | 4 | 10 | 178 | 28.6% |
| Kính | 6 | 5 | 1 | 186 | 83.3% |
| Khẩu trang | 4 | 2 | 2 | 188 | 50.0% |
| Phụ kiện khác | 11 | 4 | 7 | 181 | 36.4% |
| Độ dài tóc | 162 | 117 | 45 | 30 | 72.2% |
| Chất tóc | 5 | 2 | 3 | 187 | 40.0% |
| Kiểu tóc | 40 | 22 | 18 | 152 | 55.0% |
| Màu tóc | 179 | 157 | 22 | 13 | 87.7% |
| Vật mang theo | 70 | 13 | 57 | 122 | 18.6% |

## 9. Nhận xét trọng tâm từ thuộc tính

- **G3**: nhóm cần hạn chế mô tả khi ảnh chưa rõ là Loại quần / váy (12.3%, 171 lần có kết luận), Loại giày / dép (14.7%, 150 lần có kết luận), Kiểu tóc (18.1%, 160 lần có kết luận).
  Nhóm ổn định hơn trong batch này: Phong cách trang phục (96.0%, 124 lần có kết luận), Chất tóc (95.9%, 197 lần có kết luận), Màu tóc (93.9%, 197 lần có kết luận).
- **G4**: nhóm cần hạn chế mô tả khi ảnh chưa rõ là Loại giày / dép (13.4%, 127 lần có kết luận), Vật mang theo (18.4%, 38 lần có kết luận), Loại quần / váy (39.6%, 164 lần có kết luận).
  Nhóm ổn định hơn trong batch này: Giới tính (94.4%, 90 lần có kết luận), Màu tóc (85.2%, 169 lần có kết luận), Họa tiết áo (83.3%, 78 lần có kết luận).
- **G5**: nhóm cần hạn chế mô tả khi ảnh chưa rõ là Loại giày / dép (10.8%, 130 lần có kết luận), Vật mang theo (18.6%, 70 lần có kết luận), Loại túi (22.7%, 66 lần có kết luận).
  Nhóm ổn định hơn trong batch này: Giới tính (88.6%, 140 lần có kết luận), Màu tóc (87.7%, 179 lần có kết luận), Phong cách trang phục (87.3%, 63 lần có kết luận).

Nhận xét chung: giày/dép, vật mang theo, túi và một số chi tiết về quần/váy là các nhóm dễ tạo chi tiết sai. Prompt nên yêu cầu model chỉ nêu các thuộc tính này khi quan sát rõ; nếu không, caption ngắn hơn nhưng đúng sẽ tốt hơn.

## 10. So sánh để quyết định áp dụng

Bảng này dùng cùng tinh thần với báo cáo so sánh mô hình: mọi luồng được chấm trên batch hiện có. Tuy nhiên, metric được đặt tên theo đúng ý nghĩa của luồng evaluation hiện tại.

| Luồng | Mẫu chấm thành công | Cặp thuộc tính có thể kiểm tra | Caption đúng theo ảnh | Độ bao phủ thuộc tính | Khớp nhãn khi có nhắc | P50 thời gian chấm | Khuyến nghị |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G3 | 199 | 4179 | 22.6% | 63.8% | 64.4% | 6,592 ms | Không ưu tiên; cần giảm chi tiết sai. |
| G4 | 192 | 4032 | 80.2% | 39.7% | 60.6% | 4,893 ms | Baseline ưu tiên cho production thử nghiệm. |
| G5 | 192 | 4032 | 72.9% | 46.6% | 61.7% | 4,708 ms | Phương án dự phòng; tiếp tục tối ưu router/QA. |

### 10.1. So sánh chi tiết theo từng thuộc tính

Bảng dưới đặt ba luồng trên **cùng 185 case đã evaluation thành công ở cả G3, G4 và G5**.

**Cách đọc một ô:** `19/122 (15,6%)` nghĩa là model đã đưa ra kết luận về thuộc tính đó 122 lần và đúng 19 lần. Ví dụ ở hàng *Loại giày / dép*, nếu model không nói gì về giày thì không tính là sai; bảng chỉ kiểm tra các lần model đã chủ động mô tả giày.

| STT | Thuộc tính | G3: đúng / đã mô tả | G4: đúng / đã mô tả | G5: đúng / đã mô tả | Kết luận để áp dụng |
| --- | --- | --- | --- | --- | --- |
| 1 | Độ tuổi | 106/185 (57.3%) | 46/75 (61.3%) | 87/141 (61.7%) | Ba luồng gần tương đương. |
| 2 | Giới tính | 158/185 (85.4%) | 80/85 (94.1%) | 121/136 (89.0%) | G4 tốt nhất ở thuộc tính này. |
| 3 | Dáng người | 80/180 (44.4%) | 7/15 (46.7%) | 12/17 (70.6%) | G5 cao nhất nhưng ít case; chỉ tham khảo. |
| 4 | Loại áo | 113/184 (61.4%) | 104/183 (56.8%) | 113/184 (61.4%) | Ba luồng gần tương đương. |
| 5 | Màu áo | 136/183 (74.3%) | 116/174 (66.7%) | 124/176 (70.5%) | G3 tốt nhất ở thuộc tính này. |
| 6 | Loại quần / váy | 18/161 (11.2%) | 61/157 (38.9%) | 75/154 (48.7%) | Cả ba còn yếu; chưa tự động gán nhãn. |
| 7 | Màu quần / váy | 114/159 (71.7%) | 105/155 (67.7%) | 101/151 (66.9%) | Ba luồng gần tương đương. |
| 8 | Phong cách trang phục | 108/113 (95.6%) | 12/13 (92.3%) | 54/62 (87.1%) | G3 tốt nhất ở thuộc tính này. |
| 9 | Họa tiết áo | 165/181 (91.2%) | 62/74 (83.8%) | 80/96 (83.3%) | G3 tốt nhất ở thuộc tính này. |
| 10 | Loại giày / dép | 19/141 (13.5%) | 17/123 (13.8%) | 14/128 (10.9%) | Cả ba còn yếu; chưa tự động gán nhãn. |
| 11 | Loại túi | 14/20 (70.0%) | 12/29 (41.4%) | 15/66 (22.7%) | G3 cao nhất nhưng ít case; chỉ tham khảo. |
| 12 | Màu túi | 16/20 (80.0%) | 14/23 (60.9%) | 20/49 (40.8%) | G3 cao nhất nhưng ít case; chỉ tham khảo. |
| 13 | Mũ / nón | 9/14 (64.3%) | 6/13 (46.2%) | 4/14 (28.6%) | G3 cao nhất nhưng ít case; chỉ tham khảo. |
| 14 | Kính | 3/11 (27.3%) | 4/7 (57.1%) | 5/6 (83.3%) | G5 cao nhất nhưng ít case; chỉ tham khảo. |
| 15 | Khẩu trang | 2/6 (33.3%) | 0/6 (0.0%) | 2/4 (50.0%) | Ít case được mô tả; chỉ tham khảo. |
| 16 | Phụ kiện khác | 6/13 (46.2%) | 5/8 (62.5%) | 4/8 (50.0%) | G4 cao nhất nhưng ít case; chỉ tham khảo. |
| 17 | Độ dài tóc | 135/183 (73.8%) | 102/132 (77.3%) | 115/157 (73.2%) | Ba luồng gần tương đương. |
| 18 | Chất tóc | 175/183 (95.6%) | 1/5 (20.0%) | 2/5 (40.0%) | G3 tốt nhất ở thuộc tính này. |
| 19 | Kiểu tóc | 28/151 (18.5%) | 21/53 (39.6%) | 21/39 (53.8%) | Cả ba còn yếu; chưa tự động gán nhãn. |
| 20 | Màu tóc | 172/183 (94.0%) | 139/163 (85.3%) | 152/173 (87.9%) | G3 tốt nhất ở thuộc tính này. |
| 21 | Vật mang theo | 10/33 (30.3%) | 7/37 (18.9%) | 13/68 (19.1%) | Cả ba còn yếu; chưa tự động gán nhãn. |

**Cách đọc để ra quyết định:** G4 có tỷ lệ caption đúng theo ảnh cao nhất trong batch nên là baseline đề xuất. G5 đứng thứ hai và phù hợp để tiếp tục thử khi cần cơ chế hỏi–đáp/kiểm chứng. G3 không nên mở rộng trước khi xử lý các nhóm lỗi có tỷ lệ sai cao.

**Không đưa Micro Accuracy/Macro Accuracy/Exact-match 21 thuộc tính vào bảng này.** Những chỉ số đó chỉ hợp lệ khi lấy **21 generated attributes đã chuẩn hoá** của mỗi case so sánh trực tiếp với 21 golden attributes, nên mọi trường đều có support bằng nhau. Luồng hiện tại chủ đích chấm **caption**: một thuộc tính không được caption nhắc tới sẽ là `null` (trung tính), không phải prediction sai. Gọi tỷ lệ này là “attribute accuracy 21 trường” sẽ gây hiểu nhầm.

Nếu cần báo cáo theo đúng mẫu VLLM vs Qwen3.5 4B (Micro/Macro/Exact 21/21), bước tiếp theo là chạy thêm một evaluator riêng: `generated structured attributes + golden attributes` → chuẩn hoá taxonomy → so sánh exact match từng trường. Khi đó có thể tạo bảng model/method theo cohort chung một cách hợp lệ.

## 11. Giới hạn và bước tiếp theo

1. Gemini đang đóng vai trò judge, do đó metric là tín hiệu so sánh giữa các luồng, chưa thay thế đánh giá của con người.
2. Chưa có human override nào trong batch này. Nên review ngẫu nhiên các caption đúng/sai của G4 và G5 để kiểm tra judge có lệch hay không.
3. Nếu human review xác nhận xu hướng này, dùng G4 làm baseline và tiếp tục tối ưu prompt/loop để giảm các chi tiết sai còn lại.

## Phụ lục: hoạt động human review

| Luồng | Dòng review đã lưu | Thuộc tính đã chỉnh |
| --- | --- | --- |
| G3 | 0 | 0 |
| G4 | 0 | 0 |
| G5 | 0 | 0 |

## Phụ lục: lỗi evaluation

Không có bản ghi G3–G5 mới nhất nào có trạng thái lỗi.

## Phụ lục kỹ thuật: chạy lại báo cáo

```bash
python tools/build_g3_g5_evaluation_report.py
```

Chạy lệnh này sau evaluation hoặc sau khi lưu chỉnh sửa trên dashboard để cập nhật số liệu.
