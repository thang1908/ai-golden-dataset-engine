#!/usr/bin/env python3
"""Build a concise, reproducible Markdown report for G3–G5 evaluation results."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "evaluation" / "reviews_v3.jsonl"
DEFAULT_OVERRIDES = ROOT / "output" / "dashboard" / "review_overrides_v3.json"
DEFAULT_OUTPUT = ROOT / "docs" / "g3_g5_evaluation_report.md"
METHODS = ("g3", "g4", "g5")
ATTRIBUTE_RESULTS = ("true", "false", "null")
CAPTION_RESULTS = ("true", "false")
ATTRIBUTE_LABELS = (
    ("age", "Độ tuổi"), ("gender", "Giới tính"), ("body_build", "Dáng người"),
    ("upper_clothing_type", "Loại áo"), ("upper_clothing_color", "Màu áo"),
    ("lower_clothing_type", "Loại quần / váy"), ("lower_clothing_color", "Màu quần / váy"),
    ("clothing_style", "Phong cách trang phục"), ("upper_pattern", "Họa tiết áo"),
    ("footwear_type", "Loại giày / dép"), ("bag_type", "Loại túi"), ("bag_color", "Màu túi"),
    ("headwear", "Mũ / nón"), ("eyewear", "Kính"), ("face_mask", "Khẩu trang"),
    ("other_accessories", "Phụ kiện khác"), ("hair_length", "Độ dài tóc"),
    ("hair_texture", "Chất tóc"), ("hairstyle", "Kiểu tóc"),
    ("hair_color", "Màu tóc"), ("carried_objects", "Vật mang theo"),
)


def latest_rows(path: Path, methods: tuple[str, ...]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, method = row.get("sample_id"), row.get("method")
            if not isinstance(sample_id, str) or not isinstance(method, str):
                raise ValueError(f"Invalid sample_id or method at line {line_number}")
            if method in methods and row.get("evaluation_schema_version") == "3":
                latest[(method, sample_id)] = row
    return sorted(latest.values(), key=lambda row: (row["method"], int(row["sample_id"])))


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("overrides"), dict):
        raise ValueError("Override file must contain an overrides object")
    return payload["overrides"]


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * value) - 1)]


def ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "—"
    return f"{numerator / denominator:.1%}"


def tri_state_name(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "null"


def method_stats(rows: list[dict[str, Any]], method: str, overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scoped = [row for row in rows if row["method"] == method]
    success = [row for row in scoped if row.get("status") == "success"]
    caption = Counter(
        tri_state_name((row.get("caption_evaluation") or {}).get("is_correct"))
        for row in success
    )
    attributes = Counter(
        tri_state_name(item.get("is_correct"))
        for row in success
        for item in (row.get("caption_attribute_evaluation") or {}).values()
    )
    latencies = [float(row["latency_ms"]) for row in scoped if isinstance(row.get("latency_ms"), (int, float))]
    errors = Counter(
        ((row.get("error") or {}).get("code") or "unknown")
        for row in scoped if row.get("status") != "success"
    )
    human = [
        value for key, value in overrides.items()
        if key.startswith(f"{method}:") and isinstance(value, dict)
    ]
    human_attribute_fields = sum(
        len(value.get("attributes", {}))
        for value in human if isinstance(value.get("attributes", {}), dict)
    )
    return {
        "total": len(scoped), "success": len(success), "errors": len(scoped) - len(success),
        "caption": caption, "attributes": attributes, "attribute_total": sum(attributes.values()),
        "needs_human_review": sum(bool((row.get("caption_evaluation") or {}).get("needs_human_review")) for row in success),
        "latency_mean": statistics.mean(latencies) if latencies else None,
        "latency_p50": statistics.median(latencies) if latencies else None,
        "latency_p95": percentile(latencies, 0.95), "errors_by_code": errors,
        "human_rows": len(human), "human_attribute_fields": human_attribute_fields,
    }


def validate_strict_three_state_rows(rows: list[dict[str, Any]]) -> None:
    """Reject legacy rows that predate the strict caption-attribute contract."""
    invalid = 0
    for row in rows:
        if row.get("status") != "success":
            continue
        for item in (row.get("caption_attribute_evaluation") or {}).values():
            if isinstance(item, dict) and item.get("mentioned") is True and item.get("is_correct") is None:
                invalid += 1
    if invalid:
        raise ValueError(
            f"Found {invalid} legacy attribute verdicts with mentioned=true and is_correct=null. "
            "Re-run AI_harness_evaluation with the strict three-state prompt before rebuilding the report."
        )


def detailed_attribute_stats(rows: list[dict[str, Any]], method: str) -> dict[str, dict[str, int]]:
    """Count per-attribute caption-to-golden checks for successful samples."""
    result = {
        field: {"mentioned": 0, "correct": 0, "incorrect": 0, "not_mentioned": 0}
        for field, _ in ATTRIBUTE_LABELS
    }
    for row in rows:
        if row.get("method") != method or row.get("status") != "success":
            continue
        checks = row.get("caption_attribute_evaluation") or {}
        for field, _ in ATTRIBUTE_LABELS:
            check = checks.get(field)
            if not isinstance(check, dict):
                continue
            if not check.get("mentioned"):
                result[field]["not_mentioned"] += 1
            else:
                result[field]["mentioned"] += 1
                verdict = check.get("is_correct")
                if verdict is True:
                    result[field]["correct"] += 1
                else:
                    result[field]["incorrect"] += 1
    return result


def attribute_rate(item: dict[str, int]) -> float | None:
    denominator = item["correct"] + item["incorrect"]
    return item["correct"] / denominator if denominator else None


def format_attribute_comparison(item: dict[str, int]) -> str:
    """Format a mentor-facing attribute result: correct / decided claims (rate)."""
    decisive = item["correct"] + item["incorrect"]
    rate = ratio(item["correct"], decisive)
    return f"{item['correct']}/{decisive} ({rate})"


def format_attribute_finding(fields: list[tuple[str, float, int]]) -> str:
    labels = dict(ATTRIBUTE_LABELS)
    return ", ".join(
        f"{labels[field]} ({rate:.1%}, {count} lần có kết luận)"
        for field, rate, count in fields
    )


def duration(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_report(rows: list[dict[str, Any]], overrides: dict[str, dict[str, Any]], methods: tuple[str, ...]) -> str:
    validate_strict_three_state_rows(rows)
    stats = {method: method_stats(rows, method, overrides) for method in methods}
    details = {method: detailed_attribute_stats(rows, method) for method in methods}
    successful_ids = {
        method: {
            str(row["sample_id"])
            for row in rows
            if row.get("method") == method and row.get("status") == "success"
        }
        for method in methods
    }
    common_ids = set.intersection(*(successful_ids[method] for method in methods)) if methods else set()
    common_rows = [
        row for row in rows
        if row.get("status") == "success" and str(row.get("sample_id")) in common_ids
    ]
    common_details = {method: detailed_attribute_stats(common_rows, method) for method in methods}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_rows = sum(item["total"] for item in stats.values())
    total_success = sum(item["success"] for item in stats.values())
    best_method = max(
        methods,
        key=lambda method: (
            stats[method]["caption"]["true"] / stats[method]["success"]
            if stats[method]["success"] else -1
        ),
    )
    runner_up = sorted(
        methods,
        key=lambda method: (
            stats[method]["caption"]["true"] / stats[method]["success"]
            if stats[method]["success"] else -1
        ),
        reverse=True,
    )[1] if len(methods) > 1 else None
    lines = [
        "# Báo cáo thử nghiệm các phương pháp sinh caption (G3–G5)", "",
        f"Thời điểm tạo: {now}", "",
        "## 1. Mục tiêu", "",
        "So sánh ba cách sinh caption từ ảnh để chọn luồng phù hợp cho bước gán nhãn dữ liệu tự động. "
        "Mỗi caption được Gemini kiểm tra lại xem có mô tả sai hoặc bịa thêm chi tiết so với ảnh hay không.", "",
        "## 2. Kết luận nhanh", "",
        f"- Đã đánh giá **{total_rows}** mẫu; **{ratio(total_success, total_rows)}** hoàn thành evaluation.",
        f"- **{best_method.upper()} là lựa chọn tốt nhất ở batch hiện tại**: {ratio(stats[best_method]['caption']['true'], stats[best_method]['success'])} caption không có chi tiết sai theo ảnh.",
        (
            f"- **{runner_up.upper()} là phương án thứ hai**: {ratio(stats[runner_up]['caption']['true'], stats[runner_up]['success'])} caption đúng theo ảnh và không có lỗi evaluation trong batch này."
            if runner_up else ""
        ),
        "- G3 hiện chưa phù hợp làm luồng ưu tiên vì tỷ lệ caption có chi tiết sai cao hơn đáng kể.",
        "- Khuyến nghị: tiếp tục dùng **G4** làm baseline để mở rộng thử nghiệm; lấy thêm một mẫu ngẫu nhiên để người review xác nhận kết quả của LLM-as-judge trước khi đưa vào production.", "",
        "## 3. Ba phương pháp đã thử", "",
        "Các workflow G3–G5 là **thiết kế/điều chỉnh cho project này**, lấy cảm hứng từ các hướng nghiên cứu bên dưới; không phải triển khai nguyên văn của một paper.", "",
        "### G3 – đồng thuận nhiều lượt (multi-run consensus)", "",
        "1. Cùng một ảnh được VLM quan sát độc lập qua nhiều lượt (Run 1…Run N). Mỗi lượt trả về caption và thuộc tính có cấu trúc.",
        "2. Node **Consensus** đối chiếu các kết quả và chọn các thuộc tính xuất hiện nhất quán; các chi tiết mâu thuẫn hoặc thiếu bằng chứng có thể bị bỏ qua.",
        "3. Caption Generator viết caption cuối từ bộ thuộc tính đã đồng thuận, thay vì viết trực tiếp từ một lần quan sát duy nhất.",
        "4. **Nguồn cảm hứng:** [Self-Consistency Improves Chain of Thought Reasoning in Language Models — Wang et al., 2022](https://arxiv.org/abs/2203.11171). Paper này lấy nhiều đường suy luận rồi chọn câu trả lời nhất quán nhất; ở đây ý tưởng được chuyển thành nhiều lượt quan sát ảnh, không phải dùng nguyên phương pháp Chain-of-Thought của paper.", "",
        "### G4 – tạo, phản biện, kiểm tra (generate → critic → verifier)", "",
        "1. **Generator** đọc ảnh và tạo `draft caption + attributes`.",
        "2. **Critic** đọc lại ảnh cùng bản nháp, nêu ra chi tiết có thể sai, bịa thêm hoặc còn thiếu bằng chứng.",
        "3. **Verifier** nhận ảnh, bản nháp và nhận xét của Critic, rồi quyết định `Accept` hoặc `Reject`.",
        "4. Nếu bị `Reject`, phản hồi được đưa vào lượt **Regenerate** để Generator tạo bản mới. Luồng lặp đến khi được chấp nhận hoặc đạt số lần thử tối đa.",
        "5. **Nguồn cảm hứng:** [Self-Refine: Iterative Refinement with Self-Feedback — Madaan et al., 2023](https://arxiv.org/abs/2303.17651). Paper đề xuất sinh bản nháp, tạo feedback và tinh chỉnh lặp; project này điều chỉnh thành hai vai trò riêng Critic/Verifier có kiểm tra trực tiếp ảnh.", "",
        "### G5 – hỏi đáp để tinh chỉnh (question-guided refinement)", "",
        "1. Generator tạo `draft caption + attributes` từ ảnh.",
        "2. **Question Generator** biến các claim trong bản nháp thành câu hỏi kiểm chứng, ví dụ: “Người đó có tóc búi không?”, “Áo có màu trắng không?”.",
        "3. **Image Answerer** chỉ nhìn ảnh để trả lời từng câu hỏi; không lấy câu trả lời từ caption nháp.",
        "4. **Compare/Router** so sánh claim của caption với câu trả lời từ ảnh: khớp thì `Accept`, mâu thuẫn/không đủ bằng chứng thì gửi lý do về Generator để `Refine`.",
        "5. **Nguồn cảm hứng:** [QACE: Asking Questions to Evaluate an Image Caption — Lee et al., 2021](https://aclanthology.org/2021.findings-emnlp.395/). QACE dùng Question Generation + Question Answering để kiểm tra nội dung caption với ảnh; project này điều chỉnh cơ chế đó từ **evaluation** sang một vòng **tinh chỉnh caption khi sinh**.", "",
        "## 4. Cách đánh giá — một ví dụ từ đầu đến cuối", "",
        "Phần này minh hoạ **một case giả định** để người đọc thấy rõ hệ thống đánh giá gì và kết quả đi vào Excel như thế nào.", "",
        "### Case minh hoạ", "",
        "Giả sử ảnh có một phụ nữ trẻ, tóc đen búi, mặc áo trắng tay ngắn và váy trắng dài. Golden dataset của ảnh này đã có 21 nhãn, trong đó có `gender = female`, `upper_clothing_color = white`, `hairstyle = bun`, `footwear_type = sandals`.",
        "",
        "G4 hoặc G5 sinh caption sau:",
        "",
        "`A young adult female with dark hair in a bun wears a white short-sleeve top and a long white skirt.`", "",
        "Hệ thống không hỏi một câu chung chung là “caption có tốt không?”. Nó chấm **hai việc khác nhau**, rồi lưu cả hai để người review xem được lý do.", "",
        "### Bước 1 — Caption có nói sai về ảnh không?", "",
        "**Đưa vào Gemini:** ảnh gốc + caption vừa sinh. Không đưa golden attributes vào bước này.",
        "",
        "Gemini kiểm tra từng claim mà caption đã nói: “female”, “young adult”, “dark hair”, “bun”, “white top”, “long white skirt”. Nếu các claim này phù hợp với ảnh, kết quả là:",
        "",
        "```json",
        '{"caption_is_correct": true, "note": "Caption mô tả đúng các chi tiết quan sát được trong ảnh."}',
        "```", "",
        "Điểm quan trọng: caption **không cần kể hết ảnh** mới được đúng. Ví dụ caption rút gọn `A woman wears a white top.` vẫn được `true` nếu ảnh đúng như vậy; hệ thống không phạt vì caption chưa nói đến tóc, váy hay giày.", "",
        "Ngược lại, nếu model sinh `... She wears black boots.`, Gemini đối chiếu trực tiếp với ảnh. Khi ảnh không có/không cho thấy giày đen, kết quả Bước 1 là:",
        "",
        "```json",
        '{"caption_is_correct": false, "note": "Caption thêm chi tiết giày đen không được ảnh hỗ trợ."}',
        "```", "",
        "Nói ngắn gọn: **Bước 1 trả lời câu hỏi “model có bịa hoặc nói sai điều gì không?”**", "",
        "### Bước 2 — Những thuộc tính caption đã nói có khớp bộ nhãn chuẩn không?", "",
        "**Đưa vào Gemini:** caption + 21 golden attributes của đúng ảnh đó. Bước này không đưa ảnh vào, vì mục tiêu là đối chiếu cách caption diễn đạt với bộ nhãn chuẩn đã có.",
        "",
        "Với từng thuộc tính, hệ thống luôn trả hai thông tin: `mentioned` (caption có nói đến không) và `is_correct` (nếu có nói thì có khớp không). Ví dụ từ caption ở trên:", "",
        table_row(["Thuộc tính", "Golden label", "Caption có nhắc?", "Kết quả", "Lý do"]),
        table_row(["---"] * 5),
        table_row(["Giới tính", "female", "Có", "`true`", "Caption nói `female`, khớp nhãn."]),
        table_row(["Màu áo", "white", "Có", "`true`", "Caption nói `white ... top`, khớp nhãn."]),
        table_row(["Kiểu tóc", "bun", "Có", "`true`", "Caption nói `hair in a bun`, khớp nhãn."]),
        table_row(["Loại giày / dép", "sandals", "Không", "`null`", "Caption không nói về giày; đây là trung tính, **không phải lỗi**."]),
        table_row(["Màu túi", "null", "Không", "`null`", "Caption không nói về túi; không có gì để chấm đúng/sai."]),
        "",
        "Nếu caption sai thêm `black boots`, riêng hàng `Loại giày / dép` sẽ thành: `mentioned = true`, `is_correct = false`, vì caption đã khẳng định một thông tin khác golden label `sandals`.",
        "",
        "Nói ngắn gọn: **Bước 2 trả lời câu hỏi “khi caption có nêu một thuộc tính, nó có khớp nhãn chuẩn không?”**. Giá trị `null` chỉ nghĩa là caption không nhắc tới thuộc tính đó; nó không làm giảm điểm caption ở Bước 1.", "",
        "### Bước 3 — Người review nhìn thấy gì?", "",
        "Sau khi hai bước hoàn thành, hệ thống ghi ngay một dòng JSONL rồi xuất sang Excel/Dashboard. Một người review sẽ thấy: ảnh, caption tiếng Anh/Việt, verdict tổng `caption_is_correct`, note tiếng Việt và 21 hàng thuộc tính như bảng trên.",
        "",
        "Ví dụ dòng tóm tắt trong sheet **Caption**: `Case 123 | G4 | caption_is_correct = true | 3 thuộc tính khớp | 0 thuộc tính sai | 18 không nhắc`.",
        "Trong sheet **Thuộc tính**, người review có thể mở riêng hàng `Loại giày / dép` để thấy `mentioned = false`, `is_correct = null`; đây là lý do số lượng `null` cao không đồng nghĩa caption sai.",
        "",
        "Khi người review không đồng ý với AI judge, họ sửa verdict trên Dashboard/Excel. Human override được lưu riêng, không ghi đè kết quả gốc của Gemini; vì vậy sau này vẫn audit được AI đã chấm gì và người review đã sửa gì.", "",
        "### Quy ước khi đọc các bảng kết quả", "",
        "- **Caption đúng theo ảnh**: tỷ lệ `caption_is_correct = true` ở Bước 1.",
        "- **Thuộc tính khớp nhãn**: chỉ tính trong những thuộc tính caption có nhắc; mẫu số không bao gồm `null`.",
        "- **Null / không nhắc**: caption không nói đến thuộc tính đó hoặc chưa thể kết luận; không phải một lỗi mô tả.",
        "- **Thời gian đánh giá**: thời gian Gemini thực hiện việc kiểm tra, không phải tổng thời gian sinh caption.",
        "- Lưu ý: đây là kết quả **LLM-as-judge**. Chưa có chỉnh sửa của người review trong batch này, vì vậy chưa nên gọi là độ chính xác ground truth.", "",
        "## 5. Kết quả chính", "",
        table_row(["Luồng", "Số mẫu", "Evaluation thành công", "Lỗi", "Caption đúng theo ảnh", "Thuộc tính được nêu khớp nhãn", "P50 thời gian chấm"]),
        table_row(["---"] * 7),
    ]
    for method in methods:
        item = stats[method]
        lines.append(table_row([
            method.upper(), str(item["total"]), ratio(item["success"], item["total"]), str(item["errors"]),
            ratio(item["caption"]["true"], item["success"]),
            ratio(item["attributes"]["true"], item["attributes"]["true"] + item["attributes"]["false"]),
            f"{duration(item['latency_p50'])} ms",
        ]))
    lines.extend(["", "## 6. Chi tiết kết quả caption", "", table_row(["Luồng", "Caption đúng theo ảnh", "Caption có chi tiết sai"]), table_row(["---"] * 3)])
    for method in methods:
        item = stats[method]
        lines.append(table_row([method.upper(), *[str(item["caption"][label]) for label in CAPTION_RESULTS]]))
    lines.extend(["", "## 7. Chi tiết thuộc tính caption", "", table_row(["Luồng", "Khớp nhãn", "Không khớp nhãn", "Không nhắc (null)"]), table_row(["---"] * 4)])
    for method in methods:
        item = stats[method]
        lines.append(table_row([method.upper(), *[str(item["attributes"][label]) for label in ATTRIBUTE_RESULTS]]))
    lines.extend([
        "", "## 8. Phân tích chi tiết 21 thuộc tính", "",
        "Bảng dưới đây cho biết caption đã nhắc tới từng thuộc tính bao nhiêu lần. Tỷ lệ khớp chỉ tính các lần caption có nêu thuộc tính đó; “không nhắc” không phải lỗi.",
    ])
    for method in methods:
        lines.extend([
            "", f"### {method.upper()}", "",
            table_row(["Thuộc tính", "Caption có nhắc", "Khớp nhãn", "Không khớp", "Không nhắc", "Tỷ lệ khớp khi có nhắc"]),
            table_row(["---"] * 6),
        ])
        for field, label in ATTRIBUTE_LABELS:
            item = details[method][field]
            lines.append(table_row([
                label, str(item["mentioned"]), str(item["correct"]),
                str(item["incorrect"]), str(item["not_mentioned"]),
                ratio(item["correct"], item["correct"] + item["incorrect"]),
            ]))
    lines.extend(["", "## 9. Nhận xét trọng tâm từ thuộc tính", ""])
    for method in methods:
        minimum_mentions = max(10, stats[method]["success"] // 10)
        candidates = [
            (field, attribute_rate(item), item["correct"] + item["incorrect"])
            for field, item in details[method].items()
            if attribute_rate(item) is not None and item["mentioned"] >= minimum_mentions
        ]
        weakest = sorted(candidates, key=lambda item: item[1])[:3]
        strongest = sorted(candidates, key=lambda item: item[1], reverse=True)[:3]
        if weakest:
            lines.append(
                f"- **{method.upper()}**: nhóm cần hạn chế mô tả khi ảnh chưa rõ là "
                f"{format_attribute_finding(weakest)}."
            )
        if strongest:
            lines.append(
                f"  Nhóm ổn định hơn trong batch này: {format_attribute_finding(strongest)}."
            )
    lines.extend([
        "", "Nhận xét chung: giày/dép, vật mang theo, túi và một số chi tiết về quần/váy là các nhóm dễ tạo chi tiết sai. Prompt nên yêu cầu model chỉ nêu các thuộc tính này khi quan sát rõ; nếu không, caption ngắn hơn nhưng đúng sẽ tốt hơn.",
        "", "## 10. So sánh để quyết định áp dụng", "",
        "Bảng này dùng cùng tinh thần với báo cáo so sánh mô hình: mọi luồng được chấm trên batch hiện có. Tuy nhiên, metric được đặt tên theo đúng ý nghĩa của luồng evaluation hiện tại.", "",
        table_row(["Luồng", "Mẫu chấm thành công", "Cặp thuộc tính có thể kiểm tra", "Caption đúng theo ảnh", "Độ bao phủ thuộc tính", "Khớp nhãn khi có nhắc", "P50 thời gian chấm", "Khuyến nghị"]),
        table_row(["---"] * 8),
    ])
    for method in methods:
        item = stats[method]
        mentioned = sum(value["mentioned"] for value in details[method].values())
        supported = item["success"] * len(ATTRIBUTE_LABELS)
        recommendation = {
            "g3": "Không ưu tiên; cần giảm chi tiết sai.",
            "g4": "Baseline ưu tiên cho production thử nghiệm.",
            "g5": "Phương án dự phòng; tiếp tục tối ưu router/QA.",
        }.get(method, "Theo dõi thêm.")
        lines.append(table_row([
            method.upper(), str(item["success"]), str(supported),
            ratio(item["caption"]["true"], item["success"]),
            ratio(mentioned, supported),
            ratio(item["attributes"]["true"], item["attributes"]["true"] + item["attributes"]["false"]),
            f"{duration(item['latency_p50'])} ms", recommendation,
        ]))
    lines.extend([
        "",
        "### 10.1. So sánh chi tiết theo từng thuộc tính", "",
        f"Bảng dưới đặt ba luồng trên **cùng {len(common_ids)} case đã evaluation thành công ở cả G3, G4 và G5**.",
        "",
        "**Cách đọc một ô:** `19/122 (15,6%)` nghĩa là model đã đưa ra kết luận về thuộc tính đó 122 lần và đúng 19 lần. Ví dụ ở hàng *Loại giày / dép*, nếu model không nói gì về giày thì không tính là sai; bảng chỉ kiểm tra các lần model đã chủ động mô tả giày.",
        "",
        table_row(["STT", "Thuộc tính", "G3: đúng / đã mô tả", "G4: đúng / đã mô tả", "G5: đúng / đã mô tả", "Kết luận để áp dụng"]),
        table_row(["---"] * 6),
    ])
    for index, (field, label) in enumerate(ATTRIBUTE_LABELS, start=1):
        rates = {method: attribute_rate(common_details[method][field]) for method in methods}
        available = {method: value for method, value in rates.items() if value is not None}
        supports = {
            method: common_details[method][field]["correct"] + common_details[method][field]["incorrect"]
            for method in methods
        }
        if available:
            best_rate = max(available.values())
            best_methods = ", ".join(method.upper() for method in methods if rates[method] == best_rate)
            gap = max(available.values()) - min(available.values())
            best_support = max(supports[method] for method in methods if rates[method] == best_rate)
            if max(supports.values()) < 10:
                conclusion = "Ít case được mô tả; chỉ tham khảo."
            elif best_support < 30:
                conclusion = f"{best_methods} cao nhất nhưng ít case; chỉ tham khảo."
            elif best_rate < 0.60:
                conclusion = "Cả ba còn yếu; chưa tự động gán nhãn."
            elif gap < 0.05:
                conclusion = "Ba luồng gần tương đương."
            else:
                conclusion = f"{best_methods} tốt nhất ở thuộc tính này."
        else:
            conclusion = "Chưa có dữ liệu để kết luận."
        lines.append(table_row([
            str(index), label,
            *[format_attribute_comparison(common_details[method][field]) for method in methods],
            conclusion,
        ]))
    lines.extend([
        "",
        "**Cách đọc để ra quyết định:** G4 có tỷ lệ caption đúng theo ảnh cao nhất trong batch nên là baseline đề xuất. G5 đứng thứ hai và phù hợp để tiếp tục thử khi cần cơ chế hỏi–đáp/kiểm chứng. G3 không nên mở rộng trước khi xử lý các nhóm lỗi có tỷ lệ sai cao.",
        "",
        "**Không đưa Micro Accuracy/Macro Accuracy/Exact-match 21 thuộc tính vào bảng này.** Những chỉ số đó chỉ hợp lệ khi lấy **21 generated attributes đã chuẩn hoá** của mỗi case so sánh trực tiếp với 21 golden attributes, nên mọi trường đều có support bằng nhau. Luồng hiện tại chủ đích chấm **caption**: một thuộc tính không được caption nhắc tới sẽ là `null` (trung tính), không phải prediction sai. Gọi tỷ lệ này là “attribute accuracy 21 trường” sẽ gây hiểu nhầm.",
        "",
        "Nếu cần báo cáo theo đúng mẫu VLLM vs Qwen3.5 4B (Micro/Macro/Exact 21/21), bước tiếp theo là chạy thêm một evaluator riêng: `generated structured attributes + golden attributes` → chuẩn hoá taxonomy → so sánh exact match từng trường. Khi đó có thể tạo bảng model/method theo cohort chung một cách hợp lệ.",
        "", "## 11. Giới hạn và bước tiếp theo", "",
        "1. Gemini đang đóng vai trò judge, do đó metric là tín hiệu so sánh giữa các luồng, chưa thay thế đánh giá của con người.",
        "2. Chưa có human override nào trong batch này. Nên review ngẫu nhiên các caption đúng/sai của G4 và G5 để kiểm tra judge có lệch hay không.",
        "3. Nếu human review xác nhận xu hướng này, dùng G4 làm baseline và tiếp tục tối ưu prompt/loop để giảm các chi tiết sai còn lại.", "",
        "## Phụ lục: hoạt động human review", "", table_row(["Luồng", "Dòng review đã lưu", "Thuộc tính đã chỉnh"]), table_row(["---"] * 3)])
    for method in methods:
        item = stats[method]
        lines.append(table_row([method.upper(), str(item["human_rows"]), str(item["human_attribute_fields"])]))
    lines.extend(["", "## Phụ lục: lỗi evaluation", ""])
    has_errors = False
    for method in methods:
        item = stats[method]
        if item["errors_by_code"]:
            has_errors = True
            failures = ", ".join(f"`{code}`: {count}" for code, count in sorted(item["errors_by_code"].items()))
            lines.append(f"- **{method.upper()}**: {failures}")
    if not has_errors:
        lines.append("Không có bản ghi G3–G5 mới nhất nào có trạng thái lỗi.")
    lines.extend([
        "", "## Phụ lục kỹ thuật: chạy lại báo cáo", "",
        "```bash",
        "python tools/build_g3_g5_evaluation_report.py",
        "```", "",
        "Chạy lệnh này sau evaluation hoặc sau khi lưu chỉnh sửa trên dashboard để cập nhật số liệu.", "",
    ])
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the G3–G5 Markdown evaluation report.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    methods = tuple(args.methods)
    rows = latest_rows(args.input.resolve(), methods)
    overrides = load_overrides(args.overrides.resolve())
    atomic_write(args.output.resolve(), build_report(rows, overrides, methods))
    print(f"Wrote report for {', '.join(methods).upper()} to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
