#!/usr/bin/env python3
"""Serve the local image and two-branch caption-review dashboard on loopback only."""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import subprocess
import tempfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "output" / "dashboard" / "index.json"
REVIEWS_PATH = ROOT / "output" / "evaluation" / "reviews_v3.jsonl"
OVERRIDES_PATH = ROOT / "output" / "dashboard" / "review_overrides_v3.json"
WORKBOOK_PATH = ROOT / "output" / "evaluation" / "evaluation_review_v3.xlsx"
EXPORTER_PATH = ROOT / "tools" / "export_evaluation_to_xlsx.mjs"
STATIC_ROOTS = {
    "/dashboard/": ROOT / "dashboard",
    "/sample/test/images/": ROOT / "sample" / "test" / "images",
}
METHODS = {"g1", "g2", "g3", "g4", "g5"}
ATTRIBUTE_FIELDS = {
    "age", "gender", "body_build", "upper_clothing_type", "upper_clothing_color",
    "lower_clothing_type", "lower_clothing_color", "clothing_style", "upper_pattern",
    "footwear_type", "bag_type", "bag_color", "headwear", "eyewear", "face_mask",
    "other_accessories", "hair_length", "hair_texture", "hairstyle", "hair_color",
    "carried_objects",
}
MAX_BODY_BYTES = 40_000
FORBIDDEN = ROOT / ".dashboard-forbidden"


class DashboardError(Exception):
    pass


def latest_reviews() -> dict[tuple[str, str], dict[str, Any]]:
    if not REVIEWS_PATH.is_file():
        raise DashboardError("Chưa có evaluation v3. Hãy chạy AI_harness_evaluation trước.")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        for line in REVIEWS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, method = row.get("sample_id"), row.get("method")
            if (
                row.get("evaluation_schema_version") == "3"
                and isinstance(sample_id, str)
                and isinstance(method, str)
                and method in METHODS
            ):
                latest[(sample_id, method)] = row
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardError("Không đọc được dữ liệu evaluation v3.") from exc
    return latest


def load_overrides() -> dict[str, Any]:
    if not OVERRIDES_PATH.is_file():
        return {"schema_version": "3", "overrides": {}}
    try:
        payload = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardError("Không đọc được nhãn đã sửa v3.") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "3"
        or not isinstance(payload.get("overrides"), dict)
    ):
        raise DashboardError("File nhãn đã sửa v3 không đúng định dạng.")
    return payload


def save_overrides(payload: dict[str, Any]) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".review_overrides_v3.", suffix=".tmp", dir=OVERRIDES_PATH.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, OVERRIDES_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _nullable_boolean(value: Any) -> bool:
    return value is None or isinstance(value, bool)


def validate_override(value: Any) -> dict[str, Any]:
    expected = {"sample_id", "method", "caption", "attributes"}
    if not isinstance(value, dict) or set(value) != expected:
        raise DashboardError("Dữ liệu lưu không hợp lệ.")
    sample_id, method = value["sample_id"], value["method"]
    if not isinstance(sample_id, str) or not sample_id or method not in METHODS:
        raise DashboardError("Sample hoặc method không hợp lệ.")

    caption = value["caption"]
    if caption is not None:
        if not isinstance(caption, dict) or set(caption) != {"is_correct"} or not isinstance(caption["is_correct"], bool):
            raise DashboardError("Nhãn caption phải là true hoặc false.")

    attributes = value["attributes"]
    if not isinstance(attributes, dict) or not set(attributes).issubset(ATTRIBUTE_FIELDS):
        raise DashboardError("Có attribute không hợp lệ.")
    validated_attributes: dict[str, dict[str, bool | None]] = {}
    for field, item in attributes.items():
        if not isinstance(item, dict) or set(item) != {"is_correct"} or not _nullable_boolean(item["is_correct"]):
            raise DashboardError("Nhãn attribute phải là true, false hoặc null.")
        validated_attributes[field] = {"is_correct": item["is_correct"]}

    return {
        "sample_id": sample_id,
        "method": method,
        "caption": caption,
        "attributes": validated_attributes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def safe_child(root: Path, relative_path: str) -> Path | None:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


class DashboardHandler(SimpleHTTPRequestHandler):
    server_version = "LocalReviewDashboard/2.0"

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def api_error(self, status: HTTPStatus, message: str) -> None:
        self.send_json(status, {"error": {"message": message}})

    def route(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(self.path)
        return posixpath.normpath(unquote(parsed.path)), parse_qs(parsed.query, keep_blank_values=True)

    @staticmethod
    def query_sample_id(query: dict[str, list[str]]) -> str:
        values = query.get("sample_id", [])
        if len(values) != 1 or not values[0].strip():
            raise DashboardError("sample_id là bắt buộc.")
        return values[0].strip()

    def do_GET(self) -> None:
        path, query = self.route()
        if path == "/api/reviews":
            try:
                sample_id = self.query_sample_id(query)
                rows = latest_reviews()
                overrides = load_overrides()["overrides"]
                reviews = [
                    {"evaluation": row, "override": overrides.get(f"{method}:{sample_id}")}
                    for method in sorted(METHODS)
                    if (row := rows.get((sample_id, method))) is not None
                ]
                self.send_json(HTTPStatus.OK, {"sample_id": sample_id, "reviews": reviews})
            except DashboardError as exc:
                self.api_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        super().do_GET()

    def do_PUT(self) -> None:
        path, _ = self.route()
        if path != "/api/reviews":
            self.api_error(HTTPStatus.NOT_FOUND, "Không tìm thấy API.")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY_BYTES:
                raise DashboardError("Dữ liệu lưu quá lớn hoặc bị thiếu.")
            override = validate_override(json.loads(self.rfile.read(length).decode("utf-8")))
            if (override["sample_id"], override["method"]) not in latest_reviews():
                raise DashboardError("Không tìm thấy evaluation v3 tương ứng.")
            payload = load_overrides()
            payload["overrides"][f"{override['method']}:{override['sample_id']}"] = override
            save_overrides(payload)
            self.send_json(HTTPStatus.OK, {"override": override})
        except (DashboardError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            message = str(exc) if isinstance(exc, DashboardError) else "Dữ liệu lưu không hợp lệ."
            self.api_error(HTTPStatus.BAD_REQUEST, message)
        except OSError:
            self.api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Không thể lưu nhãn đã sửa.")

    def do_POST(self) -> None:
        path, _ = self.route()
        if path != "/api/export":
            self.api_error(HTTPStatus.NOT_FOUND, "Không tìm thấy API.")
            return
        try:
            result = subprocess.run(
                ["node", str(EXPORTER_PATH), "--input", str(REVIEWS_PATH), "--overrides", str(OVERRIDES_PATH), "--output", str(WORKBOOK_PATH)],
                cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
            )
            if result.returncode or not WORKBOOK_PATH.is_file():
                raise DashboardError("Không thể xuất Excel v3.")
            content = WORKBOOK_PATH.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            self.send_header("Content-Disposition", 'attachment; filename="evaluation_review_v3.xlsx"')
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (DashboardError, OSError, subprocess.TimeoutExpired):
            self.api_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Xuất Excel v3 thất bại.")

    def translate_path(self, path: str) -> str:
        request_path = posixpath.normpath(unquote(urlsplit(path).path))
        if request_path in {"/", "/dashboard"}:
            return str(ROOT / "dashboard" / "index.html")
        if request_path == "/output/dashboard/index.json":
            return str(INDEX_PATH)
        for prefix, root in STATIC_ROOTS.items():
            if request_path.startswith(prefix):
                child = safe_child(root, request_path.removeprefix(prefix))
                if child is not None:
                    return str(child)
        return str(FORBIDDEN)

    def send_head(self):
        if self.translate_path(self.path) == str(FORBIDDEN):
            self.send_error(HTTPStatus.NOT_FOUND, "Dashboard resource not found")
            return None
        return super().send_head()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the local v3 review dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("For safety, --host must be a loopback address.")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}/dashboard/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
