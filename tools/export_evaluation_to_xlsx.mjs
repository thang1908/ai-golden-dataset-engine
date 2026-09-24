#!/usr/bin/env node
/** Export factuality evaluation v3 to combined or method-specific Excel workbooks. */

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const SPLIT_METHODS = ["g3", "g4", "g5"];


function option(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} requires a value`);
  return value;
}


function newestRows(jsonl) {
  const latest = new Map();
  for (const line of jsonl.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (row.evaluation_schema_version === "3") latest.set(`${row.method}:${row.sample_id}`, row);
  }
  return [...latest.values()].sort((left, right) => {
    const method = String(left.method).localeCompare(String(right.method));
    return method || Number(left.sample_id) - Number(right.sample_id);
  });
}


function displayVerdict(value) {
  if (value === true) return "true";
  if (value === false) return "false";
  if (value === undefined) return "";
  return "null";
}


function interpretation(check) {
  if (!check?.mentioned) return "Không nhắc (không tính lỗi)";
  if (check.is_correct === true) return "Có nhắc và khớp nhãn";
  if (check.is_correct === false) return "Có nhắc nhưng không khớp nhãn";
  return "Dữ liệu cũ không hợp lệ; cần chấm lại";
}


function humanCaption(override) {
  return override?.caption?.is_correct === true || override?.caption?.is_correct === false
    ? displayVerdict(override.caption.is_correct)
    : "";
}


function humanAttribute(override, field) {
  if (!Object.hasOwn(override?.attributes ?? {}, field)) return "";
  return displayVerdict(override.attributes[field].is_correct);
}


function styleSheet(sheet, titleCell, subtitleCell) {
  sheet.showGridLines = false;
  sheet.getUsedRange().format.font = { name: "Arial", size: 10, color: "#1F2937" };
  sheet.getUsedRange().format.verticalAlignment = "top";
  sheet.getUsedRange().format.wrapText = true;
  sheet.getRange(titleCell).format.font = { name: "Arial", size: 15, bold: true, color: "#111827" };
  sheet.getRange(subtitleCell).format.font = { name: "Arial", size: 10, italic: true, color: "#4B5563" };
}


function styleHeader(range) {
  range.format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
}


function verdictFormats(range) {
  range.conditionalFormats.add("cellIs", {
    operator: "equal", formula: "\"true\"",
    format: { fill: "#DCFCE7", font: { color: "#166534" } },
  });
  range.conditionalFormats.add("cellIs", {
    operator: "equal", formula: "\"false\"",
    format: { fill: "#FEE2E2", font: { color: "#991B1B" } },
  });
  range.conditionalFormats.add("cellIs", {
    operator: "equal", formula: "\"null\"",
    format: { fill: "#FEF3C7", font: { color: "#92400E" } },
  });
}


function buildRows(records, overrideItems) {
  const successful = records.filter((row) => row.status === "success");
  const captions = records.map((row) => {
    const caption = row.caption_evaluation ?? {};
    const summary = row.caption_attribute_summary ?? {};
    const override = overrideItems[`${row.method}:${row.sample_id}`] ?? {};
    return [
      row.sample_id, row.filepath, row.status, displayVerdict(caption.is_correct),
      caption.needs_human_review ?? "", row.prediction?.caption ?? "", row.prediction?.caption_vi ?? "",
      caption.note ?? "", summary.mentioned ?? "", summary.correct ?? "", summary.incorrect ?? "",
      summary.not_mentioned ?? "", summary.evaluated ?? "", row.latency_ms ?? "",
      row.error?.message ?? "", humanCaption(override),
    ];
  });
  const attributes = [];
  for (const row of successful) {
    const caption = row.caption_evaluation ?? {};
    const override = overrideItems[`${row.method}:${row.sample_id}`] ?? {};
    for (const [field, check] of Object.entries(row.caption_attribute_evaluation ?? {})) {
      attributes.push([
        row.sample_id, row.filepath, row.prediction?.caption ?? "", displayVerdict(caption.is_correct),
        field, check.mentioned ?? "", JSON.stringify(row.golden_attributes?.[field]),
        displayVerdict(check.is_correct), interpretation(check), check.note ?? "", humanAttribute(override, field),
      ]);
    }
  }
  return { successful, captions, attributes };
}


function createWorkbook(records, overrideItems, title) {
  const { successful, captions, attributes } = buildRows(records, overrideItems);
  const workbook = Workbook.create();
  const captionSheet = workbook.worksheets.add("Caption");
  const attributeSheet = workbook.worksheets.add("Thuộc tính");

  captionSheet.getRange("F1").values = [[`${title} — đánh giá caption`]];
  captionSheet.getRange("F2").values = [["Caption đúng theo ảnh được đánh giá từ ảnh + caption. Caption ngắn nhưng không sai vẫn được tính đúng."]];
  captionSheet.getRange("A3:D3").values = [[
    `Số mẫu: ${records.length}`, `Thành công: ${successful.length}`,
    `Lỗi: ${records.length - successful.length}`, "Nhãn Human là phần chỉnh riêng, không thay đổi kết quả Gemini.",
  ]];
  captionSheet.getRange("A5:P5").values = [[
    "Sample ID", "Image path", "Trạng thái", "Caption đúng theo ảnh", "Cần người review",
    "Caption EN", "Caption VI", "Ghi chú Gemini", "Thuộc tính có nhắc", "Thuộc tính khớp",
    "Thuộc tính không khớp", "Thuộc tính không nhắc", "Thuộc tính đã chấm", "Latency ms", "Lỗi", "Human caption",
  ]];
  if (captions.length) captionSheet.getRangeByIndexes(5, 0, captions.length, 16).values = captions;
  styleSheet(captionSheet, "F1", "F2");
  styleHeader(captionSheet.getRange("A5:P5"));
  captionSheet.getRange("A5:P5").format.rowHeight = 32;
  captionSheet.getRange(`A6:P${captions.length + 5}`).format.rowHeight = 72;
  captionSheet.freezePanes.freezeRows(5);
  captionSheet.freezePanes.freezeColumns(2);
  [11, 45, 12, 18, 17, 52, 52, 52, 16, 15, 17, 15, 17, 12, 38, 20]
    .forEach((width, index) => captionSheet.getRangeByIndexes(0, index, captions.length + 5, 1).format.columnWidth = width);
  verdictFormats(captionSheet.getRange(`D6:D${captions.length + 5}`));
  verdictFormats(captionSheet.getRange(`P6:P${captions.length + 5}`));

  attributeSheet.getRange("C1").values = [[`${title} — chi tiết 21 thuộc tính`]];
  attributeSheet.getRange("C2").values = [["Mỗi hàng là một thuộc tính của caption. true = caption có nhắc và khớp nhãn; false = caption có nhắc nhưng không khớp; null = caption không nhắc. Không có trạng thái 'chưa xác định'."]];
  attributeSheet.getRange("A4:K4").values = [[
    "Sample ID", "Image path", "Caption EN", "Caption đúng theo ảnh", "Thuộc tính", "Caption có nhắc",
    "Golden value", "Gemini verdict", "Diễn giải", "Ghi chú Gemini", "Human verdict",
  ]];
  if (attributes.length) attributeSheet.getRangeByIndexes(4, 0, attributes.length, 11).values = attributes;
  styleSheet(attributeSheet, "C1", "C2");
  styleHeader(attributeSheet.getRange("A4:K4"));
  attributeSheet.getRange("A4:K4").format.rowHeight = 32;
  attributeSheet.getRange(`A5:K${attributes.length + 4}`).format.rowHeight = 54;
  attributeSheet.freezePanes.freezeRows(4);
  attributeSheet.freezePanes.freezeColumns(3);
  [11, 45, 52, 18, 28, 17, 28, 17, 31, 54, 18]
    .forEach((width, index) => attributeSheet.getRangeByIndexes(0, index, attributes.length + 4, 1).format.columnWidth = width);
  verdictFormats(attributeSheet.getRange(`D5:D${attributes.length + 4}`));
  verdictFormats(attributeSheet.getRange(`H5:H${attributes.length + 4}`));
  verdictFormats(attributeSheet.getRange(`K5:K${attributes.length + 4}`));
  return workbook;
}


const root = process.cwd();
const inputPath = path.resolve(root, option("--input", "output/evaluation/reviews_v3.jsonl"));
const overridesPath = path.resolve(root, option("--overrides", "output/dashboard/review_overrides_v3.json"));
const requestedMethod = option("--method", "");
const split = process.argv.includes("--split");
const outputOption = option("--output", "");
if (split && requestedMethod) throw new Error("Use either --split or --method, not both");
if (requestedMethod && !SPLIT_METHODS.includes(requestedMethod)) {
  throw new Error(`--method must be one of: ${SPLIT_METHODS.join(", ")}`);
}
if (split && outputOption) throw new Error("--output cannot be used with --split");

const records = newestRows(await fs.readFile(inputPath, "utf8"));
let overrideItems = {};
try {
  const document = JSON.parse(await fs.readFile(overridesPath, "utf8"));
  if (document?.schema_version === "3" && typeof document.overrides === "object") {
    overrideItems = document.overrides;
  }
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const targets = split
  ? SPLIT_METHODS.map((method) => ({
    method,
    outputPath: path.resolve(root, "output/evaluation", `evaluation_review_${method}_v3.xlsx`),
  }))
  : [{
    method: requestedMethod || "all",
    outputPath: path.resolve(root, outputOption || (requestedMethod
      ? `output/evaluation/evaluation_review_${requestedMethod}_v3.xlsx`
      : "output/evaluation/evaluation_review_v3.xlsx")),
  }];

for (const target of targets) {
  const selected = target.method === "all" ? records : records.filter((row) => row.method === target.method);
  const workbook = createWorkbook(selected, overrideItems, target.method === "all" ? "Tất cả phương pháp" : target.method.toUpperCase());
  await fs.mkdir(path.dirname(target.outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(target.outputPath);
  console.log(`Exported ${selected.length} latest v3 rows to ${target.outputPath}`);
}
