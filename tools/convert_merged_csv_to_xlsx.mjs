#!/usr/bin/env node
/** Convert the caption-review CSV produced by merge_caption_dataset.py to XLSX. */

import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function option(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = process.argv[index + 1];
  if (!value || value.startsWith("--")) {
    throw new Error(`${name} requires a path`);
  }
  return value;
}


function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}


const root = process.cwd();
const inputPath = path.resolve(root, option("--input", "output/review/captions_merged.csv"));
const outputPath = path.resolve(root, option("--output", "output/review/captions_merged.xlsx"));
const csvText = (await fs.readFile(inputPath, "utf8")).replace(/^\uFEFF/, "");
const headers = csvText.split(/\r?\n/, 1)[0].split(",");

if (headers.length < 2) {
  throw new Error("Input CSV must contain a header row");
}

const workbook = await Workbook.fromCSV(csvText, { sheetName: "Captions" });
const sheet = workbook.worksheets.getItem("Captions");
const lastColumn = columnName(headers.length - 1);
const headerRange = sheet.getRange(`A1:${lastColumn}1`);
const usedRange = sheet.getUsedRange();

sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
usedRange.format = {
  font: { name: "Arial", size: 10, color: "#1F1F1F" },
  verticalAlignment: "top",
  wrapText: true,
};
usedRange.format.rowHeight = 88;
headerRange.format = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF", name: "Arial", size: 10 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
};
headerRange.format.rowHeight = 32;

for (let index = 0; index < headers.length; index += 1) {
  const name = headers[index];
  const range = sheet.getRange(`${columnName(index)}:${columnName(index)}`);
  range.format.wrapText = true;
  range.format.verticalAlignment = "top";
  range.format.columnWidth = name === "sample_id" ? 10
    : name === "image_path" ? 48
      : name.endsWith("_status") ? 12
        : 42;
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Converted ${inputPath} to ${outputPath}`);
