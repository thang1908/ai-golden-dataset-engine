# Architecture decisions

## ADR index

| ID | Title | Status |
|---|---|---|
| ADR-001 | Five independent vertical flow packages | Proposed |
| ADR-002 | Plain synchronous Python orchestration | Proposed |
| ADR-003 | Strict structured output at every model stage | Proposed |
| ADR-004 | Preserve traces but omit evaluation scores | Proposed |
| ADR-005 | Bounded self-correction with strict exhaustion | Proposed |
| ADR-006 | V-LLM Medium and C1-compatible transport | Proposed |
| ADR-007 | Static local review workspace | Proposed |
| ADR-008 | Gemini multimodal judge as a separate local package | Accepted |
| ADR-009 | Static local image lookup is separate from spreadsheet review | Accepted |
| ADR-010 | Human overrides are a separate local overlay | Proposed |
| ADR-011 | Factuality-first, nullable AI Harness evaluation v2 | Proposed |
| ADR-012 | Local dashboard and Excel consume evaluation v2 separately | Proposed |
| ADR-013 | Isolate caption factuality from attribute reference context | Proposed |
| ADR-016 | Per-case request/token fields with terminal progress logging | Accepted |

## ADR-001: Five independent vertical flow packages

- Status: Accepted
- Date: 2026-09-22
- Owner: generation workflow maintainers
- Context: FR-001 asks for five folders and C1 is intentionally independent.
- Decision: create five top-level packages, each owning its adapters, taxonomy,
  contracts, prompts, orchestration, CLI, and documentation. Do not import C1 or
  another flow at runtime.
- Alternatives: one `generative_flows` package with five submodules; a shared
  `vllm_common` package; extending `golden_dataset_harness`/LangGraph.
- Consequences: easiest isolated reading/running and fewer hidden dependencies;
  some duplicated provider plumbing must be kept behaviorally consistent.
- Evidence/open items: user requested five folders; OQ-001 and OQ-002.

## ADR-002: Plain synchronous Python orchestration

- Status: Accepted
- Date: 2026-09-22
- Context: flows are linear or bounded loops for one image (FR-003–007, NFR-007).
- Decision: implement explicit functions and loops, executing G3 observers
  sequentially initially.
- Alternatives: LangGraph, asyncio fan-out, queue-backed workers.
- Consequences: straightforward traces, tests, token reuse, and failure handling;
  higher G3 wall time than parallel fan-out. Parallelism remains an extension after
  rate limits are known (OQ-004).

## ADR-003: Strict structured output at every model stage

- Status: Proposed
- Date: 2026-09-22
- Context: downstream stages cannot safely consume unconstrained prose (FR-010,
  NFR-002).
- Decision: every response uses provider JSON Schema and local semantic validation;
  schemas disallow extra properties and bound collections/text.
- Alternatives: parse free text; validate only the final annotation.
- Consequences: more schema code and possible provider incompatibility are traded
  for explicit contracts and deterministic failure.

## ADR-004: Preserve traces but omit evaluation scores

- Status: Proposed
- Date: 2026-09-22
- Context: the user wants manual viewing now and evaluation later (FR-009, BR-009).
- Decision: output validated intermediate artifacts, decisions, reasons, prompt
  version, and final annotation; do not output confidence, judge, or quality scores.
- Alternatives: final annotation only; reuse the existing harness evaluator.
- Consequences: larger but inspectable JSON and no misleading pseudo-metric. A
  future evaluator can consume the stable `final` object.

## ADR-005: Bounded self-correction with strict exhaustion

- Status: Proposed
- Date: 2026-09-22
- Context: G4 and G5 contain cycles that could otherwise run indefinitely (BR-008).
- Decision: default G4 to at most 20 generator attempts and G5 to at most 20
  refinement rounds. Publish `final` only after an accept decision; otherwise exit
  non-zero.
- Alternatives: unbounded loop; publish last rejected draft; always one retry.
- Consequences: bounded cost and honest final semantics, but loop-exhausted traces
  are not written as normal success output. OQ-005 may change this.

## ADR-006: V-LLM Medium and C1-compatible transport

- Status: Proposed
- Date: 2026-09-22
- Context: the user explicitly requested `v-llm-medium` and C1 as reference
  (FR-008, BR-001, BR-007).
- Decision: default to `v-llm-v1-medium`; reuse C1's OAuth, URL normalization,
  Chat Completions body, image limit, token cache, safe errors, and retry behavior.
  Flow-specific model config never inherits shared `VLLM_MODEL`.
- Alternatives: import C1 directly; use the existing generic harness adapter; select
  another provider/model.
- Consequences: compatible, testable behavior without coupling to C1. Multiple model
  calls increase cost by flow; no fallback is introduced without a requirement.

## ADR-007: Static local review workspace

- Status: Proposed
- Date: 2026-09-23
- Context: golden data and each G output currently require manual file switching.
- Decision: use a read-only Excel export that joins by `sample_id` and writes one
  workbook with six golden-caption columns followed by G1–G5 caption/status columns;
  it calculates no metric.
- Alternatives: HTML page, local server, manual TSV/JSONL inspection.
- Consequences: offline, simple review with no provider dependency; regenerate the
  workbook after a flow run. OQ-007 is resolved.

## ADR-008: Gemini multimodal judge as a separate local package

- Status: Accepted
- Date: 2026-09-23
- Context: the user requested an evaluation input comprising query image,
  human-labeled attributes, and generated annotation, using a Gemini API key.
- Decision: use the separate `AI_harness_evaluation/` package with stateless Gemini
  `models.generate_content`, structured response validation, and append-only JSONL.
  Its default is `gemini-2.5-pro`, overridable by `GEMINI_MODEL`; it does not modify
  G1–G5 or golden data.
- Alternatives: human-only review, V-LLM as judge, a judge inside every generator,
  or a hosted service.
- Consequences: multimodal review becomes scalable but adds a Gemini privacy/cost/
rate-limit boundary. Exact model and manifest availability are blocking OQ-009–011.

## ADR-009: Static local image lookup is separate from spreadsheet review

- Status: Accepted
- Date: 2026-09-23
- Context: an operator needs to select an image and immediately see its exact
  dataset identity without navigating image folders. Excel remains the approved
  surface for caption/evaluation comparison.
- Decision: add a static local dashboard backed by a generated, validated JSON
  index. The index joins `attributes.tsv` and `captions_merged.csv`; a narrow
  loopback server exposes only dashboard assets, the index, and test images. The
  UI renders only the exact mapped image and audit identifiers.
- Alternatives: extend Excel with embedded images; parse TSV/CSV directly in the
  browser; use folder-order image selection; add a local API/database/backend.
- Consequences: quick visual lookup with no external service, secrets, or model
  calls; the index must be regenerated after mapping changes. The later review
  extension is governed separately by ADR-010.

## ADR-010: Human overrides are a separate local overlay

- Status: Proposed
- Date: 2026-09-24
- Context: the dashboard must show Gemini review fields, allow an operator to
  correct them, and export Excel without corrupting generated or golden sources
  (FR-026–030, BR-016–019).
- Decision: store human caption/attribute verdicts and notes in one validated,
  atomic local overlay keyed by `(sample_id, method)`. Render both source and human
  fields, then have the existing Excel exporter merge the overlay into extra columns.
- Alternatives: modify `reviews.jsonl` in place; modify golden TSV; edit Excel as
  the only source; introduce a multi-user database/API.
- Consequences: clear provenance and reversible local review with minimal runtime
  complexity. The JSON overlay has no multi-user conflict protection and source
  label updates remain explicitly deferred (OQ-013–014).

## ADR-011: Factuality-first, nullable AI Harness evaluation v2

- Status: Proposed
- Date: 2026-09-24
- Owner: generation workflow maintainers
- Context: Minor/major caption severity is subjective and penalizes a caption that
  is accurate but concise. The user requires true/false factuality, a Vietnamese
  judge note, and a non-penalizing state for caption attributes that were never
  mentioned (FR-031–036, BR-020–024).
- Decision: replace the v1 severity vocabulary for new evaluations with: a
  top-level boolean caption factuality result; 21 caption checks using
  `mentioned` plus boolean/null `is_correct`; explicit unmapped claims; and 21
  structured-attribute boolean/null results. Store v2 rows and human overrides in
  separate v2 artifacts. Metrics exclude null and report coverage separately.
- Alternatives: retain the v1 level/minor/major labels; treat all missing caption
  attributes as false; judge caption only without generated/golden attributes;
  rewrite historical JSONL in place.
- Consequences: the contract directly reflects hallucination versus omission and
  supports clearer metrics. It expands the judge response and requires coordinated
  updates to evaluator, tests, dashboard, workbook exporter, and report. V1/V2
  side-by-side files avoid unsafe migration but add one explicit artifact choice.
- Evidence/open items: user requested the boolean/null semantics; OQ-015 and
  OQ-016 remain for contract freeze.

### Scope amendment

The initial implementation rebuilt only the evaluator. G1–G5 outputs remain
read-only inputs. ADR-012 re-enables dashboard, Excel exporter, and the human v2
overlay; report-builder changes remain outside this extension.

## ADR-012: Local dashboard and Excel consume evaluation v2 separately

- Status: Proposed
- Date: 2026-09-24
- Owner: pending user approval
- Context: the user now requests v2 Excel export and dashboard after evaluator v2
  is built (FR-037–039, BR-025–026).
- Decision: switch the local dashboard server and Excel exporter to v2 JSONL and a
  separate v2 human overlay, rendering boolean/null factuality rather than v1
  severity labels. Keep the same loopback-only architecture and never change G1–G5,
  historical v1 artifacts, or Gemini source rows.
- Alternatives: overwrite v1 files; add v2 rows to v1 workbook/dashboard; build a
  database-backed reviewer; make the browser write JSONL directly.
- Consequences: clear schema/provenance separation and minimal new infrastructure;
  users must run v2 evaluation before viewing/exporting v2 data. Existing v1
workbook/dashboard data stays available as historical output.

## ADR-013: Isolate caption factuality from attribute reference context

- Status: Proposed
- Date: 2026-09-24
- Owner: pending user approval
- Context: the user requires caption evaluation to inspect image and caption only
  (FR-040–041, BR-027).
- Decision: perform separate caption and attribute Gemini requests with separate
  schemas/prompts, then combine their validated results in the existing v2 row.
- Consequences: removes evidence leakage but doubles normal request count and either
  stage can fail the task.

## ADR-014: Persist two independent evaluator products in v3

- Status: Accepted
- Date: 2026-09-24
- Owner: User-approved
- Context: image-grounded caption factuality, caption wording versus golden taxonomy,
  answer different questions. Combining them makes a correct but concise caption
  appear false when metadata is wrong (FR-042–044,
  BR-028–030).
- Decision: make two strict-schema Gemini calls: image+caption for top-level
  factuality and caption+golden attributes for mentioned-field matching. Generated
  structured attributes are retained as prediction provenance but not scored.
  Persist the branches in `reviews_v3.jsonl`, with separate v3 human overlay and
  workbook paths.
- Alternatives: retain the single combined v2 call; score generated structured
  metadata separately; add one judge's result as context to another; overwrite v2
  rows.
- Consequences: clear caption metric semantics and no evidence leakage between
  branches, at the cost of two requests per sample and coordinated
  evaluator/dashboard/export changes. Historical v2 results stay intact and cannot
  block v3 resume behavior.

## ADR-015: Split v3 review workbooks by generator method

- Status: Accepted
- Date: 2026-09-24
- Context: a combined workbook forces the reviewer to filter interleaved G3/G4/G5
  rows and obscures per-method attribute behavior (FR-046–047).
- Decision: retain the combined workbook for compatibility and add one generated
  workbook per G3, G4, and G5. Each method workbook has a caption summary and a
  complete 21-field caption-attribute detail sheet.
- Consequences: clearer manual review at the cost of three derived files. Inputs,
  generative outputs, evaluator JSONL, and overrides remain unchanged.

## Implementation reference

The current code behavior of G3, G4, and G5 is recorded in
[`g3_g5_flow_implementation_guide.md`](g3_g5_flow_implementation_guide.md).
This is a descriptive reference, not a new ADR: future changes to loop semantics,
dataset mapping, or observability still require an explicit decision where they
change approved behavior.

## ADR-016: Per-case request/token fields with terminal progress logging

- Status: Proposed
- Date: 2026-09-24
- Owner: pending user approval
- Context: mentor needs simple per-case request and token fields for all G1–G5,
  immediately in prediction output, while the user prefers terminal logs over files
  (FR-048–051).
- Decision: maintain an in-memory, thread-safe counter at each package's VLLM client
  boundary. Attach `input_token`, `output_token`, `num_request`, `num_retry`, and
  `num_error_request` to the final prediction row; print safe per-request progress to
  the terminal. Do not write request-event JSONL.
- Alternatives: retain detailed request-event JSONL; use hosted Langfuse/LangSmith;
  pipeline timers only. The first is more detailed but conflicts with the requested
  output/log simplicity; hosted tools add a dependency; timers miss retries.
- Consequences: concise outputs and no telemetry artifacts; node-level historical
  retry diagnosis is no longer retained. Exact token values remain conditional on OQ-023.
