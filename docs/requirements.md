# Requirements: five generative annotation flows

## Purpose and scope

This design covers five command-line pipelines that turn one person image into a
caption and the existing 21 person attributes. The immediate goal is automated
label generation for later manual inspection. The implementation will reuse the
request, OAuth, image-normalization, strict JSON Schema, validation, retry, and
atomic-output behavior observed in `c1_vllm_medium`.

In scope:

- one independently runnable folder per flow;
- `v-llm-v1-medium` by default, without inheriting a shared Large model setting;
- single-image CLI operation and JSON output containing final and useful
  intermediate artifacts;
- exact compatibility with `c1_vllm_medium/attribute_schema.json`;
- bounded retries for provider failures and bounded refinement loops.

Out of scope for this phase:

- offline or online AI evaluation, quality scores, accuracy/F1, judge metrics,
  auto-accept thresholds, or benchmark reports;
- a web API, UI, database, queue, distributed worker, or Label Studio integration;
- executing the pipelines against the real model as part of implementation;
- batch processing unless requested after the five single-image flows exist.

## Requirement sources

| Source | Classification | Contribution |
|---|---|---|
| User message, 2026-09-22 | Confirmed product requirement | Five supplied pipeline topologies; manual review now; use V-LLM Medium; code but do not run the model |
| `c1_vllm_medium` | Observed current behavior | OAuth, Chat Completions contract, model default, image preparation, taxonomy validation, retry, CLI, output safety |
| `c1_vllm_medium/attribute_schema.json` | Existing data contract | 21 attributes and allowed class codes |
| This design package | Proposed technical design | Folder names, intermediate schemas, loop bounds, module boundaries |

## Actors and external systems

| Actor/system | Role | Permission/trust boundary |
|---|---|---|
| Operator | Runs a local CLI, supplies an image, reads JSON | May read local images and output; owns credentials in `.env` |
| V-LLM service | Generates structured artifacts | External service receives normalized image and prompts |
| OAuth service | Issues short-lived bearer tokens | External service receives client credentials and project ID |
| Local filesystem | Stores input and optional output | Output is written atomically with mode `0600`, matching C1 |

There is no tenant or application-role model in this CLI-only scope.

## Functional requirements

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-001 | Provide five separate top-level Python packages, one for each supplied flow. | Each package supports `python -m <package> --image IMAGE [--output FILE]` and does not import another flow package. |
| FR-002 | Every successful flow emits a final non-empty caption plus exactly 21 valid attributes. | Output passes the vendored taxonomy validator and has no extra attribute fields. |
| FR-003 | Flow 1 performs one direct image-to-caption-and-attributes call. | Trace contains exactly one successful generation stage. |
| FR-004 | Flow 2 runs visual analysis, structured-fact extraction, then caption generation. | Output records all three artifacts; the caption stage consumes structured facts instead of the image. |
| FR-005 | Flow 3 runs four independent observations, forms consensus attributes, then writes a caption. | Output preserves four candidates, consensus evidence, final attributes, and final caption. |
| FR-006 | Flow 4 runs generator, critic, and verifier and regenerates rejected drafts. | Acceptance terminates immediately; rejection feeds structured issues into the next generator call; attempts are bounded. |
| FR-007 | Flow 5 generates a draft, generates verification questions, answers them from the image, compares claims, and accepts or refines. | Output preserves questions, answers, comparison, revisions, and the final annotation; refinements are bounded. |
| FR-008 | Reuse C1-compatible provider access. | Requests use OAuth Client Credentials, `/v1/chat/completions`, Base64 JPEG image parts where needed, `response_format.type=json_schema`, and `enable_thinking=false`. |
| FR-009 | Preserve intermediate artifacts for manual inspection. | JSON has `flow`, `model`, `final`, and a flow-specific `trace` object without scores presented as evaluation metrics. |
| FR-010 | Reject malformed model output. | Invalid JSON, missing/extra fields, invalid class codes, wrong label types, duplicate multi-label values, and invalid decisions raise a safe typed error. |
| FR-011 | Avoid partial output files. | `--output` uses same-directory temporary write, fsync, permission `0600`, and atomic rename. |
| FR-012 | Do not run real model calls during implementation verification. | Tests use request/body construction, pure validation, and mocked HTTP transport only. |

## Business rules and invariants

| ID | Rule |
|---|---|
| BR-001 | The default model is exactly `v-llm-v1-medium`; a flow-specific model setting may override it, but shared `VLLM_MODEL` must not silently select Large. |
| BR-002 | All 21 taxonomy keys are required and no unknown key is allowed. |
| BR-003 | A single-label value is an allowed string or `null`; a multi-label value is a unique array of allowed strings or `null`. |
| BR-004 | `[]` means visibly no applicable multi-label class; `null` means not determinable; `unknown` and `none` are used only when the taxonomy permits them. |
| BR-005 | Captions describe visible appearance, clothing, and carried items; they do not infer identity, relationships, location, intent, or hidden details. |
| BR-006 | Provider response bodies, tokens, client secrets, and full data URLs never appear in user-facing errors or normal logs. |
| BR-007 | Retryable HTTP status codes remain `408, 409, 429, 500, 502, 503, 504`; HTTP 401 invalidates the token once within the bounded request attempt budget. |
| BR-008 | Refinement loops always terminate. The configured defaults are 20 generator attempts for Flow 4 and 20 refinement rounds for Flow 5. |
| BR-009 | A verifier/comparator decision is workflow control, not an evaluation result; no confidence or quality score is generated in this phase. |
| BR-010 | A failed required stage produces no successful final annotation. A diagnostic trace may be retained only in memory; the CLI exits non-zero. |

## Workflows and failure paths

1. The operator invokes one package with an image path.
2. Configuration is resolved in CLI override, process environment, `.env`, default
   order; secrets have no CLI option.
3. The image is decoded, converted to RGB JPEG, resized/compressed within C1's
   limits, and represented as a data URL.
4. The selected flow performs its bounded sequence of structured model calls.
5. Every stage is validated before the next stage consumes it.
6. The final annotation is serialized to stdout or atomically written.

Failures: missing configuration, unsupported image, authentication failure,
exhausted transient retries, non-retryable provider error, malformed structured
output, or exhausted workflow refinement without an accepted/valid final result.
All failures return a non-zero CLI status and a redacted message.

## Data classification

| Data | Classification | Handling |
|---|---|---|
| Client secret/access token | Secret | Environment or ignored `.env`; memory only; never output |
| Person image | Potentially personal/sensitive | Read locally; sent to configured V-LLM; not copied by the pipeline |
| Caption/attributes/trace | Derived annotation data | Written only when requested, mode `0600` |
| Prompts and taxonomy | Non-secret configuration | Version controlled |

Retention, consent, jurisdiction, and provider data-use policies are not supplied;
see OQ-003.

## Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-001 | Python 3.11+ and the repository's current `httpx` and Pillow stack. |
| NFR-002 | Deterministic contracts: strict JSON Schema plus local validation at every model boundary. |
| NFR-003 | Reliability: token caching, bounded exponential backoff, `Retry-After`, and atomic output, consistent with C1. |
| NFR-004 | Operability: optional verbose stage-level logs to stderr; JSON alone on stdout. |
| NFR-005 | Security: least secret exposure and no raw provider error body. |
| NFR-006 | Testability: model transport is injectable; all tests can run without network or credentials. |
| NFR-007 | Simplicity: no LangGraph, database, queue, or server is required for these linear, local flows. |
| NFR-008 | Latency/cost targets are not yet defined; each output records stage count so manual comparison is possible without calling it evaluation. |

## Traceability

| Workflow | Requirements | Proposed package |
|---|---|---|
| Direct | FR-001–003, FR-008–012 | `g1_direct_annotation` |
| Facts first | FR-001–002, FR-004, FR-008–012 | `g2_facts_then_caption` |
| Four-run consensus | FR-001–002, FR-005, FR-008–012 | `g3_consensus_annotation` |
| Critic/verifier | FR-001–002, FR-006, FR-008–012 | `g4_critic_verifier` |
| Question/answer refinement | FR-001–002, FR-007–012 | `g5_qa_refinement` |

The generation and evaluation interfaces remain CLI/JSON artifacts. The separate
local image-lookup dashboard is defined below; it does not replace the spreadsheet
review workspace.

## Proposed review workspace (2026-09-23)

FR-013: a local read-only command shall join each query image and all six golden
caption variants (EN/VI × short/medium/long) from `sample/test` with available
`output/g1`–`output/g5` caption rows by `sample_id`.

FR-014: it shall create one Excel worksheet with six golden-caption columns
followed by caption EN, caption VI, and status columns for each flow; missing/error
rows are shown, not dropped.

FR-015: it shall not calculate metrics, choose a winner, mutate source files, or
make model/network calls.

## Gemini evaluation extension

The existing no-evaluation rule still applies inside G1–G5. The proposed separate
evaluator adds:

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-016 | Evaluate a successful prediction from the query image, matching golden attributes, and generated caption/attributes. | One review record per `(sample_id, method)`. |
| FR-017 | Use Gemini multimodal input and structured output through the official Python SDK. | Image plus bounded text context produces locally validated JSON; `gemini-2.5-pro` is the configurable default. |
| FR-018 | Persist an audit-friendly JSONL review artifact. | Each row has exact image identity, copied inputs, caption result, 21 attribute results, model/prompt version, status, and latency. |
| FR-019 | Preserve human adjudication. | Gemini can return `needs_human_review`, `not_visible`, or `golden_needs_review`; it never mutates input data. |
| FR-020 | Stream results. | Completed review rows are flushed before subsequent work. |

| ID | Rule |
|---|---|
| BR-011 | `GEMINI_API_KEY` is only read from environment or ignored `.env`; it never appears in a CLI argument, log, error, or artifact. |
| BR-012 | `output/review/captions_merged.csv` selects the image by `sample_id`. A missing or ambiguous mapping is terminal; the evaluator never guesses from image folder order. |
| BR-013 | The Gemini result is a recommendation and not an automatic release, correction, or golden-label mutation. |

## Proposed local image-lookup dashboard

The dashboard is a local, read-only visual aid. Its confirmed primary interaction
is entering a full identity code such as `01ede6c6a9ac45ef05f82dec41c8cedb` and
viewing the one image selected for that identity.

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-021 | Provide a gallery of mapped dataset images; selecting one image resolves its identity. | The operator opens evaluation without entering a code or browsing dataset folders. |
| FR-022 | Resolve a key through `sample/test/attributes.tsv` (`person_key` → `person_id`) and the existing `output/review/captions_merged.csv` mapping (`sample_id` → `image_path`). | The displayed image path is the existing exact mapped path, never a guessed crop or folder-order selection. |
| FR-023 | Generate a compact browser-consumable local index from those two authoritative files. | The generator rejects duplicate, missing, or unreadable mappings and reports the affected key. |
| FR-024 | Display a clear empty, invalid/not-found, stale-index, and missing-image state. | No failed lookup silently shows an unrelated image. |
| FR-025 | Remain read-only with respect to datasets, golden labels, G outputs, and evaluation artifacts. | The dashboard writes only its generated index under `output/dashboard/`. |

| ID | Rule |
|---|---|
| BR-014 | Matching is by exact `person_key`, then exact `sample_id`; partial text, filename heuristics, and image-folder order are not valid identity resolution. |
| BR-015 | The dashboard has no API key, model call, upload, mutation, or external network dependency. It is intended to be served only from the local machine. |

### UI inventory

One desktop-first screen is required: a gallery of mapped images, selection state,
and the resolved image with its `person_key`, `sample_id`, and relative `image_path`
for auditability. Captions, generated outputs, and Gemini review panels are added by
the editable-review extension below.

## Proposed editable evaluation review dashboard

The operator has now requested that the local dashboard display evaluation data and
allow manual corrections before exporting an Excel review artifact. This extends,
but does not alter, the original G1–G5 and Gemini source outputs.

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-026 | After exact `person_key` lookup, show available G1–G5 evaluation methods and let the operator select one. | The selected `(sample_id, method)` loads its latest row from `output/evaluation/reviews.jsonl`, or an explicit no-evaluation state. |
| FR-027 | Show the mapped image, machine caption EN/VI, Gemini caption verdict/issues/note, and all 21 attribute prediction/golden/verdict/note values that appear in the Excel export. | The screen presents the same source fields as the Summary and Attributes worksheets for the selected evaluation. |
| FR-028 | Let the operator save a human caption verdict/note and a human verdict/note for each attribute. | Invalid verdict values are rejected locally; a saved override is visible after page reload. |
| FR-029 | Provide an export action that creates an Excel file containing the latest evaluation rows plus the saved human-override columns. | The exported workbook preserves machine source columns and adds explicit human-review columns; it never silently replaces a Gemini value. |
| FR-030 | Keep generator JSONL, Gemini review JSONL, golden TSV, and source Excel read-only. | Human input is written only to a dashboard override artifact. |

| ID | Rule |
|---|---|
| BR-016 | A dashboard override is identified by exact `(sample_id, method)` and may only reference the 21 known taxonomy fields. |
| BR-017 | Machine prediction, golden attribute, Gemini verdict, and generated caption values are immutable in the dashboard; human corrections are a separate overlay with their own labels. |
| BR-018 | Only valid caption labels and attribute verdict enums may be saved. Empty human fields mean no override and must not be exported as a fabricated judgment. |
| BR-019 | The review UI and its write/export endpoints are loopback-only, single-operator utilities; they do not provide multi-user authentication or collaborative conflict resolution. |

### Extended UI inventory

The lookup screen gains a method selector, source-evaluation summary, caption
review editor, 21-row attribute review editor, explicit Save action, save/error
state, and Export Excel action. Source values are visually read-only; only fields
headed “Human verdict” and “Human note” are editable. Missing evaluations, pending
saves, invalid form values, write failure, and export failure are explicit states.

## AI Harness evaluation contract v2 (proposed)

This supersedes the subjective caption scale (`correct`, `minor_error`,
`major_error`, `invalid`) for new runs. The scope is factuality, not caption
completeness: a caption may omit visible details and still be correct.

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-031 | Evaluate the generated caption against its exact image, generated attributes, and golden attributes. | Each success has caption factuality, a Vietnamese note, 21 caption field checks, and explicit unmapped claims. |
| FR-032 | Evaluate each structured generated attribute independently. | Each of 21 fields records prediction, golden, `is_correct: true|false|null`, Vietnamese note, and human-review flag. |
| FR-033 | Do not penalize omitted caption fields. | An unmentioned field is `mentioned=false, is_correct=null` and is excluded from correctness denominators. |
| FR-034 | Mark caption false only for a contradicted or hallucinated claim. | Top-level caption factuality is false only when at least one evaluated claim is false. |
| FR-035 | Keep human changes separate from Gemini source output. | Dashboard overlay and Excel have explicit source/human fields. |
| FR-036 | Calculate factuality without treating unknown evidence as an error. | Accuracy uses boolean results only; coverage and unresolved rate are separate metrics. |

| ID | Rule |
|---|---|
| BR-020 | `null` means insufficient/conflicting visual or reference evidence; it is neither true nor false. |
| BR-021 | `mentioned=false` requires `is_correct=null`; omission is never an error. |
| BR-022 | Low caption coverage cannot make a factual caption fail. |
| BR-023 | Claims outside the 21-field taxonomy (such as person count) are retained as explicit unmapped claims. |
| BR-024 | All judge notes are Vietnamese; schema keys stay machine-readable English. |

### V2 workflow

The loader exact-joins image, generated caption/attributes, and golden attributes by
`sample_id`; Gemini returns strict structured JSON; a local semantic validator
enforces BR-020–023; one JSONL row is flushed; dashboard/Excel/report consume the
validated result. New output is proposed as `output/evaluation/reviews_v2.jsonl` to
avoid mixing it with historical rows using the old severity schema.

### Scope correction — evaluator only

FR-031–036 change only `AI_harness_evaluation/` and its tests. G1–G5 are existing,
read-only producers under `output/g1` through `output/g5`; they are not rebuilt,
rerun, imported, or changed. Dashboard, Excel export, and reporting were out of the
initial evaluator-only change; the v2 review-consumer extension below re-enables the
first two only.

### V2 review consumer extension (proposed)

The user has now requested that the existing local Excel export and dashboard consume
the completed evaluator v2 output; G1–G5 remain read-only inputs.

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-037 | Export the newest v2 evaluation per `(sample_id, method)` to Excel. | Workbook reads `reviews_v2.jsonl`, displays caption factuality/checks and 21 tri-state attribute results, and writes no source change. |
| FR-038 | Display and edit the newest v2 review in the local dashboard. | The selected image/method shows v2 source values and saves human tri-state corrections in a separate v2 overlay. |
| FR-039 | Keep v1 and v2 review artifacts isolated. | Dashboard/export defaults use only v2 paths; historical v1 files are neither rewritten nor silently merged. |

| ID | Rule |
|---|---|
| BR-025 | A blank human field means no human override; explicit human `null` is a saved undetermined judgment. |
| BR-026 | Human review may change only human factuality/note fields for known caption checks, unmapped claims, and 21 attributes; source Gemini/prediction/golden fields remain immutable. |

### Caption/attribute judge isolation (proposed)

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-040 | Judge caption factuality from query image and generated English caption only. | The caption Gemini request contains no generated attributes and no golden attributes. |
| FR-041 | Judge structured attributes in a separate request. | The attribute Gemini request contains image, generated attributes, and golden attributes, but no generated caption. |

| ID | Rule |
|---|---|
| BR-027 | Caption and attribute decisions are independent evidence products. An attribute result cannot make a caption false. |

### Two-branch evaluation protocol v3 (approved)

FR-040 and FR-041 are superseded for new runs by the following more explicit
two-branch contract. V2 artifacts remain immutable historical output.

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-042 | Judge overall caption factuality from the exact image and generated English caption only. | `caption_evaluation` has one `is_correct: true|false`, one Vietnamese note, and optional human-review flag; no generated or golden attributes are supplied to this Gemini call. |
| FR-043 | Check only attribute claims that the caption actually expresses against golden attributes. | `caption_attribute_evaluation` has all 21 fields. Every field records `mentioned`; unmentioned fields have `is_correct=null`; mentioned fields are matched only against the corresponding golden value. This call receives caption plus golden attributes and no image. |
| FR-044 | Preserve historical v2 results and do not let `--resume` suppress the new protocol. | Fresh runs write `output/evaluation/reviews_v3.jsonl`; `--resume` appends only missing successful keys. V3 dashboard overlay and workbook use distinct v3 paths. |

| ID | Rule |
|---|---|
| BR-028 | Top-level caption factuality means only “every claim made in the caption is grounded by the image”; caption brevity and omitted visible details never make it false. |
| BR-029 | Caption-to-golden checks answer only whether a mentioned taxonomy claim agrees with the golden field; they are not image-factuality verdicts. |
| BR-030 | The evaluator does not score generated structured attributes in v3. `null` remains excluded from caption-attribute correctness denominators. |

### Method-specific Excel export v3 (approved)

| ID | Requirement | Acceptance condition |
|---|---|---|
| FR-046 | Export one review workbook per evaluated method: G3, G4, and G5. | The exporter writes `evaluation_review_g3_v3.xlsx`, `evaluation_review_g4_v3.xlsx`, and `evaluation_review_g5_v3.xlsx` without mixing methods. |
| FR-047 | Make every caption-to-golden attribute verdict directly reviewable. | Each workbook includes per-sample, per-attribute rows with `mentioned`, golden value, Gemini `true|false|null`, Vietnamese note, and human override. |

The v3 dashboard and workbook display the two evidence products separately: image
caption factuality and caption-vs-golden attribute claims. Generated structured
attributes are retained as read-only prediction provenance, not a v3 score. Human
overrides remain a separate provenance overlay.

## Current G3–G5 implementation reference

The code-level behavior of the three active generative flows is documented in
[`g3_g5_flow_implementation_guide.md`](g3_g5_flow_implementation_guide.md). It
distinguishes observed implementation behavior from the requirements above,
including node inputs, call counts, retry behavior, loop limits, and output states.
