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
| BR-008 | Refinement loops always terminate. Proposed defaults are three generator attempts for Flow 4 and two refinement rounds for Flow 5. |
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

There is no UI/UX inventory: the confirmed interface is a CLI and JSON artifact.

