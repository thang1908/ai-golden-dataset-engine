# Interface and contract design

## Interface style

The confirmed public interface is a local CLI, not an HTTP application API. Each
flow provides:

```text
python -m <flow_package> --image PATH [--output PATH] [--dotenv PATH]
```

Non-secret overrides mirror C1: `--base-url`, `--client-id`, `--project-id`,
`--model`, `--timeout`, `--max-tokens`, `--max-retries`, and `--verbose`. Client
secret is intentionally absent. G4 additionally exposes `--max-attempts`; G5
exposes `--max-refinements`; safe defaults are applied when omitted.

Success writes JSON to stdout or `--output` and exits 0. Failure writes one
redacted line to stderr and exits 1. CLI paths are local filesystem paths; no
authentication or authorization layer exists beyond local OS permissions.

## External provider contracts

### OAuth

`POST {service_url}/oauth/token` with JSON `client_id`, `client_secret`, and
`project_id`. A successful response must contain non-empty `access_token` and
numeric `expires_in`. The token is cached in memory with the C1 refresh skew.

### Structured completion

`POST {service_url}/v1/chat/completions` with bearer token and JSON:

```json
{
  "model": "v-llm-v1-medium",
  "messages": [{"role": "user", "content": []}],
  "stream": false,
  "temperature": 0,
  "max_tokens": 1024,
  "chat_template_kwargs": {"enable_thinking": false},
  "response_format": {
    "type": "json_schema",
    "json_schema": {"name": "stage_specific_name", "schema": {}}
  }
}
```

For a vision stage, `content` contains text and an `image_url` data URL. For a
text-only stage, `content` contains only text. The response content must be a JSON
string at `choices[0].message.content`; schema enforcement is followed by local
validation (FR-008, FR-010).

## Common result contract

```json
{
  "schema_version": "1",
  "flow": "g1_direct_annotation",
  "model": "v-llm-v1-medium",
  "final": {
    "caption": "A person wearing a black jacket and blue jeans.",
    "attributes": {
      "age": "adult",
      "gender": null,
      "bag_type": [],
      "carried_objects": []
    }
  },
  "trace": {}
}
```

The abbreviated example omits taxonomy keys for readability only; actual output
requires all 21. Caption length is 1–500 after trimming. Attributes obey BR-002–004.
`schema_version`, `flow`, `model`, `final`, and `trace` are required; no extra
top-level fields are allowed.

## Flow-specific trace contracts

| Flow | Required trace fields |
|---|---|
| G1 | `generation` containing the same validated annotation used as `final` |
| G2 | `visual_analysis`, `structured_facts`, `caption_generation` |
| G3 | `candidates` (exactly four, each with observer role and annotation), `consensus` (per-field selected values and supporting run indices), `caption_generation` |
| G4 | `attempts` (1..max; each contains draft, critic issues, verifier decision/reasons), `termination` (`accepted` only on success) |
| G5 | `rounds` (draft, generated questions, image-grounded answers, comparison decision/corrections), `termination` (`accepted` only on success) |

Intermediate contracts:

- `visual_analysis`: arrays of visible observations and uncertainties; no identity,
  intent, or location fields.
- `structured_facts`: canonical visible facts plus exactly 21 attributes.
- `critic`: an array of `{target, issue, suggested_correction}`; it does not emit a
  numeric score.
- `verifier`: `{decision: "accept"|"reject", reasons: [...]}`.
- `question`: `{id, target, question}` with unique IDs and a configured maximum.
- `answer`: `{question_id, answer, evidence, determinable}`.
- `comparison`: `{decision: "accept"|"refine", reasons, replacement}` where
  `replacement` is null on accept and a full valid annotation on refine.

Stage schemas use `additionalProperties: false`, required fields, bounded string
and array lengths, and enums. Semantic validators enforce unique IDs, one answer
per question, legal target names, exact candidate count, valid supporting run
indices, and the 21-field taxonomy.

## Errors, retry, and rate limits

No provider error body is exposed. Error categories are configuration, image,
authentication, provider API, structured-response validation, and workflow
exhaustion. Network failures and BR-007 statuses retry with bounded exponential
backoff; `Retry-After` is honored up to the C1 cap. A malformed 2xx model response
is terminal for that stage rather than silently repaired outside the designed
flow. Rate-limit policy beyond retry is unknown (OQ-004).

## Versioning and compatibility

Output begins at `schema_version: "1"`. Additive trace details still require a
schema-version change if `additionalProperties: false` would reject them. Taxonomy
remains the exact SigLIP-compatible vendored version from C1. There are no
pagination, filtering, webhooks, streaming, uploads, or long-running-operation
HTTP contracts in scope.

## Gemini judge contract

There is no new repository HTTP API. The only external call is the official
`google-genai` Python SDK. The proposed adapter uses stateless
`models.generate_content`: one review has all needed evidence in one request, so
no conversation state, tool use, or agent is needed. It sends an inline query image
and structured text context, requests JSON MIME output with a JSON Schema, then
performs local Pydantic/semantic validation. Gemini errors or malformed content are
redacted into an error row; raw provider bodies and API keys are never persisted.

## Local dashboard delivery contract

There is no REST/GraphQL API. The narrow loopback server exposes only approved
paths to the local browser:

| Browser path | Source | Purpose |
|---|---|---|
| `/dashboard/` | static UI assets | code entry and result rendering |
| `/output/dashboard/index.json` | generated index | exact local lookup data |
| `/sample/test/images/...` | mapped local image | browser-visible image bytes |

The client-side lookup contract is an exact JSON object lookup by complete
`person_key`. A missing key, missing field, unavailable index, or failed image load
is a user-visible error state, not an alternate search. The browser makes no calls
to V-LLM, Gemini, OAuth, or another network endpoint.

## Legacy v1 local review API

The loopback server adds no public API and no authentication layer; its endpoints
are available only at the same local origin as the dashboard (BR-019).

| Method | Path | Requirement | Purpose |
|---|---|---|---|
| `GET` | `/api/evaluations?sample_id={id}` | FR-026 | Return methods with latest source evaluation for an exact sample. |
| `GET` | `/api/evaluation?sample_id={id}&method={g#}` | FR-027 | Return the complete latest source row plus any human overlay. |
| `PUT` | `/api/review-override` | FR-028 | Validate then atomically save one human overlay for `(sample_id, method)`. |
| `POST` | `/api/export-review` | FR-029 | Regenerate and return the local Excel review workbook. |

`PUT /api/review-override` accepts only this shape (unknown fields are rejected):

```json
{
  "sample_id": "2",
  "method": "g1",
  "caption": { "verdict": "minor_error", "note": "Human review note" },
  "attributes": {
    "footwear_type": { "verdict": "incorrect", "note": "Visible slippers" }
  }
}
```

This endpoint contract is retained only to document the existing v1 implementation.
Valid caption verdicts are the existing caption labels from Gemini output; valid
attribute verdicts are `correct`, `incorrect`, `not_visible`, `golden_needs_review`,
and `not_applicable`. Read requests return 404 for a missing source evaluation.
Invalid input returns 400, an unavailable/malformed source returns 500, and an
exporter failure returns 500 without replacing the last successful workbook. Every
write is local-only and affects the overlay artifact, never source data.

## V2 Gemini judge and local review contracts (proposed)

Gemini continues to receive a strict `response_json_schema`. It returns a caption
branch with `is_correct`, `needs_human_review`, Vietnamese `note`, 21 `{mentioned,
is_correct, note}` checks, and bounded `{claim, is_correct, note}` unmapped claims.
It also returns 21 structured attribute `{is_correct, needs_human_review, note}`
entries; the pipeline attaches immutable prediction/golden values before persistence.

Local validation additionally enforces: an unmentioned field must be null; a false
top-level caption must have at least one false evaluated claim; a true caption has no
false evaluated claim; only booleans affect correctness denominators. The proposed
v2 review `PUT` saves a human caption boolean/null plus note and optional known
attribute boolean/null plus notes. Unknown keys or invalid combinations return 400;
source evaluation rows remain immutable.

The v2 dashboard `PUT` contract is deferred. This change exposes only
`python -m AI_harness_evaluation` and its JSONL output contract.

### V2 local review API (proposed)

The existing loopback-only endpoints keep their paths but switch to v2 artifacts:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/reviews?sample_id={id}` | Return latest v2 source rows plus optional v2 human overlay. |
| `PUT` | `/api/reviews` | Validate and atomically save one v2 human overlay. |
| `POST` | `/api/export` | Run the v2 exporter and return `evaluation_review_v2.xlsx`. |

The PUT body is bounded and exact: identity, an optional human caption
`is_correct`/note, and optional known attribute/caption-check corrections. Booleans
and `null` are valid human results; omitted values mean no override. Invalid shapes,
unknown taxonomy fields, or a missing source evaluation return 400. There is no
authentication because the server is loopback-only and single-operator.

### Isolated Gemini request contracts (proposed)

Caption request content is image plus caption prompt and has a caption-only schema.
Attribute request content is image plus attribute prompt and has an attribute-only
schema. The pipeline combines only locally validated responses into the existing
v2 envelope.

## V3 isolated Gemini request contracts (proposed)

Each task makes two strict-schema requests, with no shared prompt context:

| Branch | Inputs crossing the Gemini boundary | Persisted branch |
|---|---|---|
| Caption factuality | image, generated English `caption` | `caption_evaluation` (`is_correct`, `needs_human_review`, Vietnamese `note`) |
| Caption attribute match | generated English `caption`, golden 21-field object | `caption_attribute_evaluation` (21 `{mentioned, is_correct|null, note}` entries) |

Local validation rejects unknown fields; enforces `mentioned=false` implies
`is_correct=null`; and never derives one branch's boolean from another. Generated
structured attributes are returned only as immutable prediction provenance. The local
review endpoints retain their paths but move to v3 source/overlay/workbook paths and
return the two branch objects separately.

The local exporter also accepts a bounded method selector (`g3`, `g4`, or `g5`) or
`--split`; split output produces one workbook per selected method. No method-specific
workbook write changes the evaluation JSONL or human overlay contract.

For the local CLI call sequence and per-node VLLM request contracts of the G3–G5
generators, see [`g3_g5_flow_implementation_guide.md`](g3_g5_flow_implementation_guide.md).

## Generation counter interface

There is no new network or user-facing API. `VllmClient.complete()` receives an
internal call context (`run_id`, `method`, `sample_id`, `stage`, `logical_call_id`)
per-case counter and updates it after each HTTP attempt. The provider call remains
the same Chat Completions endpoint. The final JSONL contract adds:

```json
{"input_token": 123, "output_token": 45, "num_request": 7, "num_retry": 2, "num_error_request": 1}
```

`input_token` and `output_token` are `null` when the provider does not return usage.

Internally, `VLLM_MAX_REQUESTS_PER_MINUTE` limits Chat Completions attempts per
process. OAuth token calls are outside that limiter and prediction metric contract.
