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

