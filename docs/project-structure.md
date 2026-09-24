# Project structure

## Current repository constraints

The repository is a Python 3.11 project. `c1_vllm_medium` is an independent package
with small responsibility-based modules (`config`, `auth`, `image`, `taxonomy`,
`api`, `output`, `cli`) and a `python -m` entrypoint. Root tests use pytest and
mocked `httpx` transport. The working tree already contains unrelated user changes
under `c1_vllm_medium/sample`, `golden_dataset_harness/evaluation`, and a deleted
sample image; implementation must not alter them.

## Proposed tree

Legend: `[E]` existing, `[P]` proposed, `[G]` runtime-generated.

```text
.
├── c1_vllm_medium/                         [E] reference only
├── g1_direct_annotation/                   [P] flow 1 package
│   ├── __init__.py, __main__.py, cli.py
│   ├── config.py, auth.py, image.py, client.py
│   ├── taxonomy.py, contracts.py, prompts.py, pipeline.py
│   ├── output.py, errors.py
│   ├── attribute_schema.json, .env.example, requirements.txt, README.md
├── g2_facts_then_caption/                  [P] same shell; G2 contracts/pipeline/prompts
├── g3_consensus_annotation/                [P] same shell; G3 contracts/pipeline/prompts
├── g4_critic_verifier/                     [P] same shell; G4 contracts/pipeline/prompts
├── g5_qa_refinement/                       [P] same shell; G5 contracts/pipeline/prompts
├── tests/
│   ├── flow_test_helpers.py                [P] test-only builders and mock transport
│   ├── test_g1_direct_annotation.py        [P]
│   ├── test_g2_facts_then_caption.py       [P]
│   ├── test_g3_consensus_annotation.py     [P]
│   ├── test_g4_critic_verifier.py          [P]
│   └── test_g5_qa_refinement.py            [P]
├── output/                                 [G] optional operator-selected outputs
├── docs/                                   [P] reviewed design package
└── pyproject.toml                           [E, update package discovery in implementation]
```

"Same shell" means each folder owns equivalent adapter modules so it is runnable
and understandable independently; it does not mean symlinks or imports across
flows. Only test fixtures may be shared. This intentionally accepts small adapter
duplication to honor five self-contained experiments (ADR-001).

## Folder and module responsibilities

| Path/module | Responsibility | Public interface |
|---|---|---|
| `gN_*/pipeline.py` | Flow-specific stage ordering and loop bounds | `run(image: bytes, client) -> FlowResult` |
| `gN_*/contracts.py` | Stage and final JSON Schemas plus local validators | Schema builders and `validate_*` |
| `gN_*/prompts.py` | Versioned, pure prompt construction | One function per stage |
| `gN_*/client.py` | Generic structured V-LLM call for image/text inputs | `complete(prompt, schema, image=None)` |
| `gN_*/taxonomy.py` | Exact vendored 21-field contract | `TAXONOMY`, `attribute_schema`, validator |
| `gN_*/config.py` | Prefix-aware settings resolution | `Settings`, `load_settings` |
| `gN_*/auth.py` | OAuth token lifecycle | `TokenProvider` |
| `gN_*/image.py` | Decode/resize/compress image | `to_data_url` |
| `gN_*/output.py` | Serialization and atomic write | `serialize_result`, `write_atomically` |
| `gN_*/cli.py` | CLI adapter and safe errors | `main` |
| `tests/flow_test_helpers.py` | Test-only data and mocked responses | No production import allowed |

## Organizing principle and dependencies

Each flow is a small vertical package. Dependency direction is:

```text
__main__ -> cli -> pipeline -> client -> auth/config
                    |    |       |
                    |    |       +-> image
                    |    +-> prompts/contracts/taxonomy
                    +-> output/contracts
```

Adapters must not import the CLI. Contracts and prompts must be pure and must not
perform HTTP or filesystem I/O. A flow package must not import another flow or
`c1_vllm_medium`; C1 is a behavioral reference, not a runtime dependency. Test code
may parameterize equivalent behavior but production `common`/`utils` is prohibited
until at least three stable, identical implementations exist and independent-flow
ownership is explicitly relaxed.

## Entrypoints, schemas, generated data, and secrets

- Entrypoint: each `__main__.py` delegates to `cli.main()`.
- Model schemas: generated in `contracts.py`; taxonomy source is the vendored
  `attribute_schema.json` in each folder.
- Prompts: named stage functions in `prompts.py`; prompt revision constants make
  trace output interpretable.
- Output: operator chooses `--output`; otherwise stdout. No default generated file.
- Secrets: root `.env` or an explicitly selected dotenv file, never committed.
- Caches/build artifacts: `.venv`, `__pycache__`, `.pytest_cache`, `.ruff_cache` stay
  ignored and are never treated as inputs.

## Naming conventions

Packages/modules/functions use `snake_case`; classes use `PascalCase`; JSON fields
use `snake_case`. Environment prefixes are proposed as `G1_` through `G5_`, with
fallback only to shared connection/credential variables (`VLLM_BASE_URL`,
`VLLM_CLIENT_ID`, `VLLM_CLIENT_SECRET`, `VLLM_PROJECT_ID`, timeout/token/retry
settings). The model setting never falls back to `VLLM_MODEL` (BR-001).

Flow output names are `g1_direct_annotation`, `g2_facts_then_caption`,
`g3_consensus_annotation`, `g4_critic_verifier`, and `g5_qa_refinement`.

## Test structure

- Unit: prompt purity, JSON Schema shape, local semantic validators, routing and
  loop termination.
- HTTP contract: OAuth request, model request, retry/401 behavior, error redaction.
- Pipeline integration: mocked ordered responses assert stage inputs and final
  trace, with no network.
- CLI: stdout/output path, exit codes, and secret-safe errors.
- No end-to-end real-model test is run in this phase (FR-012).

## Architecture-to-source mapping

| Component/deployable unit | Source | Build/deploy boundary |
|---|---|---|
| G1 executable | `g1_direct_annotation/` | Local Python package |
| G2 executable | `g2_facts_then_caption/` | Local Python package |
| G3 executable | `g3_consensus_annotation/` | Local Python package |
| G4 executable | `g4_critic_verifier/` | Local Python package |
| G5 executable | `g5_qa_refinement/` | Local Python package |
| OAuth/V-LLM | External | Not built or deployed here |

## Extension examples

- New model stage: add its prompt and contract, then call it from that flow's
  `pipeline.py`; do not put orchestration into `client.py`.
- Batch job: add a separate `batch.py` per chosen flow after concurrency semantics
  are confirmed; keep single-image `pipeline.run` unchanged.
- API endpoint/database/entity: not applicable to current scope; add transport and
  persistence layers only after requirements exist.
- Sixth AI flow: create a sixth vertical package with its own contract and tests.

## Transition steps after approval

1. Scaffold the five packages and vend the exact schema.
2. Implement shared behavioral shell independently in each package.
3. Implement and mock-test G1, then G2–G5 stage contracts and routing.
4. Add packages/package data to `pyproject.toml` and add README commands.
5. Run static/unit/mock tests only; do not invoke the configured V-LLM endpoint.

Alternative shared-core structures are recorded in ADR-001.

## Gemini evaluation package

```text
AI_harness_evaluation/
├── __main__.py, cli.py, config.py
├── dataset.py                 # query/attribute/prediction join
├── gemini_client.py           # only Gemini SDK/provider boundary
├── prompts.py, contracts.py, pipeline.py, output.py
└── parallel.py                # bounded concurrent execution

tests/test_ai_harness_evaluation.py # mocked Gemini transport and local fixtures
```

`AI_harness_evaluation/` reads G1–G5 output but no generator imports it. The command
is `python -m AI_harness_evaluation`. `dataset.py` uses the existing
`output/review/captions_merged.csv` sample-to-image mapping and rejects missing or
ambiguous rows; it never guesses image order from an identity folder.

## Proposed local dashboard structure

```text
dashboard/                              # [I] static, local UI
├── index.html                           # [I] lookup screen
├── app.js                               # [I] exact-key lookup and UI states
└── styles.css                           # [I] minimal desktop-first presentation

tools/build_dashboard_index.py           # [I] validates and produces the index
tools/serve_dashboard.py                 # [I] narrow loopback-only static server
output/dashboard/index.json              # [G] generated person_key → image mapping
```

`dashboard/` contains no secrets, source data copy, model client, or backend.
`tools/build_dashboard_index.py` is the only new writer and produces only the
generated index after validating `sample/test/attributes.tsv`,
`output/review/captions_merged.csv`, and each referenced image file. Static assets
are served by the narrow loopback server together with the allowed test-image
paths; the repository root itself is never served.

## Proposed dashboard review extension

```text
dashboard/
├── index.html                           # [M] review controls and result regions
├── app.js                               # [M] lookup, load, edit, save, export UI
└── styles.css                           # [M] readable source/editing states

tools/serve_dashboard.py                 # [M] narrow static files plus local review API
tools/export_evaluation_to_xlsx.mjs      # [M] merges human overrides into Excel output
output/dashboard/review_overrides.json   # [G] atomic human-review overlay
tests/test_dashboard_review.py            # [P] mapping, validation, read/write API tests
```

The review extension remains a local vertical feature: browser code calls only
`serve_dashboard.py`; that server owns reads/writes of the override artifact and
invokes the exporter. The exporter reads source evaluation JSONL plus overrides but
does not import dashboard browser code. Generated override and Excel files are not
source inputs to G1–G5 or Gemini evaluation.

## AI Harness evaluation v2 transition (proposed)

```text
AI_harness_evaluation/
├── contracts.py             # [M] v2 schema + semantic truth table
├── prompts.py               # [M] factuality-only Vietnamese judge instructions
├── pipeline.py              # [M] v2 JSONL envelope and nullable summaries
├── cli.py, output.py         # [M] reviews_v2.jsonl selection/default
├── gemini_client.py          # [M] existing provider boundary receives v2 schema
└── dataset.py                # [E] exact image/prediction/golden join

tools/export_evaluation_to_xlsx.mjs       # [M] v2 worksheets and human columns
tools/build_g3_g5_evaluation_report.py    # [M] factuality, accuracy, coverage
tools/serve_dashboard.py                   # [M] v2 review endpoint validation
dashboard/                                 # [M] v2 source and human review fields
tests/test_ai_harness_evaluation.py        # [M] schema and null-invariant tests
output/evaluation/reviews_v2.jsonl         # [G] v2 results
output/dashboard/review_overrides_v2.json  # [G] v2 human overlay
```

`contracts.py` is the only owner of result invariants. Presentation/reporting code
may consume validated rows but cannot reinterpret a null as an error. V1 remains a
legacy artifact and is not mixed with v2 by default.

Only evaluator contracts, prompt, pipeline/CLI/output as necessary, and evaluator
tests are modified. Dashboard/export/report paths are deliberately excluded.

### V2 consumer extension (proposed)

```text
dashboard/
├── app.js, index.html, styles.css          # [M] v2 tri-state source/editor fields
tools/
├── serve_dashboard.py                      # [M] reads/saves v2 review/overlay
└── export_evaluation_to_xlsx.mjs           # [M] exports v2 review workbook
tests/
└── test_dashboard_review.py                # [P] v2 load/override validation tests
output/dashboard/review_overrides_v2.json   # [G] human v2 overlay
output/evaluation/evaluation_review_v2.xlsx # [G] v2 workbook
```

The server owns validation and atomic overlay writes. Browser code never reads raw
filesystem paths other than allowed images/index, and the Node exporter never imports
browser code. These modules depend on the v2 evaluation contract but not Gemini SDK
or any G-flow package.

### Isolated judge implementation (proposed)

`prompts.py` gains separate caption/attribute builders; `contracts.py` owns separate
schemas; `gemini_client.py` accepts the selected schema; `pipeline.py` performs and
combines two validated calls. Tests assert that caption context cannot contain
generated or golden attribute payloads.

### Two-branch v3 transition (approved)

```text
AI_harness_evaluation/
├── prompts.py                         # [M] two isolated prompt builders
├── contracts.py                       # [M] two response schemas + invariants
├── gemini_client.py                   # [M] selected response schema per call
├── pipeline.py                        # [M] two-call orchestration, one row flush
├── cli.py, output.py                  # [M] reviews_v3.jsonl defaults
└── tests/test_ai_harness_evaluation.py # [M] context-isolation and null tests

tools/serve_dashboard.py                # [M] v3 review API/read model
tools/export_evaluation_to_xlsx.mjs     # [M] v3 separate branch worksheets
dashboard/                              # [M] v3 labels and separate panels
output/evaluation/reviews_v3.jsonl      # [G] v3 source evaluation
output/dashboard/review_overrides_v3.json # [G] v3 human overlay
output/evaluation/evaluation_review_v3.xlsx # [G] derived v3 workbook
```

Existing v2 files remain read-only historical artifacts. No G1–G5 module, output,
or test image is modified by this transition.

`tools/export_evaluation_to_xlsx.mjs --split` writes generated method-specific
v3 workbooks under `output/evaluation/`; it remains a passive consumer of evaluator
JSONL and dashboard overrides.

The detailed code-reading map for the three active generator packages is
[`docs/g3_g5_flow_implementation_guide.md`](g3_g5_flow_implementation_guide.md).
