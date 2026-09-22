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

