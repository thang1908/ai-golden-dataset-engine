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

## ADR-001: Five independent vertical flow packages

- Status: Proposed
- Date: 2026-09-22
- Owner: pending user approval
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

- Status: Proposed
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
- Decision: default G4 to at most three generator attempts and G5 to at most two
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

