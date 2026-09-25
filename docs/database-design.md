# Database design

No database is required for the confirmed single-image CLI workflows (FR-001,
FR-009, NFR-007). Inputs are read from a supplied file; stage artifacts live in
memory; the final result is printed or atomically written to a user-selected JSON
file. Consequently there are no entities, tables, relationships, indexes,
transactions, migrations, tenant boundaries, or database retention rules to
define.

The JSON output is an artifact, not a database record. Its proposed top-level
shape is:

```json
{
  "schema_version": "1",
  "flow": "g2_facts_then_caption",
  "model": "v-llm-v1-medium",
  "final": {
    "caption": "A person wearing a dark jacket.",
    "attributes": {}
  },
  "trace": {}
}
```

`attributes` is populated with all 21 keys at runtime. Each flow-specific `trace`
schema is documented in `docs/api-design.md` and validated before serialization.
The only atomicity boundary is the output file write: temporary file, flush/fsync,
permission `0600`, and rename (FR-011).

A database design becomes necessary if requirements add resumable batches,
multi-user review, queryable annotation history, distributed jobs, audit history,
or cross-run comparison. At that point retention, image identity, duplicates,
versioning, corrections, access control, and migration semantics must be decided
before selecting a datastore.

## Proposed review model

No datastore is added. One ephemeral `ReviewSample` is keyed by `sample_id` and
contains the image path, six golden captions, and zero-to-five `PredictionPanel`
values keyed by `(sample_id, method)`. The workbook flattens only captions and
status into prefixed columns. Source TSV/JSONL files remain authoritative; absent
panels remain explicit.

## Evaluation artifact model

No database is added. `output/evaluation/reviews.jsonl` contains one append-only
row per `(sample_id, method, prompt_version)`: `image_id`, `filepath`, copied
prediction, copied golden attributes, `caption_evaluation`, exactly 21
`attribute_evaluation` entries, `judge_model`, `latency_ms`, and `status/error`.
The main writer is the only writer, so every flush is a durable per-row boundary.
Rerun/overwrite semantics remain OQ-011.

## Local dashboard index

The dashboard does not add a database. `output/dashboard/index.json` is a generated
read model, not a source of truth or mutable record. Each value contains only:

```json
{
  "person_key": "01ede6c6a9ac45ef05f82dec41c8cedb",
  "sample_id": "2",
  "image_path": "sample/test/images/.../maximum_area.jpg"
}
```

The conceptual key is unique `person_key`. The builder validates a one-to-one
`person_key` → `sample_id` → `image_path` chain and refuses to emit an index if it
cannot prove that chain. `attributes.tsv` and `captions_merged.csv` remain
authoritative. Deleting the index is safe: regenerate it from those sources.

## Legacy v1 human-review overlay

No database is added. `output/dashboard/review_overrides.json` is a single,
atomically replaced local overlay keyed by `"{method}:{sample_id}"`. It holds only
operator-entered evaluation corrections:

```json
{
  "schema_version": "1",
  "overrides": {
    "g1:2": {
      "sample_id": "2",
      "method": "g1",
      "caption": { "verdict": "minor_error", "note": "Human review note" },
      "attributes": {
        "footwear_type": { "verdict": "incorrect", "note": "Visible slippers" }
      },
      "updated_at": "2026-09-24T00:00:00Z"
    }
  }
}
```

An override does not copy or mutate prediction/golden/Gemini fields. It may contain
only the selected evaluation identity, the allowed caption verdict/note, and known
attribute field verdict/note. A whole-file atomic replace is the write boundary;
the confirmed local single-operator scope has no concurrent writer or migration
requirement. A future shared review service would require audit entries, identity,
authorization, and conflict semantics (OQ-014).

## Proposed v2 evaluation artifacts

`output/evaluation/reviews_v2.jsonl` is append-only. Selection uses the newest row
per `(sample_id, method)`. A successful row records `evaluation_schema_version: "2"`,
provenance, prediction, golden attributes, and these branches:

```json
{
  "caption_evaluation": {
    "is_correct": true,
    "needs_human_review": false,
    "note": "Các chi tiết được nêu đều phù hợp với ảnh.",
    "checks": {"gender": {"mentioned": true, "is_correct": true, "note": "Mô tả phù hợp."}},
    "unmapped_claims": []
  },
  "attribute_evaluation": {
    "gender": {"prediction": "female", "golden": "female", "is_correct": true, "needs_human_review": false, "note": "Phù hợp với ảnh."}
  }
}
```

The abbreviated example still requires all 21 caption checks and attribute entries.
Unmentioned caption fields are false/null (`mentioned=false, is_correct=null`).
`output/dashboard/review_overrides_v2.json` is a distinct atomic overlay keyed by
`"{method}:{sample_id}"`; it holds only operator verdicts/notes and never copies or
mutates source data. No database or multi-user conflict semantics are introduced.

The v2 human overlay is not implemented in this evaluator-only scope. The only new
runtime artifact is the v2 evaluator JSONL; existing dashboard overlay is untouched.

### V2 review overlay and workbook (proposed)

`review_overrides_v2.json` has `schema_version: "2"` and is keyed by
`"{method}:{sample_id}"`. It may contain a human caption factuality boolean plus
note, optional known caption-check/unmapped-claim corrections, and optional known
attribute `is_correct: true|false|null` plus note. Omitted entries are not human
judgments. The Excel file is derived output only and can be deleted/regenerated.

### Isolated judge provenance (proposed)

The v2 row shape remains unchanged, but `caption_evaluation` is produced only from
image plus caption while `attribute_evaluation` is produced only from image plus
generated/golden attributes. A new prompt version distinguishes rerun rows.

## Proposed v3 evaluation artifacts

`output/evaluation/reviews_v3.jsonl` is append-only and selected newest-first by
`(sample_id, method)`. It stores `evaluation_schema_version: "3"` and a
`prompt_versions` object for the two independently versioned judge calls:

```json
{
  "caption_evaluation": {
    "is_correct": true,
    "needs_human_review": false,
    "note": "Caption phù hợp với các chi tiết nhìn thấy trong ảnh."
  },
  "caption_attribute_evaluation": {
    "gender": {"mentioned": true, "is_correct": true, "note": "Caption nêu giới tính phù hợp nhãn."},
    "bag_type": {"mentioned": false, "is_correct": null, "note": "Caption không đề cập túi."}
  }
}
```

`caption_attribute_evaluation` always contains all 21 known fields. Generated
structured attributes remain under immutable `prediction.attributes` but have no v3
evaluation map. The v3 overlay and workbook
are `review_overrides_v3.json` and `evaluation_review_v3.xlsx`; they are separate
from v2 to preserve provenance and avoid resume collisions.

The derived v3 workbooks may additionally be split by method as
`evaluation_review_g3_v3.xlsx`, `evaluation_review_g4_v3.xlsx`, and
`evaluation_review_g5_v3.xlsx`. They contain no new source data and can be safely
deleted and regenerated from JSONL plus overlay.

The final generator records and the transient state intentionally not persisted in
those records are documented in [`g3_g5_flow_implementation_guide.md`](g3_g5_flow_implementation_guide.md).

## Proposed generation telemetry artifact

No database is added. `output/telemetry/<run_id>/request_events.jsonl` is an
append-only local event artifact, one row per VLLM HTTP attempt. Its compound
logical identity is `(run_id, logical_call_id, http_attempt)` and it is joined
read-only with `predictions.jsonl` by `(run_id, method, sample_id)`. Prompts, images,
credentials and raw provider payloads are excluded (BR-031). See
[`generation_benchmark_telemetry_design.md`](generation_benchmark_telemetry_design.md).
