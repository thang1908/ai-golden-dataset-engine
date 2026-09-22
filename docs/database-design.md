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

