# G2 — visual facts then caption

G2 is independent of G1. For each query image it calls `v-llm-v1-medium` three
times:

```text
Image → Visual Analysis → Structured Facts + 21 attributes → Caption
```

It reads query images directly from `sample/test/pairs_en_medium.tsv`; images are
not copied into this package. The generated final captions and attributes are
written to `output/g2/predictions.jsonl` in the same JSONL row format as G1. It does
not evaluate or compare the output.

G2 reads shared root `.env` values: `VLLM_BASE_URL`, `VLLM_CLIENT_ID`,
`VLLM_CLIENT_SECRET`, `VLLM_PROJECT_ID`, `VLLM_TIMEOUT_SECONDS`,
`VLLM_MAX_TOKENS`, and `VLLM_MAX_RETRIES`. The model defaults to
`v-llm-v1-medium` and ignores shared `VLLM_MODEL`.

Run:

```bash
source .venv/bin/activate
python -m g2_facts_then_caption
```

Optional overrides are `--test-dir`, `--output-dir`, `--dotenv`, and non-secret
connection/model/retry flags. The model is not called by tests.

