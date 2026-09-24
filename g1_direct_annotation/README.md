# G1 — direct caption + attributes

G1 independently sends each query image in the supplied test sample to
`v-llm-v1-medium` once. It writes only generated predictions; it does not calculate
metrics or compare against `attributes.tsv`.

It reads query images directly from the external sample at
`sample/test/pairs_en_medium.tsv` (`is_query=1`) and validates that
the test dataset has the same 21-attribute contract. It never copies images into
this package. The default output is one file: `output/g1/predictions.jsonl`.
Every row follows the sample format:
`sample_id`, `method`, `model_id`, `prompt_version`, `caption`, `attributes`,
`latency_ms`, and `status`.

## Configure

G1 reads the shared repository `.env` directly, or a file passed through `--dotenv`.
It uses `VLLM_BASE_URL`, `VLLM_CLIENT_ID`, `VLLM_CLIENT_SECRET`,
`VLLM_PROJECT_ID`, `VLLM_TIMEOUT_SECONDS`, `VLLM_MAX_TOKENS`, and
`VLLM_MAX_RETRIES`.

```bash
cp g1_direct_annotation/.env.example .env
```

G1 always defaults to `v-llm-v1-medium` and deliberately ignores shared
`VLLM_MODEL`; use `--model` only when intentionally overriding Medium.

## Generate the sample predictions

```bash
python3 -m g1_direct_annotation \
  --output-dir output/g1 \
  --workers 3
```

`--test-dir` is optional and defaults to the external sample above. Then open
`output/g1/predictions.jsonl` and compare it manually with the
sample captions and `attributes.tsv`. A per-image provider failure is retained as a
JSONL row with `status: "error"`; successful rows always contain a caption and all
21 validated attributes.

`--workers` defaults to `1`. With a larger value, each worker processes one image
independently and the main process appends rows as jobs finish, so JSONL order may
vary but `sample_id` is retained and rows cannot overwrite one another.

The model is not invoked by unit tests.
