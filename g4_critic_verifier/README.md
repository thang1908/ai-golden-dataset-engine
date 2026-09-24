# G4 — generator, critic, verifier

For each image, G4 generates a caption plus 21 attributes, critiques the draft
against the image, then verifies it. A rejection regenerates with structured
feedback; the default maximum is 20 attempts. Only accepted drafts are written.

It reads `sample/test`, shared root `.env`, and writes final JSONL rows to
`output/g4/predictions.jsonl`. No metrics or automated evaluation are produced.

```bash
source .venv/bin/activate
python -m g4_critic_verifier --workers 2
```

`--workers` defaults to `1`. Each image keeps its generator → critic → verifier
sequence, while images run independently; only the main process writes streamed
rows to `predictions.jsonl`.

HTTP 429 retries follow shared `VLLM_MAX_RETRIES` in root `.env`, or can be
overridden for one run with `--max-retries 10`.
