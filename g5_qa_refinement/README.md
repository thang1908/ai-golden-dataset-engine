# G5 — question-answer refinement

For each image, G5 generates a draft, generates verification questions, answers
them from the image, and compares the answers with the draft. A refine decision
regenerates the draft with corrections; the default maximum is 20 refinements.

It reads `sample/test`, shared root `.env`, and writes final JSONL rows to
`output/g5/predictions.jsonl`. No metrics or automated evaluation are produced.

```bash
source .venv/bin/activate
python -m g5_qa_refinement --workers 1
```

`--workers` defaults to `1`. It is intentionally conservative for G5 because one
image can perform many refinement calls and increase HTTP 429 risk. If capacity
allows, try `--workers 2`; completed rows are streamed safely to one JSONL file.
HTTP 429 retries follow shared `VLLM_MAX_RETRIES` in root `.env`, or can be
overridden for one run with `--max-retries 10`.
