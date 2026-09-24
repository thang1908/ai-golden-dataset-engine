# G3 — four-run attribute consensus

G3 is independent of G1 and G2. Per query image, it performs:

```text
Image → Observer 1/2/3/4 → Attribute consensus → Caption
```

The four observers use different evidence-focus prompts. A fifth model call sees
the image plus all candidates and selects the final 21 attributes. A sixth,
text-only call creates the caption from those final attributes.

G3 reads images directly from `sample/test` and shared credentials from root `.env`.
It writes final results only—without metrics—to `output/g3/predictions.jsonl`.

```bash
source .venv/bin/activate
python -m g3_consensus_annotation --workers 2
```

The model defaults to `v-llm-v1-medium`; shared `VLLM_MODEL` is ignored.
`--workers` defaults to `1`; use a small value because each image invokes six model
calls. Completed rows are streamed safely to the one JSONL output file.
