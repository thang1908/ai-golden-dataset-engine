# G5 — question-answer refinement

For each image, G5 generates a draft, generates verification questions, answers
them from the image, and compares the answers with the draft. A refine decision
regenerates the draft with corrections; the default maximum is two refinements.

It reads `sample/test`, shared root `.env`, and writes final JSONL rows to
`output/g5/predictions.jsonl`. No metrics or automated evaluation are produced.

```bash
source .venv/bin/activate
python -m g5_qa_refinement
```

