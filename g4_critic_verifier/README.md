# G4 — generator, critic, verifier

For each image, G4 generates a caption plus 21 attributes, critiques the draft
against the image, then verifies it. A rejection regenerates with structured
feedback; the default maximum is three attempts. Only accepted drafts are written.

It reads `sample/test`, shared root `.env`, and writes final JSONL rows to
`output/g4/predictions.jsonl`. No metrics or automated evaluation are produced.

```bash
source .venv/bin/activate
python -m g4_critic_verifier
```

