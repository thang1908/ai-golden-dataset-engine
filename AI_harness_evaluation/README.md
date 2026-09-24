# AI harness evaluation

Evaluates successful G1–G5 predictions using Gemini with the matching query image
and human-labeled attributes. It writes one durable JSONL review row per completed
`(sample_id, method)`; workers never write the output file directly.

## Configure

Add the following to the shared root `.env` (do not commit the key):

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-pro
GEMINI_MAX_RETRIES=5
```

The required image mapping is `output/review/captions_merged.csv`. It maps each
`sample_id` to its exact `image_path`; the harness refuses to guess from an identity
folder containing three images.

## Run

```bash
python -m AI_harness_evaluation --workers 1
```

Output: `output/evaluation/reviews_v3.jsonl`.

The harness reads existing successful rows from `output/g1` through `output/g5`; it
does not invoke or change any generative flow. It evaluates factuality only:

- Caption factuality uses only the image and generated English caption. It is false
  only for a wrong or hallucinated caption claim.
- Caption-to-golden attribute matching is a second independent request. Omitted
  fields use `mentioned=false` and `is_correct=null`. A mentioned field must be
  exactly `true` or `false`; there is no “undetermined” verdict. Null is excluded
  from its correctness denominator.
- Generated JSON attributes remain stored prediction provenance; v3 does not score
  them and they never affect caption factuality.

Use `--resume` to retain existing rows and skip only successful `(sample_id, method)`
pairs produced by the current attribute-judge prompt; older prompt versions are
automatically evaluated again. Failed rows are retried.
Use `--methods g1 g4` to evaluate selected generator outputs only.

Transient network, 429, and Gemini 5xx errors retry automatically. `GEMINI_MAX_RETRIES`
is the number of retries after the initial request; use `--max-retries` to override it.

Machine labels remain English for filtering. Gemini writes all review notes in Vietnamese.
