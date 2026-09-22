# siglip_infer

Standalone SigLIP inference for person search. Two capabilities, one shared
768-dimensional embedding space:

1. **Image → embedding + attributes.** A person crop in; an L2-normalized 768-d
   vector and 21 predicted attributes out, with natural class names.
2. **Text → embedding.** A query string in (any language — the vocabulary is
   multilingual); an L2-normalized 768-d vector out.

Because both towers emit unit vectors in the same space, cosine similarity is a
plain dot product: `float(text_vector @ image_vector)`.

This bundle is self-contained. It has no dependency on the parent OmniSeek
project and needs nothing at runtime except the packages in
`requirements.txt`.

This is package **2.0.0 / model v2**: the epoch-35 retrieval checkpoint plus
the best post-trained detached MLP attribute head. Its embedding space is not
compatible with the former epoch-100 package; existing image vectors must be
re-embedded before text or image retrieval uses this version.

---

## Install

```bash
pip install -r requirements.txt
```

For GPU inference, swap the runtime (the two packages conflict — do not install
both) and pass `--device cuda`:

```bash
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
```

Verify the whole bundle end to end — both towers, real inference, checked
outputs:

```bash
python -m siglip_infer selftest
```

The self-test verifies every model artifact's byte count and SHA-256 before
loading either tower, which prevents a mixed old/new model directory from
silently producing meaningless cross-modal similarities.

## Quick start

```python
from siglip_infer import VisionEncoder, TextEncoder

vision = VisionEncoder()  # device="auto": CUDA if present, else CPU
result = vision.encode("crop.jpg")  # path | PIL.Image | numpy array | bytes

result.embedding  # (768,) float32, L2-normalized
result.attributes["gender"].code  # "female"   <- store and filter on this
result.attributes["gender"].label  # "Female"   <- show this
result.attributes["gender"].confidence  # 0.97
result.attributes["bag_type"].labels  # ("Backpack", "Tote bag")
result.attribute_logits  # (200,) raw logits, if you want your own thresholds

text = TextEncoder()
query = text.encode("người mặc áo đỏ")  # (768,) float32, L2-normalized
similarity = float(query @ result.embedding)
```

Batching, and cropping to a detector box:

```python
results = vision.encode_batch(["a.jpg", "b.jpg"], batch_size=16)
vectors = text.encode_batch(["áo đỏ", "blue jeans"])

# box is (left, top, right, bottom); fractional 0-1 or pixel coordinates
result = vision.encode("frame.jpg", box=(0.31, 0.12, 0.44, 0.78))
```

There is a runnable version of all of the above in `examples/quickstart.py`.

## Command line

```bash
python -m siglip_infer image crop.jpg                  # table of attributes
python -m siglip_infer image crop.jpg --json           # embedding + attributes as JSON
python -m siglip_infer image *.jpg --embedding-out embeddings.npy
python -m siglip_infer text "người mặc áo đỏ" --json
python -m siglip_infer match crop.jpg "a man with a backpack" "a woman in a red dress"
python -m siglip_infer taxonomy                        # the full 200-logit layout
python -m siglip_infer selftest
```

Example:

```
$ python -m siglip_infer image crop.jpg
crop.jpg
  embedding: 768-d, unit norm
  Age                   Young adult (0.85)
  Gender                Male (1.00)
  Upper clothing type   Shirt (long sleeve) (0.99)
  Upper clothing color  Pink (0.98)
  Lower clothing type   Dress trousers (0.94)
  Lower clothing color  Black (1.00)
  ...
  Bag type              Backpack (0.96)
```

## What is in here

```
siglip_infer/
├── models/
│   ├── siglip_vitb_attr.onnx     vision tower + attribute head  (357 MB)
│   ├── siglip_text.onnx          text tower                     (1.1 GB)
│   ├── siglip_tokenizer.json     SigLIP2 tokenizer               (33 MB)
│   └── siglip_manifest.json      checkpoint provenance + preprocessing contract
├── assets/
│   ├── attribute_taxonomy.json   ← logit index → attribute + natural class name
│   └── attribute_schema.json     the upstream contract the taxonomy is generated from
├── siglip_infer/
│   ├── vision.py                 VisionEncoder
│   ├── text.py                   TextEncoder, tokenizer, canonicalization
│   ├── preprocess.py             the resize/normalize contract
│   ├── taxonomy.py               taxonomy loading + logit decoding
│   ├── runtime.py                ONNX Runtime session and device selection
│   └── cli.py
├── tools/build_taxonomy.py       regenerates assets/attribute_taxonomy.json
├── examples/quickstart.py
└── tests/
```

The model files are **not in git** (`models/.gitignore`); they travel with the
archive.

## The attribute taxonomy

The model emits one flat vector of 200 logits. `assets/attribute_taxonomy.json`
is what gives it meaning — it maps every logit index to an attribute and a
natural class name, and it is the file to read if you are consuming attributes
from another language or service.

```json
{
  "name": "upper_clothing_type",
  "label": "Upper clothing type",
  "type": "single_label",
  "display_tier": "strong",
  "logit_start": 11,
  "logit_stop": 36,
  "classes": [
    { "index": 0, "logit_index": 11, "code": "tshirt_short_sleeve", "label": "T-shirt (short sleeve)" },
    { "index": 1, "logit_index": 12, "code": "tshirt_long_sleeve",  "label": "T-shirt (long sleeve)" }
  ]
}
```

It also carries a flat `index_to_class` array — 200 entries, one per logit — for
consumers that want a direct lookup rather than a slice walk.

**21 attributes, 200 logits, in this fixed order:**

| # | attribute | classes | type | tier |
|---|-----------|--------:|------|------|
| 0 | age | 6 | single | weak |
| 1 | gender | 2 | single | strong |
| 2 | body_build | 3 | single | hidden |
| 3 | upper_clothing_type | 25 | single | strong |
| 4 | upper_clothing_color | 15 | single | strong |
| 5 | lower_clothing_type | 16 | single | strong |
| 6 | lower_clothing_color | 15 | single | strong |
| 7 | clothing_style | 5 | single | hidden |
| 8 | upper_pattern | 7 | single | moderate |
| 9 | footwear_type | 13 | single | moderate |
| 10 | bag_type | 11 | **multi** | unvalidated |
| 11 | bag_color | 15 | **multi** | unvalidated |
| 12 | headwear | 8 | single | moderate |
| 13 | eyewear | 4 | single | moderate |
| 14 | face_mask | 5 | single | weak |
| 15 | other_accessories | 8 | **multi** | unvalidated |
| 16 | hair_length | 6 | single | strong |
| 17 | hair_texture | 3 | single | hidden |
| 18 | hairstyle | 5 | single | strong |
| 19 | hair_color | 12 | single | hidden |
| 20 | carried_objects | 16 | **multi** | unvalidated |

### Decoding

* **single-label** — softmax over the attribute's slice, then argmax.
* **multi-label** — independent sigmoid per class; every class at or above
  `multilabel_threshold` (0.20, measured for binary macro F1 on the private
  human-labelled evaluation set) is selected.

### Three states that must not be conflated

| state | meaning |
|-------|---------|
| no result at all | inference never ran |
| `"unknown"` | the model ran and judged the attribute unclear — **this is an answer** |
| empty multi-label list | the model ran and found none (this person carries no bag) |

A filter on `unknown` must return people the model found unclear, never people
who were never described.

### Display tiers

`display_tier` comes from measured lift over an always-predict-the-majority
baseline, **not** from raw accuracy. Two attributes have the highest raw accuracy
in the whole set and carry almost no information; a UI that ranks by accuracy
leads with exactly those.

| tier | what a UI may do |
|------|------------------|
| `strong` | display prominently; safe as a search facet |
| `moderate` | display, confidence-gated; a positive is more informative than a negative |
| `weak` | supporting detail only — never a filter |
| `unvalidated` | per-class reliability unmeasured; display without prominence |
| `hidden` | do not display or filter — the model adds nothing over a constant |

`gender` is confidence-gated despite being `strong`: it has no `unknown` class,
so the head must pick a value even from a crop that cannot support one.
`prediction.passes_gate(0.35)` applies this rule.

## The preprocessing contract

The exported graph starts at a normalized `(batch, 3, 384, 128)` tensor.
Everything before that lives in `preprocess.py` and must match training exactly:

```
RGB → squash-resize to 384×128 (bicubic) → /255 → (x − 0.5) / 0.5 → CHW float16
```

"Squash" means the aspect ratio is **not** preserved — a person crop is stretched
to 3:1. That is what the model was trained on, so an aspect-preserving or
letterboxed resize is a bug, not an improvement.

⚠️ Every plausible variation of this — letterboxing, bilinear interpolation,
ImageNet mean/std, BGR channel order, an un-rotated EXIF photo — still produces a
finite unit-norm embedding and a full set of confident attributes. Nothing
raises. The results are just quietly worse, and no downstream check can detect
it. Use `preprocess_image` rather than reimplementing the resize.

The same applies to text: the tokenizer reproduces OpenCLIP's
`basic_clean` + `canonicalize` byte-for-byte, at context length 64. A query
cleaned differently still returns a perfectly well-formed 768-d vector that
still ranks results — it just answers a slightly different question than the one
the image embeddings were trained against.

## Model provenance

`models/siglip_manifest.json` records the exact checkpoint, its SHA-256, and the
preprocessing contract. In short:

| | |
|---|---|
| base weights | `ViT-B-16-SigLIP2-naflex` |
| package / model | `2.0.0` / `v2` |
| retrieval checkpoint | epoch 35, fixed-resolution 384×128 (**not** a NaFlex model) |
| attribute head | post-trained detached MLP `768 → 512 → 200`, best checkpoint at epoch 7 |
| embedding | 768-d, L2-normalized **inside** the graph |
| text context length | 64 tokens, 256k multilingual vocabulary |
| ONNX opset | 18 |

Image and text embeddings are only comparable when they come from the same
export. Changing the encoder means re-embedding everything already indexed.
The manifest pins the retrieval checkpoint, detached head, and all shipped
artifacts by SHA-256.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

Tests that need the model weights are marked `models` and skip cleanly without
them, so the taxonomy, preprocessing, and canonicalization tests run on any
checkout:

```bash
python -m pytest tests -q -m "not models"
```

Two conventions carried over from the parent project:

* **Tests name the failure they prevent.** `test_a_wrong_length_logit_vector_is_
  fatal_rather_than_truncated` pins a specific way the system can be wrong while
  looking correct. When one fails, the fix is almost never to change the test.
* **`⚠️` in a comment marks a silent failure mode** — one that produces plausible
  output rather than an error.

## Regenerating the taxonomy

`assets/attribute_taxonomy.json` is generated and checked in; the runtime never
needs the tool. Regenerate it after changing a label or the schema:

```bash
python tools/build_taxonomy.py
python tools/build_taxonomy.py --check   # CI: fail if the checked-in file is stale
```

⚠️ Attribute order in the schema **is** the logit layout. Reordering it silently
reinterprets every attribute the model has ever predicted — the loader and
`tools/build_taxonomy.py` both refuse a taxonomy whose slices do not tile
0–199 exactly.
