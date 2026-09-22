#!/usr/bin/env python3
"""The whole API in one file.

python examples/quickstart.py path/to/person_crop.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from siglip_infer import TextEncoder, VisionEncoder


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <person_crop.jpg>", file=sys.stderr)
        return 2
    crop = Path(sys.argv[1])

    # ---- (1) image -> embedding + 21 attributes -------------------------
    vision = VisionEncoder()  # device="auto": CUDA when available, else CPU
    result = vision.encode(crop)

    print(f"embedding: {result.embedding.shape[0]}-d, L2-normalized")
    print("attributes:")
    for prediction in result.attributes:
        if prediction.is_multi_label:
            value = ", ".join(prediction.labels) or "—"
        else:
            value = f"{prediction.label} ({prediction.confidence:.0%})"
        print(f"  {prediction.attribute.label:<22} {value}")

    # Only the codes are stable. Natural names are for display; store and
    # filter on `prediction.code` / `prediction.codes`.
    print("\nmachine-readable:", result.attributes["gender"].code)

    # ---- (2) text -> embedding ------------------------------------------
    text = TextEncoder()
    queries = [
        "a person wearing a red shirt",
        "người đeo ba lô",  # Vietnamese goes straight in; no translation step
        "an empty street",
    ]
    scores = text.encode_batch(queries) @ result.embedding

    print("\nsimilarity to this crop (cosine, higher is closer):")
    for query, score in sorted(
        zip(queries, (float(s) for s in scores), strict=True), key=lambda pair: -pair[1]
    ):
        print(f"  {score:+.4f}  {query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
