"""Batch pipeline runner.

Discovers images in a directory, builds the annotation graph once, and
processes all images with bounded concurrency.

Usage::

    python -m golden_dataset_harness.workflow.runner \\
        --input-dir ./sample_images/ \\
        --settings ./configs/settings.yaml \\
        --concurrency 4 \\
        --output-dir ./output/
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from golden_dataset_harness.schemas.annotation import AnnotationRecord
from golden_dataset_harness.workflow.graph import build_annotation_graph, load_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _discover_images(image_dir: str) -> list[Path]:
    """Find all image files in a directory (non-recursive)."""
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return images


async def _process_one(
    image_path: Path,
    graph: Any,
    semaphore: asyncio.Semaphore,
) -> AnnotationRecord | None:
    """Process a single image through the pipeline."""
    async with semaphore:
        image_id = image_path.stem
        initial_state = {
            "image_path": str(image_path),
            "image_id": image_id,
        }
        try:
            result = await graph.ainvoke(initial_state)
            annotation = result.get("annotation")
            if annotation:
                logger.info("Completed: %s → %s", image_id, annotation.review_status)
            return annotation
        except Exception:
            logger.exception("Failed to process %s", image_path)
            return None


async def run_batch(
    image_dir: str,
    settings_path: str | None = None,
    max_concurrency: int = 4,
    output_dir: str | None = None,
) -> list[AnnotationRecord]:
    """Process all images in a directory through the annotation pipeline.

    Args:
        image_dir: Path to directory containing image files.
        settings_path: Path to settings YAML (uses default if None).
        max_concurrency: Maximum number of images processed in parallel.
        output_dir: If set, write results to ``dataset.jsonl`` in this directory.

    Returns:
        List of successfully generated AnnotationRecords.
    """
    settings = load_settings(settings_path)
    graph = build_annotation_graph(settings)

    images = _discover_images(image_dir)
    logger.info("Discovered %d images in %s", len(images), image_dir)

    if not images:
        logger.warning("No images found — nothing to process")
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [_process_one(img, graph, semaphore) for img in images]
    results = await asyncio.gather(*tasks)

    annotations = [r for r in results if r is not None]
    failed = len(results) - len(annotations)

    logger.info(
        "Batch complete: %d succeeded, %d failed out of %d total",
        len(annotations), failed, len(results),
    )

    # Write JSONL output
    if output_dir and annotations:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        jsonl_path = out / "dataset.jsonl"
        with open(jsonl_path, "w") as f:
            for record in annotations:
                f.write(record.model_dump_json() + "\n")
        logger.info("Wrote %d records to %s", len(annotations), jsonl_path)

    return annotations


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the annotation pipeline on a directory of images",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing images")
    parser.add_argument("--settings", default=None, help="Path to settings.yaml")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent tasks")
    parser.add_argument("--output-dir", default="./output", help="Output directory for JSONL")

    args = parser.parse_args()
    asyncio.run(
        run_batch(
            image_dir=args.input_dir,
            settings_path=args.settings,
            max_concurrency=args.concurrency,
            output_dir=args.output_dir,
        )
    )
