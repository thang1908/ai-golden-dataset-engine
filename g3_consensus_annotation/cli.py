from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .client import VllmClient
from .config import DEFAULT_DOTENV_PATH, load_settings
from .dataset import generate_dataset, load_query_samples
from .errors import G3Error
from .output import json_line, write_atomically
from .pipeline import run

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = REPOSITORY_ROOT / "sample" / "test"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "output" / "g3"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="G3: generate caption + 21 attributes for the supplied test dataset"
    )
    result.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="External sample directory (default: sample/test)",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder (default: output/g3)",
    )
    result.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV_PATH)
    result.add_argument("--base-url", dest="VLLM_BASE_URL")
    result.add_argument("--client-id", dest="VLLM_CLIENT_ID")
    result.add_argument("--project-id", dest="VLLM_PROJECT_ID")
    result.add_argument("--model", dest="G3_MODEL")
    result.add_argument("--timeout", type=float, dest="VLLM_TIMEOUT_SECONDS")
    result.add_argument("--max-tokens", type=int, dest="VLLM_MAX_TOKENS")
    result.add_argument("--max-retries", type=int, dest="VLLM_MAX_RETRIES")
    result.add_argument("--verbose", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    overrides = {
        name: getattr(args, name)
        for name in (
            "VLLM_BASE_URL",
            "VLLM_CLIENT_ID",
            "VLLM_PROJECT_ID",
            "G3_MODEL",
            "VLLM_TIMEOUT_SECONDS",
            "VLLM_MAX_TOKENS",
            "VLLM_MAX_RETRIES",
        )
    }
    try:
        settings = load_settings(dotenv_path=args.dotenv, overrides=overrides)
        samples = load_query_samples(args.test_dir)
        with VllmClient(settings) as client:
            rows = generate_dataset(samples, lambda image: run(image, client), model=settings.model)
        target = args.output_dir / "predictions.jsonl"
        write_atomically(target, "".join(json_line(row) for row in rows))
        succeeded = sum(row["status"] == "success" for row in rows)
        print(f"G3 generated {succeeded}/{len(rows)} predictions: {target}")
        return 0 if succeeded == len(rows) else 1
    except (G3Error, OSError, ValueError) as exc:
        print(f"G3 error: {exc}", file=sys.stderr)
        return 1


