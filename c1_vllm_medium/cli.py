"""Thin command line adapter: configure, call once, print or write attributes."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from c1_vllm_medium.api import VllmAttributeClient
from c1_vllm_medium.config import DEFAULT_DOTENV_PATH, load_settings
from c1_vllm_medium.errors import C1Error
from c1_vllm_medium.output import serialize_annotation, write_atomically


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="C1: direct V-LLM Medium vision extraction with JSON Schema output"
    )
    result.add_argument("--image", type=Path, required=True, help="Person image/crop")
    result.add_argument(
        "--dotenv",
        type=Path,
        default=DEFAULT_DOTENV_PATH,
        help="Shared .env file (default: repository .env)",
    )
    result.add_argument("--base-url", dest="C1_BASE_URL")
    result.add_argument("--client-id", dest="C1_CLIENT_ID")
    result.add_argument("--project-id", dest="C1_PROJECT_ID")
    result.add_argument("--model", dest="C1_MODEL")
    result.add_argument("--timeout", type=float, dest="C1_TIMEOUT_SECONDS")
    result.add_argument("--max-tokens", type=int, dest="C1_MAX_TOKENS")
    result.add_argument("--max-retries", type=int, dest="C1_MAX_RETRIES")
    result.add_argument("--output", type=Path, help="Write JSON attributes atomically to this file")
    result.add_argument("--verbose", action="store_true", help="Write operational logs to stderr")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    overrides = {
        name: getattr(args, name)
        for name in (
            "C1_BASE_URL", "C1_CLIENT_ID", "C1_PROJECT_ID", "C1_MODEL",
            "C1_TIMEOUT_SECONDS", "C1_MAX_TOKENS", "C1_MAX_RETRIES",
        )
    }
    try:
        settings = load_settings(dotenv_path=args.dotenv, overrides=overrides)
        with VllmAttributeClient(settings) as client:
            annotation = client.annotate(args.image.read_bytes())
        payload = serialize_annotation(annotation)
        if args.output:
            write_atomically(args.output, payload)
        else:
            print(payload, end="")
    except (C1Error, OSError, ValueError) as exc:
        print(f"C1 error: {exc}", file=sys.stderr)
        return 1
    return 0
