from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .client import VllmClient
from .config import DEFAULT_DOTENV_PATH, load_settings
from .dataset import generate_dataset, load_query_samples
from .errors import G4Error
from .output import json_line, write_atomically
from .parallel import bounded_parallel
from .pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4: generator, critic, verifier")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "sample" / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "g4")
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--model", dest="G4_MODEL")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--max-retries", type=int, dest="VLLM_MAX_RETRIES")
    parser.add_argument("--workers", type=int, default=1, help="Maximum images processed concurrently (default: 1)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        settings = load_settings(
            dotenv_path=args.dotenv,
            overrides={"G4_MODEL": args.G4_MODEL, "VLLM_MAX_RETRIES": args.VLLM_MAX_RETRIES},
        )
        samples = load_query_samples(args.test_dir)
        target = args.output_dir / "predictions.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        success = 0
        with target.open("w", encoding="utf-8") as output, VllmClient(settings) as client:
            worker = lambda sample: generate_dataset(
                [sample],
                lambda image: run(image, client, max_attempts=args.max_attempts),
                model=settings.model,
            )[0]
            for row in bounded_parallel(samples, worker, args.workers):
                output.write(json_line(row))
                output.flush()
                os.fsync(output.fileno())
                success += row["status"] == "success"
        print(f"G4 generated {success}/{len(samples)} predictions: {target}")
        return 0 if success == len(samples) else 1
    except (G4Error, OSError, ValueError) as exc:
        print(f"G4 error: {exc}", file=sys.stderr)
        return 1
