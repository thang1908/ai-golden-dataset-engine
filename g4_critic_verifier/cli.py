from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .client import VllmClient
from .config import DEFAULT_DOTENV_PATH, load_settings
from .dataset import generate_dataset, load_query_samples
from .errors import G4Error
from .output import json_line, write_atomically
from .pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4: generator, critic, verifier")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "sample" / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "g4")
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--model", dest="G4_MODEL")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        settings = load_settings(dotenv_path=args.dotenv, overrides={"G4_MODEL": args.G4_MODEL})
        samples = load_query_samples(args.test_dir)
        with VllmClient(settings) as client:
            rows = generate_dataset(samples, lambda image: run(image, client, max_attempts=args.max_attempts), model=settings.model)
        target = args.output_dir / "predictions.jsonl"
        write_atomically(target, "".join(json_line(row) for row in rows))
        success = sum(row["status"] == "success" for row in rows)
        print(f"G4 generated {success}/{len(rows)} predictions: {target}")
        return 0 if success == len(rows) else 1
    except (G4Error, OSError, ValueError) as exc:
        print(f"G4 error: {exc}", file=sys.stderr)
        return 1

