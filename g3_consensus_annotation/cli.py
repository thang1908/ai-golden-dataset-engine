from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from .client import VllmClient
from .config import DEFAULT_DOTENV_PATH, load_settings
from .dataset import generate_dataset, load_query_samples
from .errors import G3Error
from .output import json_line, write_atomically
from .parallel import bounded_parallel
from .pipeline import run
from .telemetry import TelemetryWriter, default_run_id

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
    result.add_argument("--workers", type=int, default=1, help="Maximum images processed concurrently (default: 1)")
    result.add_argument("--telemetry-dir", type=Path, default=REPOSITORY_ROOT / "output" / "telemetry")
    result.add_argument("--run-id", help="Identifier shared by prediction rows and request telemetry")
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
        target = args.output_dir / "predictions.jsonl"
        run_id = args.run_id or default_run_id("g3")
        telemetry_path = args.telemetry_dir / run_id / "g3_request_events.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        succeeded = 0
        with target.open("w", encoding="utf-8") as output, TelemetryWriter(telemetry_path) as telemetry, VllmClient(settings, telemetry=telemetry) as client:
            worker = lambda sample: generate_dataset(
                [sample], lambda image, sample_id, telemetry_run_id: run(image, client, sample_id=sample_id, run_id=telemetry_run_id), model=settings.model, run_id=run_id,
                metrics_for_sample=lambda sample_id: telemetry.sample_summary(run_id=run_id, method="g3", sample_id=sample_id),
            )[0]
            for row in bounded_parallel(samples, worker, args.workers):
                output.write(json_line(row))
                output.flush()
                os.fsync(output.fileno())
                succeeded += row["status"] == "success"
        print(f"G3 generated {succeeded}/{len(samples)} predictions: {target}\nrun_id: {run_id}\nrequest telemetry: {telemetry_path}")
        return 0 if succeeded == len(samples) else 1
    except (G3Error, OSError, ValueError) as exc:
        print(f"G3 error: {exc}", file=sys.stderr)
        return 1
