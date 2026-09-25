from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .client import VllmClient
from .config import DEFAULT_DOTENV_PATH, load_settings
from .dataset import generate_dataset, load_query_samples
from .errors import G5Error
from .output import json_line, write_atomically
from .parallel import bounded_parallel
from .pipeline import run
from .telemetry import TelemetryWriter, default_run_id

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G5: question-answer refinement")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "sample" / "test")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "g5")
    parser.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV_PATH)
    parser.add_argument("--model", dest="G5_MODEL")
    parser.add_argument("--max-refinements", type=int, default=20)
    parser.add_argument("--max-retries", type=int, dest="VLLM_MAX_RETRIES")
    parser.add_argument("--workers", type=int, default=1, help="Maximum images processed concurrently (default: 1)")
    parser.add_argument("--telemetry-dir", type=Path, default=ROOT / "output" / "telemetry")
    parser.add_argument("--run-id", help="Identifier shared by prediction rows and request telemetry")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(
            dotenv_path=args.dotenv,
            overrides={"G5_MODEL": args.G5_MODEL, "VLLM_MAX_RETRIES": args.VLLM_MAX_RETRIES},
        )
        samples = load_query_samples(args.test_dir)
        target = args.output_dir / "predictions.jsonl"
        run_id = args.run_id or default_run_id("g5")
        telemetry_path = args.telemetry_dir / run_id / "g5_request_events.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        success = 0
        with target.open("w", encoding="utf-8") as output, TelemetryWriter(telemetry_path) as telemetry, VllmClient(settings, telemetry=telemetry) as client:
            worker = lambda sample: generate_dataset(
                [sample],
                lambda image, sample_id, telemetry_run_id: run(image, client, max_refinements=args.max_refinements, sample_id=sample_id, run_id=telemetry_run_id),
                model=settings.model,
                run_id=run_id,
                metrics_for_sample=lambda sample_id: telemetry.sample_summary(run_id=run_id, method="g5", sample_id=sample_id),
            )[0]
            for row in bounded_parallel(samples, worker, args.workers):
                output.write(json_line(row))
                output.flush()
                os.fsync(output.fileno())
                success += row["status"] == "success"
        print(f"G5 generated {success}/{len(samples)} predictions: {target}\nrun_id: {run_id}\nrequest telemetry: {telemetry_path}")
        return 0 if success == len(samples) else 1
    except (G5Error, OSError, ValueError) as exc:
        print(f"G5 error: {exc}", file=sys.stderr)
        return 1
