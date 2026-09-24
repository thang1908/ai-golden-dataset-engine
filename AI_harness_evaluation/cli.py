from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_DOTENV_PATH, load_settings
from .dataset import load_tasks
from .errors import EvaluationError
from .gemini_client import GeminiJudge
from .output import completed_keys, persist_row
from .parallel import bounded_parallel
from .pipeline import evaluate_task
from .prompts import CAPTION_ATTRIBUTE_PROMPT_VERSION

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("g1", "g2", "g3", "g4", "g5")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate G1-G5 annotations with Gemini")
    result.add_argument("--test-dir", type=Path, default=ROOT / "sample" / "test")
    result.add_argument("--outputs-root", type=Path, default=ROOT / "output")
    result.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "evaluation" / "reviews_v3.jsonl",
    )
    result.add_argument("--dotenv", type=Path, default=DEFAULT_DOTENV_PATH)
    result.add_argument("--model", dest="GEMINI_MODEL")
    result.add_argument("--timeout", type=float, dest="GEMINI_TIMEOUT_SECONDS")
    result.add_argument("--max-tokens", type=int, dest="GEMINI_MAX_TOKENS")
    result.add_argument("--max-retries", type=int, dest="GEMINI_MAX_RETRIES")
    result.add_argument("--workers", type=int, default=1)
    result.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    result.add_argument(
        "--resume",
        action="store_true",
        help="Skip successful (sample_id, method) rows",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in (
            "GEMINI_MODEL",
            "GEMINI_TIMEOUT_SECONDS",
            "GEMINI_MAX_TOKENS",
            "GEMINI_MAX_RETRIES",
        )
    }
    try:
        settings = load_settings(dotenv_path=args.dotenv, overrides=overrides)
        tasks = load_tasks(args.test_dir, args.outputs_root, tuple(args.methods))
        target = args.output
        completed = (
            completed_keys(
                target,
                schema_version="3",
                caption_attribute_prompt_version=CAPTION_ATTRIBUTE_PROMPT_VERSION,
            )
            if args.resume else set()
        )
        tasks = [task for task in tasks if (task.sample_id, task.method) not in completed]
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.resume else "w"
        judge = GeminiJudge(settings)
        success = 0
        with target.open(mode, encoding="utf-8") as output:
            worker = lambda task: evaluate_task(task, judge, model=settings.model)
            for row in bounded_parallel(tasks, worker, args.workers):
                persist_row(output, row)
                success += row["status"] == "success"
        print(f"Evaluation generated {success}/{len(tasks)} review rows: {target}")
        return 0 if success == len(tasks) else 1
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1
