"""Run with python -m c1_vllm_medium --help."""

import argparse
import asyncio
import sys

from c1_vllm_medium.runner import load_settings, run
from golden_dataset_harness.baselines.io import (
    BaselineConfigError,
    add_input_arguments,
    discover_images,
    print_summary,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="C1: direct V-LLM Medium attribute extraction")
    add_input_arguments(parser)
    parser.add_argument("--model", help="Exact Medium model ID; or set C1_VLLM_MODEL")
    args = parser.parse_args(argv)
    try:
        paths = discover_images(args.image, args.input_dir)
        settings = load_settings(args.model)
        summary = asyncio.run(run(paths, args.output_dir, settings, overwrite=args.overwrite))
    except BaselineConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError):
        print("Run failed; check output permissions and runtime configuration.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed rows remain in predictions.jsonl.", file=sys.stderr)
        return 130
    return print_summary(summary, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
