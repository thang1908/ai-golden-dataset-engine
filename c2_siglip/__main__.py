"""Run with python -m c2_siglip --help; no remote credentials required."""

import argparse
import asyncio
import sys

from c2_siglip.runner import SiglipLoadError, run
from golden_dataset_harness.baselines.io import (
    BaselineConfigError,
    add_input_arguments,
    discover_images,
    print_summary,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="C2: local SigLIP person attribute inference")
    add_input_arguments(parser)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--threshold", type=float, default=0.2, help="Multi-label sigmoid threshold",
    )
    args = parser.parse_args(argv)
    try:
        paths = discover_images(args.image, args.input_dir)
        summary = asyncio.run(run(
            paths, args.output_dir, device=args.device,
            threshold=args.threshold, overwrite=args.overwrite,
        ))
    except BaselineConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except SiglipLoadError as exc:
        print(f"SigLIP setup error: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError):
        print("Run failed; check output permissions and runtime configuration.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; completed rows remain in predictions.jsonl.", file=sys.stderr)
        return 130
    return print_summary(summary, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
