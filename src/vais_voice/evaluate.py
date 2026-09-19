"""Evaluate supplied test predictions. Checkpoint inference is a later milestone."""

import argparse
from pathlib import Path

from vais_voice.evaluation.report import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.checkpoint:
        parser.exit(
            2,
            "Checkpoint inference is not implemented; no checkpoint is loaded. "
            "Use --predictions with existing, independently generated scores.\n",
        )
    if not all([args.predictions, args.manifest, args.preparation]):
        parser.error("--predictions, --manifest and --preparation are required")
    try:
        report = evaluate_predictions(
            args.manifest,
            args.predictions,
            args.preparation,
            args.output,
            args.project_root,
        )
        print(report["overall"])
        print(f"Report: {args.output / 'report.md'}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
