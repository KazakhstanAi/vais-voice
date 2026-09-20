"""Synthetic dataset generation CLI (planning is separate from execution)."""

import argparse
from pathlib import Path

from vais_voice.generation.inventory import extract_text_inventory, write_inventory
from vais_voice.generation.plan import create_plan
from vais_voice.generation.report import build_report
from vais_voice.generation.run import run_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--manifest", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--languages", nargs="+", default=["kk", "ru"])
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--existing-policy", choices=["skip", "fail", "regenerate"])
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            rows = extract_text_inventory(args.manifest, languages=set(args.languages))
            write_inventory(args.output, rows)
            result = args.output
        elif args.command == "plan":
            result = create_plan(args.config)
        elif args.command == "run":
            result = run_plan(args.plan, run_dir=args.output, existing_policy=args.existing_policy)
        else:
            result = build_report(args.run)
        print(result)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Generation {args.command} failed: {exc}\n")


if __name__ == "__main__":
    main()
