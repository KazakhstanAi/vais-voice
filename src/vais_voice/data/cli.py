"""Unified research command line."""

import argparse
import importlib
import sys


def main() -> None:
    modules = {
        "prepare": "data.prepare",
        "split": "data.split",
        "predict": "predict",
        "evaluate": "evaluate",
        "train": "train",
        "doctor": "doctor",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=modules)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv = [f"vais-voice {args.command}", *sys.argv[2:]]
    importlib.import_module(f"vais_voice.{modules[args.command]}").main()
