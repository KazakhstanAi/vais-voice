"""Extract a pinned FLEURS parquet split into the reviewed local adapter layout."""

import argparse
from pathlib import Path

from vais_voice.data.fleurs_extract import extract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--maximum", type=int)
    args = parser.parse_args()
    try:
        print(f"Extracted {extract(args.source, args.destination, args.split, args.maximum)} rows")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Extraction failed: {exc}\n")


if __name__ == "__main__":
    main()
