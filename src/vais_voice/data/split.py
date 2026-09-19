"""Write protected splits beside the prepared audio directory."""

import argparse
from pathlib import Path

from vais_voice.data.validate import validate_files, verify_manifest
from vais_voice.datasets.manifest import read_manifest, write_manifest
from vais_voice.datasets.split import assign_splits
from vais_voice.utils.io import sha256, write_json


def split(manifest: Path, preparation: Path) -> Path:
    prep = verify_manifest(manifest, preparation)
    validate_files(manifest)
    config = prep["config"]
    rows = assign_splits(
        read_manifest(manifest),
        seed=prep["seed"],
        train_fraction=config["train_fraction"],
        val_fraction=config["val_fraction"],
        unseen_generators=config["unseen_generators"],
    )
    output = manifest.parent
    names = ["manifest.split.parquet", "train.parquet", "val.parquet", "test.parquet"]
    if any((output / name).exists() for name in [*names, "split_preparation.json"]):
        raise ValueError("Split outputs already exist; use a new dataset version")
    write_manifest(output / names[0], rows)
    for subset in ("train", "val", "test"):
        write_manifest(output / f"{subset}.parquet", [r for r in rows if r.split == subset])
    write_json(
        output / "split_preparation.json",
        {
            **prep,
            "manifest_hashes": {name: sha256(output / name) for name in names},
            "parent_preparation_sha256": sha256(preparation),
            "counts": {s: sum(r.split == s for r in rows) for s in ("train", "val", "test")},
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(split(args.manifest, args.preparation))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Split failed: {exc}\n")


if __name__ == "__main__":
    main()
