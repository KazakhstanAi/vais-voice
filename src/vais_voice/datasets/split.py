"""Deterministic connected-component splitting: speakers, sources and derivatives."""

import hashlib
from collections import defaultdict

from vais_voice.datasets.manifest import Sample


def connected_groups(rows: list[Sample]) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners: dict[str, int] = {}
    for i, row in enumerate(rows):
        tokens = [f"speaker:{s}" for s in [row.speaker_id, *row.related_speaker_ids]]
        tokens += [f"source:{row.source_id}", f"sample:{row.sample_id}"]
        if row.parent_sample_id:
            tokens.append(f"sample:{row.parent_sample_id}")
        if row.source_sha256:
            tokens.append(f"hash:{row.source_sha256}")
        for token in tokens:
            if token in owners:
                parent[find(i)] = find(owners[token])
            else:
                owners[token] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    return list(groups.values())


def assign_splits(
    rows: list[Sample],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    unseen_generators: list[str],
) -> list[Sample]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Train and validation fractions must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("A positive test fraction is required")
    supplied = [row.split is not None for row in rows]
    if any(supplied) and not all(supplied):
        raise ValueError("Provide splits for every row or none")
    result = list(rows)
    for group in connected_groups(rows):
        holdout = any(rows[i].generator_id in unseen_generators for i in group)
        if all(supplied):
            splits = {rows[i].split for i in group}
            if len(splits) != 1:
                raise ValueError("Speaker/source/parent/duplicate leakage across supplied splits")
            split = rows[group[0]].split
            if holdout and split != "test":
                raise ValueError("Unseen generator_id component must be in test")
        else:
            key = "|".join(sorted(rows[i].sample_id for i in group))
            digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
            value = int.from_bytes(digest[:8], "big") / 2**64
            split = (
                "test"
                if holdout
                else "train"
                if value < train_fraction
                else "val"
                if value < train_fraction + val_fraction
                else "test"
            )
        for i in group:
            result[i] = rows[i].model_copy(update={"split": split})
    for split in ("train", "val", "test"):
        labels = {row.label for row in result if row.split == split}
        if labels != {"real", "synthetic"}:
            raise ValueError(
                f"{split} needs both classes. Add independent groups or supply reviewed splits; "
                "never tune the seed against model results."
            )
    return result
