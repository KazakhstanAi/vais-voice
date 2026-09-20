"""Research-partition isolation for synthetic generator families."""

from typing import Literal

from vais_voice.datasets.schema import Sample

ResearchPartition = Literal["train", "protected_test", "external_challenge"]


def validate_generator_roles(rows: list[Sample], partition: ResearchPartition) -> None:
    allowed = {
        "train": {"train", "validation"},
        "protected_test": {"unseen_test"},
        "external_challenge": {"external_challenge"},
    }[partition]
    violations = sorted(
        {
            f"{row.generator_id}:{row.intended_role or 'missing_role'}"
            for row in rows
            if row.label == "synthetic"
            and (row.intended_role is not None or row.source_text_id or row.generator_family)
            and row.intended_role not in allowed
        }
    )
    if violations:
        raise ValueError(f"Generator-role violation for {partition}: {', '.join(violations)}")
