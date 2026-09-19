"""Evaluate exact ID-matched predictions; partial coverage and unknown IDs are errors."""

import csv
import json
import math
from pathlib import Path

import yaml

from vais_voice.data.validate import verify_manifest
from vais_voice.datasets.manifest import read_manifest
from vais_voice.evaluation.metrics import detection_metrics
from vais_voice.utils.io import sha256, write_json
from vais_voice.utils.tracking import snapshot


def evaluate_predictions(
    manifest: Path,
    predictions: Path,
    preparation: Path,
    output: Path,
    project_root: Path,
) -> dict:
    rows = read_manifest(manifest)
    prep = verify_manifest(manifest, preparation)
    provenance = {"detector": "external_unverified", "checkpoint": None}
    sidecar = predictions.with_suffix(".meta.json")
    if sidecar.exists():
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        if provenance.get("manifest_sha256") != sha256(manifest) or provenance.get(
            "predictions_sha256"
        ) != sha256(predictions):
            raise ValueError("Prediction provenance checksum mismatch")
    test = {row.sample_id: row for row in rows if row.split == "test"}
    if not test:
        raise ValueError("No test samples")
    scores = {}
    with predictions.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sample_id", "synthetic_score"]:
            raise ValueError("Prediction CSV columns must be sample_id,synthetic_score")
        for item in reader:
            sample_id = item["sample_id"]
            if sample_id in scores or sample_id not in test:
                raise ValueError("Duplicate prediction ID or ID outside test split")
            score = float(item["synthetic_score"])
            if not math.isfinite(score):
                raise ValueError("Prediction scores must be finite")
            scores[sample_id] = score
    if set(scores) != set(test):
        raise ValueError("Predictions must cover exactly all test samples; no silent dropping")

    def metrics(selected: list) -> dict:
        if not selected:
            return {"status": "empty", "n": 0, "roc_auc": None, "eer": None, "fpr_at_tpr95": None}
        return detection_metrics(
            [int(row.label == "synthetic") for row in selected],
            [scores[row.sample_id] for row in selected],
        )

    selected = list(test.values())
    slices = {
        field: {
            str(value): metrics([row for row in selected if getattr(row, field) == value])
            for value in sorted({getattr(row, field) for row in selected}, key=str)
        }
        for field in ("language", "codec", "condition")
    }
    # Per-generator_id slice includes genuine reference audio, otherwise AUC/EER is undefined.
    slices["generator_with_all_real_references"] = {
        generator_id: metrics(
            [row for row in selected if row.label == "real" or row.generator_id == generator_id]
        )
        for generator_id in sorted({row.generator_id for row in selected if row.generator_id})
    }
    report = {
        **snapshot(project_root),
        "report_type": "prediction_evaluation_not_model_training",
        "inference": provenance,
        "dataset_version": prep["dataset_version"],
        "split_version": prep["split_version"],
        "seed": prep["seed"],
        "manifest_sha256": sha256(manifest),
        "predictions_sha256": sha256(predictions),
        "preparation_sha256": sha256(preparation),
        "score_direction": "higher_is_more_synthetic",
        "positive_class": "synthetic",
        "roc_convention": "linear interpolation; EER crossing; first TPR95 crossing",
        "calibration": "not_assessed",
        "overall": metrics(selected),
        "slices": slices,
        "unseen_generators": prep["config"]["unseen_generators"],
        "limitations": [
            "Prediction-only evaluation does not verify inference or checkpoint provenance.",
            "Generator comparisons use all genuine test references; inspect channel confounding.",
            "FPR@TPR95 is a descriptive test-ROC metric, not a deployment threshold.",
            "No confidence intervals or calibration estimates are implemented yet.",
        ],
    }
    artifacts = [
        "metrics.json",
        "report.md",
        "config.yaml",
        "dataset_snapshot.json",
        "experiment.json",
    ]
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError("Evaluation artifacts already exist")
    target_predictions = output / "predictions.csv"
    same_predictions = target_predictions.resolve() == predictions.resolve()
    if not same_predictions and target_predictions.exists():
        raise ValueError("Output predictions already exist")
    output.mkdir(parents=True, exist_ok=True)
    if not same_predictions:
        with target_predictions.open("xb") as stream:
            stream.write(predictions.read_bytes())
    with (output / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump({"data": prep["config"], "inference": provenance}, stream)
    write_json(
        output / "dataset_snapshot.json",
        {
            "dataset_version": prep["dataset_version"],
            "split_version": prep["split_version"],
            "manifest_sha256": sha256(manifest),
            "samples": [r.model_dump() for r in rows],
        },
    )
    write_json(
        output / "experiment.json",
        {
            **snapshot(project_root),
            "seed": prep["seed"],
            "inference": provenance,
            "checkpoint": provenance.get("checkpoint"),
            "metrics": "metrics.json",
        },
    )
    write_json(output / "metrics.json", report)
    lines = [
        "# VAIS Voice experiment report",
        "",
        "Prediction-only evaluation; not evidence of training a detector.",
        provenance.get("warning", "External scores: model provenance is unverified."),
        "",
        f"Dataset: {report['dataset_version']} / split: {report['split_version']}",
        "",
        "| Slice | N | ROC-AUC | EER | FPR@TPR95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, value in [("overall", report["overall"])] + [
        (f"{kind}/{key}", item) for kind, group in slices.items() for key, item in group.items()
    ]:
        numbers = [
            "N/A" if value[k] is None else f"{value[k]:.6f}"
            for k in ("roc_auc", "eer", "fpr_at_tpr95")
        ]
        lines.append(f"| {name} | {value['n']} | {' | '.join(numbers)} |")
    with (output / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return report
