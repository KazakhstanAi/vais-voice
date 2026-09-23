"""Reproducible detector training, validation thresholding, and protected test export."""

from __future__ import annotations

import csv
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn
from torch.utils.data import DataLoader

from vais_voice.data.validate import validate_files, verify_manifest
from vais_voice.datasets.manifest import read_manifest
from vais_voice.datasets.schema import Sample
from vais_voice.detection.dataset import ManifestAudioDataset
from vais_voice.detection.models import ModelSpec, build_model
from vais_voice.evaluation.metrics import detection_metrics, select_eer_threshold
from vais_voice.utils.io import load_yaml, sha256, write_json
from vais_voice.utils.tracking import snapshot


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)


class LoaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(default=16, ge=1)
    num_workers: int = Field(default=0, ge=0)
    require_training_eligible: bool = True


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: str = Field(min_length=1)
    seed: int = Field(ge=0)
    project_root: str
    prepared_manifest: str
    preparation_report: str
    run_dir: str
    device: Literal["auto", "cpu", "cuda"] = "auto"
    epochs: int = Field(default=20, ge=1)
    patience: int = Field(default=5, ge=1)
    amp: bool = True
    model: dict[str, Any]
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    unseen_generators: list[str] = Field(default_factory=list)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)


def _eligible(rows: list[Sample], required: bool) -> list[Sample]:
    if not required:
        return rows
    return [
        row
        for row in rows
        if row.label == "real"
        or (
            row.training_eligible is True
            and row.quality_gate in {"pass", "warning"}
            and row.diagnostic_only is not True
        )
    ]


def _require_two_classes(rows: list[Sample], name: str) -> None:
    if {row.label for row in rows} != {"real", "synthetic"}:
        raise ValueError(f"{name} needs eligible real and synthetic samples")


def _loader(
    rows: list[Sample],
    manifest: Path,
    spec: ModelSpec,
    config: LoaderConfig,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = ManifestAudioDataset(rows, manifest.parent, spec.sample_rate, spec.max_samples)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp: bool,
) -> tuple[float, list[int], list[float], list[str]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    scores: list[float] = []
    sample_ids: list[str] = []
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for waveform, target, identifiers in loader:
            waveform = waveform.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(
                device_type=device.type, enabled=amp and device.type == "cuda"
            )
            with amp_context:
                logits = model(waveform)
                loss = criterion(logits, target)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach()) * len(target)
            labels.extend(target.detach().cpu().int().tolist())
            scores.extend(torch.sigmoid(logits.detach()).cpu().tolist())
            sample_ids.extend(identifiers)
    return total_loss / len(labels), labels, scores, sample_ids


def _slice_metrics(
    rows: list[Sample], scores: dict[str, float], unseen_generators: set[str]
) -> dict[str, Any]:
    def metrics(selected: list[Sample]) -> dict[str, Any]:
        if not selected:
            return {"status": "empty", "n": 0, "roc_auc": None, "eer": None, "fpr_at_tpr95": None}
        return detection_metrics(
            [int(row.label == "synthetic") for row in selected],
            [scores[row.sample_id] for row in selected],
        )

    real = [row for row in rows if row.label == "real"]
    seen_synthetic = [
        row
        for row in rows
        if row.label == "synthetic" and row.generator_id not in unseen_generators
    ]
    unseen_synthetic = [
        row for row in rows if row.label == "synthetic" and row.generator_id in unseen_generators
    ]
    result: dict[str, Any] = {
        "overall": metrics(rows),
        "protocol": {
            "test_seen": metrics([*real, *seen_synthetic]),
            "test_unseen": metrics([*real, *unseen_synthetic]),
        },
        "language": {
            language: metrics([row for row in rows if row.language == language])
            for language in sorted({row.language for row in rows})
        },
        "generator_with_real_references": {
            generator: metrics([*real, *[row for row in rows if row.generator_id == generator]])
            for generator in sorted({row.generator_id for row in rows if row.generator_id})
        },
    }
    return result


def train_detector(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = TrainingConfig.model_validate(load_yaml(config_path))
    root = (config_path.parent / config.project_root).resolve()
    manifest = root / config.prepared_manifest
    preparation = root / config.preparation_report
    prep = verify_manifest(manifest, preparation)
    validate_files(manifest)
    rows = read_manifest(manifest)
    if any(row.split is None for row in rows):
        raise ValueError("Training requires a reviewed split manifest")
    spec = ModelSpec(**config.model)
    if spec.sample_rate != prep["config"]["sample_rate"]:
        raise ValueError("Model and prepared audio sample rates differ")
    train_rows = _eligible(
        [row for row in rows if row.split == "train"], config.loader.require_training_eligible
    )
    val_rows = _eligible(
        [row for row in rows if row.split == "val"], config.loader.require_training_eligible
    )
    test_rows = [row for row in rows if row.split == "test"]
    _require_two_classes(train_rows, "train")
    _require_two_classes(val_rows, "validation")
    _require_two_classes(test_rows, "test")
    unseen = set(config.unseen_generators or prep["config"].get("unseen_generators", []))
    if any(row.generator_id in unseen for row in train_rows + val_rows):
        raise ValueError("Unseen generator leakage into train/validation")

    output = root / config.run_dir
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model = build_model(spec).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    synthetic = sum(row.label == "synthetic" for row in train_rows)
    genuine = len(train_rows) - synthetic
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(genuine / synthetic, device=device))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    train_loader = _loader(
        train_rows, manifest, spec, config.loader, shuffle=True, seed=config.seed
    )
    val_loader = _loader(val_rows, manifest, spec, config.loader, shuffle=False, seed=config.seed)
    history: list[dict[str, Any]] = []
    best_auc = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0
    best_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss, train_labels, train_scores, _ = _run_epoch(
            model, train_loader, device, criterion, optimizer, scaler, config.amp
        )
        val_loss, val_labels, val_scores, _ = _run_epoch(
            model, val_loader, device, criterion, None, None, config.amp
        )
        train_metrics = detection_metrics(train_labels, train_scores)
        val_metrics = detection_metrics(val_labels, val_scores)
        val_auc = val_metrics["roc_auc"]
        comparable_auc = -1.0 if val_auc is None else float(val_auc)
        improved = comparable_auc > best_auc or (
            comparable_auc == best_auc and val_loss < best_loss
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train": train_metrics,
                "validation": val_metrics,
                "selected": improved,
            }
        )
        if improved:
            best_auc, best_loss = comparable_auc, val_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    _, val_labels, val_scores, _ = _run_epoch(
        model, val_loader, device, criterion, None, None, config.amp
    )
    threshold = select_eer_threshold(val_labels, val_scores)
    test_loader = _loader(test_rows, manifest, spec, config.loader, shuffle=False, seed=config.seed)
    test_loss, test_labels, test_scores, test_ids = _run_epoch(
        model, test_loader, device, criterion, None, None, config.amp
    )
    score_map = dict(zip(test_ids, test_scores, strict=True))
    benchmark = _slice_metrics(test_rows, score_map, unseen)
    checkpoint = output / "best.pt"
    torch.save(
        {
            "format_version": 1,
            "model_id": config.experiment,
            "model_spec": spec.to_dict(),
            "state_dict": best_state,
            "threshold": threshold["threshold"],
            "score_direction": "higher_is_more_synthetic",
            "validation": threshold,
        },
        checkpoint,
    )
    with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "synthetic_score", "decision"])
        for sample_id in test_ids:
            score = score_map[sample_id]
            writer.writerow(
                [
                    sample_id,
                    f"{score:.10f}",
                    "synthetic" if score >= threshold["threshold"] else "real",
                ]
            )
    environment = snapshot(root)
    report = {
        **environment,
        "report_type": "trained_detector_benchmark",
        "model_id": config.experiment,
        "architecture": spec.architecture,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset_version": prep["dataset_version"],
        "split_version": prep["split_version"],
        "manifest_sha256": sha256(manifest),
        "preparation_sha256": sha256(preparation),
        "checkpoint": "best.pt",
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": threshold,
        "selection_rule": "maximum validation ROC-AUC, then minimum validation loss",
        "test_loss": test_loss,
        "test": benchmark,
        "unseen_generators": sorted(unseen),
        "counts": {"train": len(train_rows), "validation": len(val_rows), "test": len(test_rows)},
        "limitations": [
            "Checkpoint and threshold were selected using validation data only.",
            "Test metrics are descriptive research results, not proof of authorship or intent.",
            "Protected unseen metrics are undefined until unseen synthetic and "
            "real references exist.",
        ],
    }
    write_json(output / "history.json", history)
    write_json(output / "benchmark.json", report)
    write_json(
        output / "model.json",
        {
            "model_id": config.experiment,
            "architecture": spec.architecture,
            "checkpoint_sha256": report["checkpoint_sha256"],
            "threshold": threshold["threshold"],
            "sample_rate": spec.sample_rate,
            "max_duration_seconds": spec.max_duration_seconds,
        },
    )
    with (output / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config.model_dump(mode="json"), stream, sort_keys=False)
    return output


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    spec = ModelSpec(**payload["model_spec"])
    model = build_model(spec)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload
