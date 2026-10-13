"""Classification evaluation with raw and fine-tuned-normalized accuracy."""

import csv
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from task_arithmetic_clip.backbone import ClassificationHead, CLIPViT
from task_arithmetic_clip.checkpoints import load_backbone, write_json
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.data import DatasetBundle, build_dataset
from task_arithmetic_clip.heads import load_head
from task_arithmetic_clip.runtime import autocast_context, resolve_device


@torch.no_grad()
def evaluate_dataset(
    model: CLIPViT,
    head: ClassificationHead,
    dataset: Dataset,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a vision encoder with one task's frozen classifier.

    Args:
        model: Vision encoder to evaluate.
        head: Classification head corresponding to this task.
        dataset: Validation or test dataset.
        config: Batch size, worker, and precision settings.
        device: Evaluation device.
    Returns:
        Accuracy, average cross-entropy loss, and sample count.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    model.eval().to(device)
    head.eval().to(device)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    correct, total, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, config.precision):
            logits = head(model(images))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Evaluation produced non-finite logits.")
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.numel()
    return {"accuracy": correct / total, "loss": loss_sum / total, "samples": total}


def load_task(
    task: str, config: ExperimentConfig, payload: dict[str, Any]
) -> tuple[DatasetBundle, ClassificationHead]:
    """Load a task's saved split and classifier metadata.

    Args:
        task: Classification task name.
        config: Experiment configuration.
        payload: Backbone checkpoint metadata.
    Returns:
        Dataset partitions and a compatible frozen classification head.
    """
    bundle = build_dataset(task, config, payload["processor_config"])
    head, _ = load_head(
        Path(config.output_root) / "heads" / f"{task}.pt",
        payload["base_id"],
        bundle.class_names,
        bundle.split_id,
    )
    return bundle, head


def evaluate_checkpoints(
    checkpoints: list[str | Path],
    tasks: list[str],
    config: ExperimentConfig,
    split: str,
    destination: str | Path,
    references: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Evaluate checkpoints on common held-out classification partitions.

    Args:
        checkpoints: Backbone checkpoint paths to compare.
        tasks: Classification tasks to evaluate.
        config: Experiment settings used to reconstruct datasets and heads.
        split: Validation or test partition.
        destination: JSON destination, with a CSV companion using the same stem.
        references: Optional mapping from every task to its specialized fine-tuned model.
    Returns:
        A report containing per-task metrics and equal-task-weight aggregate metrics.
    """
    if split not in {"val", "test"} or not checkpoints or not tasks:
        raise ValueError("Evaluation requires checkpoints, tasks, and a val or test split.")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Evaluation task names must be unique.")
    if references is not None and set(references) != set(tasks):
        raise ValueError("Provide exactly one specialized reference for every evaluated task.")
    device = resolve_device(config.device, config.precision)
    reference_accuracy = {}
    reference_splits = {}
    for task, path in (references or {}).items():
        model, payload = load_backbone(path)
        if payload.get("metadata", {}).get("task") != task:
            raise ValueError(f"{path} is not a specialized fine-tuned checkpoint for {task}.")
        bundle, head = load_task(task, config, payload)
        reference_accuracy[task] = evaluate_dataset(
            model, head, getattr(bundle, split), config, device
        )["accuracy"]
        reference_splits[task] = (payload["base_id"], bundle.split_id)
        if reference_accuracy[task] <= 0:
            raise ValueError(f"Cannot normalize by zero accuracy on {task}.")
        del model, head, bundle
    results = []
    base_id = None
    for path in checkpoints:
        model, payload = load_backbone(path)
        if base_id is not None and base_id != payload["base_id"]:
            raise ValueError("Compared checkpoints must share the same pretrained initialization.")
        base_id = payload["base_id"]
        per_task = {}
        for task in tasks:
            bundle, head = load_task(task, config, payload)
            metrics = evaluate_dataset(model, head, getattr(bundle, split), config, device)
            metrics["split_id"] = bundle.split_id
            if references:
                if reference_splits[task] != (base_id, bundle.split_id):
                    raise ValueError("A reference uses a different base or dataset partition.")
                metrics["normalized_accuracy"] = metrics["accuracy"] / reference_accuracy[task]
            per_task[task] = metrics
            del bundle, head
        result = {
            "checkpoint": str(path),
            "tasks": per_task,
            "mean_accuracy": sum(row["accuracy"] for row in per_task.values()) / len(tasks),
        }
        if references:
            result["mean_normalized_accuracy"] = sum(
                row["normalized_accuracy"] for row in per_task.values()
            ) / len(tasks)
        results.append(result)
        del model
    report = {
        "split": split,
        "base_id": base_id,
        "reference_accuracy": reference_accuracy,
        "results": results,
    }
    destination = Path(destination)
    write_json(destination, report)
    fields = ["checkpoint", "task", "split", "accuracy", "loss", "samples", "normalized_accuracy"]
    with destination.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for task, metrics in result["tasks"].items():
                writer.writerow(
                    {
                        "checkpoint": result["checkpoint"],
                        "task": task,
                        "split": split,
                        **{key: metrics.get(key, "") for key in fields[3:]},
                    }
                )
    return report
