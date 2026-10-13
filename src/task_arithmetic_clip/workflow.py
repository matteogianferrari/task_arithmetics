"""Model initialization, classifier preparation, and validation-only coefficient search."""

import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import (
    CLIPConfig,
    CLIPImageProcessor,
    CLIPModel,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from task_arithmetic_clip.arithmetic import TaskVector
from task_arithmetic_clip.backbone import CLIPViT
from task_arithmetic_clip.checkpoints import (
    FORMAT_VERSION,
    cpu_state,
    load_backbone,
    load_payload,
    save_payload,
    state_identity,
    write_json,
)
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.data import build_dataset
from task_arithmetic_clip.evaluation import evaluate_dataset, load_task
from task_arithmetic_clip.heads import (
    DEFAULT_TEMPLATES,
    TEMPLATES,
    build_head,
    load_head,
    save_head,
)
from task_arithmetic_clip.runtime import resolve_device


def initialize(config: ExperimentConfig) -> Path:
    """Save a common pretrained vision initialization and its preprocessing metadata.

    Args:
        config: Pretrained model source and experiment directory settings.
    Returns:
        The base backbone checkpoint path.
    """
    path = Path(config.output_root) / "base.pt"
    if path.exists():
        base = load_payload(path, "backbone")
        if (
            base["model_id"] != config.model_id
            or base["metadata"]["requested_revision"] != config.revision
            or state_identity(base["state_dict"], base["vision_config"]) != base["base_id"]
        ):
            raise ValueError(
                "The existing base differs from the requested model. Use a new output_root."
            )
        return path
    pretrained = CLIPModel.from_pretrained(
        config.model_id, revision=config.revision, torch_dtype=torch.float32
    ).cpu()
    backbone = CLIPViT.from_clip(pretrained)
    processor = CLIPImageProcessor.from_pretrained(config.model_id, revision=config.revision)
    revision = getattr(pretrained.config, "_commit_hash", None) or config.revision
    scale = float(pretrained.logit_scale.detach().exp().item())
    del pretrained
    state = cpu_state(backbone)
    vision_config = backbone.encoder.config.to_dict()
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": "backbone",
        "base_id": state_identity(state, vision_config),
        "state_dict": state,
        "vision_config": vision_config,
        "processor_config": json.loads(processor.to_json_string()),
        "model_id": config.model_id,
        "revision": revision,
        "logit_scale": scale,
        "metadata": {"operation": "pretrained", "requested_revision": config.revision},
    }
    save_payload(path, payload)
    write_json(Path(config.output_root) / "experiment.json", config.to_dict())
    return path


def prepare_tasks(
    tasks: list[str],
    config: ExperimentConfig,
    custom_templates: dict[str, list[str]] | None = None,
) -> list[Path]:
    """Prepare datasets and frozen text-prototype heads for classification tasks.

    Args:
        tasks: Paper tasks, ImageNet control, or explicitly supplied custom image tasks.
        config: Pretrained source, dataset, and experiment settings.
        custom_templates: Optional per-task prompt-template lists.
    Returns:
        Classification head checkpoint paths in the requested task order.
    """
    device = resolve_device(config.device, config.precision)
    base_path = initialize(config)
    base = load_payload(base_path, "backbone")
    del base["state_dict"]
    if base["model_id"] != config.model_id:
        raise ValueError("The configuration model_id differs from the saved base.")
    text_encoder = None
    tokenizer = None
    paths = []
    for task in tasks:
        bundle = build_dataset(task, config, base["processor_config"])
        templates = tuple(
            (custom_templates or {}).get(task, TEMPLATES.get(task, DEFAULT_TEMPLATES))
        )
        if not templates:
            raise ValueError(f"No prompt templates were supplied for {task}.")
        path = Path(config.output_root) / "heads" / f"{task}.pt"
        if path.exists():
            _, saved = load_head(path, base["base_id"], bundle.class_names, bundle.split_id)
            if saved["templates"] != list(templates):
                raise ValueError(f"{task} already uses different prompts. Use a new output_root.")
        else:
            if text_encoder is None:
                parent_config = CLIPConfig.from_pretrained(
                    base["model_id"], revision=base["revision"]
                )
                text_config = parent_config.text_config.to_dict()
                text_config["projection_dim"] = parent_config.projection_dim
                text_encoder = CLIPTextModelWithProjection.from_pretrained(
                    base["model_id"],
                    revision=base["revision"],
                    config=CLIPTextConfig.from_dict(text_config),
                    torch_dtype=torch.float32,
                )
                tokenizer = CLIPTokenizer.from_pretrained(
                    base["model_id"], revision=base["revision"]
                )
            head = build_head(
                task,
                bundle.class_names,
                text_encoder,
                tokenizer,
                base["logit_scale"],
                device,
                templates,
            )
            save_head(
                path, head, task, bundle.class_names, base["base_id"], bundle.split_id, templates
            )
            del head
        paths.append(path)
        del bundle
    return paths


def combine_vectors(paths: list[str | Path], coefficients: list[float] | None = None) -> TaskVector:
    """Combine vectors with optional independent coefficients.

    Args:
        paths: Saved task vector paths.
        coefficients: One finite coefficient per vector, or None for unit coefficients.
    Returns:
        The weighted sum of the compatible vectors.
    """
    coefficients = coefficients if coefficients is not None else [1.0] * len(paths)
    if not paths or len(paths) != len(coefficients):
        raise ValueError("Supply at least one vector and exactly one coefficient per vector.")
    combined = None
    for path, coefficient in zip(paths, coefficients):
        vector = TaskVector.load(path) * coefficient
        combined = vector if combined is None else combined + vector
    return combined


def select_coefficient(
    rows: list[dict[str, Any]], selection: str, minimum_control_accuracy: float | None = None
) -> dict[str, Any]:
    """Select a coefficient using validation metrics only.

    Args:
        rows: Candidate coefficients and their validation scores.
        selection: Maximize mean accuracy, or forget while retaining control accuracy.
        minimum_control_accuracy: Required control accuracy for forgetting.
    Returns:
        The selected coefficient's validation record.
    """
    if not rows:
        raise ValueError("The coefficient search has no candidates.")
    if selection == "maximize":
        return min(rows, key=lambda row: (-row["score"], row["lambda"]))
    if selection != "forget" or minimum_control_accuracy is None:
        raise ValueError("Forgetting requires a control-accuracy constraint.")
    feasible = [row for row in rows if row["control_accuracy"] >= minimum_control_accuracy]
    if not feasible:
        raise ValueError("No coefficient satisfies the control constraint. Include lambda zero.")
    return max(feasible, key=lambda row: row["lambda"])


def sweep_coefficients(
    vector: TaskVector,
    tasks: list[str],
    lambdas: list[float],
    config: ExperimentConfig,
    output_path: str | Path,
    report_path: str | Path,
    references: dict[str, str | Path] | None = None,
    selection: str = "maximize",
    control: str | None = None,
    retain: float = 0.95,
) -> dict[str, Any]:
    """Search a global task-vector coefficient on held-out validation examples.

    For forgetting, supply a negated vector and choose the largest coefficient
    that preserves the requested fraction of the pretrained control accuracy.

    Args:
        vector: Combined vector, with any task-specific signs and coefficients.
        tasks: Target classification tasks.
        lambdas: Nonnegative global coefficient candidates.
        config: Dataset, precision, and evaluation settings.
        output_path: Destination for the selected edited backbone.
        report_path: Destination for all validation candidate metrics.
        references: Optional specialized checkpoints for normalized accuracy selection.
        selection: Maximize accuracy or select forgetting subject to control retention.
        control: Optional control dataset, required for forgetting.
        retain: Fraction of pretrained control accuracy to preserve.
    Returns:
        The complete validation search report, including the selected coefficient.
    """
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Supply unique target task names.")
    if not lambdas or any(not math.isfinite(value) or value < 0 for value in lambdas):
        raise ValueError("Global coefficient candidates must be finite and nonnegative.")
    if not 0 < retain <= 1 or selection not in {"maximize", "forget"}:
        raise ValueError("Invalid selection rule or control retention fraction.")
    if selection == "forget" and (not control or control in tasks):
        raise ValueError("Forgetting requires a separate --control dataset.")
    if references is not None and set(references) != set(tasks):
        raise ValueError("Provide exactly one specialized reference for each target task.")
    device = resolve_device(config.device, config.precision)
    base_path = Path(config.output_root) / "base.pt"
    if Path(output_path).resolve() == base_path.resolve():
        raise ValueError("The edited output must differ from the common pretrained base.")
    base = vector.apply_to(base_path, scale=0.0)
    all_tasks = tasks + ([control] if control and control not in tasks else [])
    bundles, heads = {}, {}
    for task in all_tasks:
        bundles[task], heads[task] = load_task(task, config, base)
    reference_accuracy = {}
    for task, path in (references or {}).items():
        reference, payload = load_backbone(path)
        metadata = payload["metadata"]
        if (
            payload["base_id"] != base["base_id"]
            or metadata.get("task") != task
            or metadata.get("split_id") != bundles[task].split_id
        ):
            raise ValueError(f"Reference {path} is incompatible with {task}.")
        accuracy = evaluate_dataset(reference, heads[task], bundles[task].val, config, device)[
            "accuracy"
        ]
        if accuracy <= 0:
            raise ValueError(f"Cannot normalize by zero validation accuracy on {task}.")
        reference_accuracy[task] = accuracy
        del reference
    model = CLIPViT.from_config(base["vision_config"]).to(device)
    model.load_state_dict(base["state_dict"], strict=True)
    baseline_control = None
    if control:
        baseline_control = evaluate_dataset(
            model, heads[control], bundles[control].val, config, device
        )["accuracy"]
        if baseline_control <= 0:
            raise ValueError("The pretrained control accuracy must be positive.")
    rows = []
    for coefficient in sorted(set(lambdas)):
        edited_state = {
            name: value + coefficient * vector.delta[name] if value.is_floating_point() else value
            for name, value in base["state_dict"].items()
        }
        if any(
            not torch.isfinite(value).all()
            for value in edited_state.values()
            if value.is_floating_point()
        ):
            raise FloatingPointError("A candidate coefficient produced non-finite weights.")
        model.load_state_dict(edited_state, strict=True)
        del edited_state
        metrics = {
            task: evaluate_dataset(model, heads[task], bundles[task].val, config, device)
            for task in all_tasks
        }
        mean_accuracy = sum(metrics[task]["accuracy"] for task in tasks) / len(tasks)
        score = mean_accuracy
        row = {"lambda": coefficient, "tasks": metrics, "mean_accuracy": mean_accuracy}
        if references:
            score = sum(
                metrics[task]["accuracy"] / reference_accuracy[task] for task in tasks
            ) / len(tasks)
            row["mean_normalized_accuracy"] = score
        row["score"] = score
        if control:
            row["control_accuracy"] = metrics[control]["accuracy"]
            row["control_retention"] = row["control_accuracy"] / baseline_control
        rows.append(row)
        print(f"lambda={coefficient:.3f} validation_score={score:.5f}", flush=True)
    chosen = select_coefficient(rows, selection, retain * baseline_control if control else None)
    edited = vector.apply_to(base_path, chosen["lambda"])
    edited["metadata"]["validation_selection"] = {
        "tasks": tasks,
        "control": control,
        "selection": selection,
        "split_ids": {task: bundles[task].split_id for task in all_tasks},
    }
    save_payload(output_path, edited)
    report = {
        "split": "val",
        "selection": selection,
        "base_id": base["base_id"],
        "reference_accuracy": reference_accuracy,
        "baseline_control_accuracy": baseline_control,
        "retain": retain if control else None,
        "selected_lambda": chosen["lambda"],
        "selected_checkpoint": str(output_path),
        "candidates": rows,
    }
    write_json(report_path, report)
    return report
