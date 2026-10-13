"""Offline integration of CLIP preparation, fine-tuning, editing, and evaluation."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import CLIPModel

from task_arithmetic_clip.arithmetic import TaskVector
from task_arithmetic_clip.backbone import ClassificationHead
from task_arithmetic_clip.checkpoints import load_backbone, load_payload, save_payload
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.evaluation import evaluate_checkpoints, load_task
from task_arithmetic_clip.training import InfiniteBatchSampler, train_task
from task_arithmetic_clip.workflow import (
    initialize,
    prepare_tasks,
    select_coefficient,
    sweep_coefficients,
)


def test_projection_matches_full_clip(experiment: ExperimentConfig) -> None:
    """Check that vision-only extraction preserves the parent's projection dimension and outputs.

    Args:
        experiment: Local CPU experiment configuration.
    Returns:
        None.
    """
    base_path = initialize(experiment)
    model, _ = load_backbone(base_path)
    original = CLIPModel.from_pretrained(experiment.model_id)
    model.eval()
    original.eval()
    pixels = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        expected = original.get_image_features(pixel_values=pixels)
        actual = model(pixels)
    assert model.feature_dim == 12
    torch.testing.assert_close(actual, expected)


def test_frozen_classifier() -> None:
    """Ensure classifiers propagate image gradients without adding trainable head parameters.

    Args:
        None.
    Returns:
        None.
    """
    head = ClassificationHead(torch.eye(3), 10.0)
    features = torch.randn(2, 3, requires_grad=True)
    head(features).sum().backward()
    assert features.grad is not None
    assert list(head.parameters()) == []
    assert head.weight.grad is None


def test_sampler_resume() -> None:
    """Verify that a restored microbatch cursor produces the same sample sequence.

    Args:
        None.
    Returns:
        None.
    """
    iterator = iter(InfiniteBatchSampler(11, 3, 7))
    expected = [next(iterator) for _ in range(10)]
    resumed = iter(InfiniteBatchSampler(11, 3, 7, start_batch=5))
    assert [next(resumed) for _ in range(5)] == expected[5:]


def test_complete_classification_workflow(experiment: ExperimentConfig, tmp_path: Path) -> None:
    """Exercise local head creation, independent training, vector merging, and held-out evaluation.

    Args:
        experiment: Local CPU experiment configuration.
        tmp_path: Temporary pytest directory.
    Returns:
        None.
    """
    base_path = initialize(experiment)
    prepare_tasks(["ToyA", "ToyB"], experiment)
    model, base = load_backbone(base_path)
    bundle, head_before = load_task("ToyA", experiment, base)
    del model, bundle
    first_path = train_task("ToyA", experiment)
    second_path = train_task("ToyB", experiment)
    _, head_after = load_task("ToyA", experiment, base)
    torch.testing.assert_close(head_before.weight, head_after.weight, rtol=0, atol=0)
    first = TaskVector.from_checkpoints(base_path, first_path)
    second = TaskVector.from_checkpoints(base_path, second_path)
    assert first.norm() > 0 and second.norm() > 0
    assert first.delta["encoder.visual_projection.weight"].norm() > 0
    vector = first + second
    merged_path = tmp_path / "merged.pt"
    save_payload(merged_path, vector.apply_to(base_path, scale=0.5))
    report = evaluate_checkpoints(
        [base_path, first_path, merged_path],
        ["ToyA", "ToyB"],
        experiment,
        "test",
        tmp_path / "metrics.json",
        {"ToyA": first_path, "ToyB": second_path},
    )
    assert report["split"] == "test" and len(report["results"]) == 3
    assert (tmp_path / "metrics.csv").exists()
    sweep = sweep_coefficients(
        vector,
        ["ToyA", "ToyB"],
        [0.0, 0.5, 1.0],
        experiment,
        tmp_path / "selected.pt",
        tmp_path / "sweep.json",
        {"ToyA": first_path, "ToyB": second_path},
    )
    assert sweep["split"] == "val" and len(sweep["candidates"]) == 3
    assert sweep["selected_lambda"] in {0.0, 0.5, 1.0}
    assert (tmp_path / "selected.pt").exists()


def test_training_resume(experiment: ExperimentConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare uninterrupted training with a run interrupted after a resumable checkpoint.

    Args:
        experiment: Local CPU experiment configuration.
        monkeypatch: Pytest patching fixture.
    Returns:
        None.
    """
    from task_arithmetic_clip import training

    initialize(experiment)
    prepare_tasks(["ToyA"], experiment)
    continuous = load_payload(train_task("ToyA", experiment), "backbone")
    resumed_config = replace(experiment, output_root=experiment.output_root + "-resumed")
    initialize(resumed_config)
    prepare_tasks(["ToyA"], resumed_config)
    original_save = training.save_payload

    def interrupt_after_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
        """Simulate an interruption immediately after the second optimizer update is saved.

        Args:
            path: Checkpoint destination.
            payload: Saved checkpoint payload.
        Returns:
            None.
        """
        original_save(path, payload)
        if Path(path).name == "last.pt" and payload["training"]["step"] == 2:
            raise RuntimeError("Simulated interruption.")

    monkeypatch.setattr(training, "save_payload", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="Simulated"):
        train_task("ToyA", resumed_config)
    monkeypatch.setattr(training, "save_payload", original_save)
    resumed_path = Path(resumed_config.output_root) / "ToyA" / "last.pt"
    resumed = load_payload(train_task("ToyA", resumed_config, resumed_path), "backbone")
    torch.testing.assert_close(continuous["state_dict"], resumed["state_dict"], rtol=0, atol=0)
    last = load_payload(resumed_path, "backbone")
    assert last["training"]["step"] == 4 and last["training"]["micro_batches"] == 8


def test_coefficient_selection() -> None:
    """Check accuracy maximization and the paper's constrained forgetting selection rule.

    Args:
        None.
    Returns:
        None.
    """
    rows = [
        {"lambda": 0.0, "score": 0.5, "control_accuracy": 0.8},
        {"lambda": 0.5, "score": 0.9, "control_accuracy": 0.77},
        {"lambda": 1.0, "score": 0.8, "control_accuracy": 0.7},
    ]
    assert select_coefficient(rows, "maximize")["lambda"] == 0.5
    assert select_coefficient(rows, "forget", 0.95 * 0.8)["lambda"] == 0.5
    with pytest.raises(ValueError, match="No coefficient"):
        select_coefficient(rows, "forget", 1.0)
