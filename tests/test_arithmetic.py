"""Task-vector invariants, exact-base checks, and arithmetic edge cases."""

from pathlib import Path

import pytest
import torch

from task_arithmetic_clip.arithmetic import TaskVector
from task_arithmetic_clip.checkpoints import load_payload, save_payload
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.workflow import initialize


def test_vector_operators() -> None:
    """Verify weighted addition, negation, summation, and analogy identities.

    Args:
        None.
    Returns:
        None.
    """
    first = TaskVector({"weight": torch.tensor([1.0, 2.0])}, "base", "schema")
    second = TaskVector({"weight": torch.tensor([3.0, 5.0])}, "base", "schema")
    torch.testing.assert_close((2 * first - second).delta["weight"], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(sum([first, second]).delta["weight"], torch.tensor([4.0, 7.0]))
    torch.testing.assert_close((-first + first).delta["weight"], torch.zeros(2))
    torch.testing.assert_close((second + first - first).delta["weight"], second.delta["weight"])


def test_incompatible_vectors() -> None:
    """Reject mismatched bases, architectures, shapes, and non-finite coefficients.

    Args:
        None.
    Returns:
        None.
    """
    vector = TaskVector({"weight": torch.ones(2)}, "base", "schema")
    for other in (
        TaskVector({"weight": torch.ones(2)}, "other", "schema"),
        TaskVector({"weight": torch.ones(2)}, "base", "other"),
        TaskVector({"weight": torch.ones(3)}, "base", "schema"),
        TaskVector({"different": torch.ones(2)}, "base", "schema"),
    ):
        with pytest.raises(ValueError):
            _ = vector + other
    with pytest.raises(ValueError):
        _ = vector * float("nan")
    with pytest.raises(ValueError):
        TaskVector({"weight": torch.tensor([float("inf")])}, "base", "schema")


def test_checkpoint_arithmetic(experiment: ExperimentConfig, tmp_path: Path) -> None:
    """Recover fine-tuned weights at scale one and preserve non-floating buffers.

    Args:
        experiment: Local CPU experiment configuration.
        tmp_path: Temporary pytest directory.
    Returns:
        None.
    """
    base_path = initialize(experiment)
    base = load_payload(base_path, "backbone")
    base["state_dict"]["test_integer_buffer"] = torch.tensor([1, 4], dtype=torch.int64)
    from task_arithmetic_clip.checkpoints import state_identity

    base["base_id"] = state_identity(base["state_dict"], base["vision_config"])
    save_payload(base_path, base)
    tuned = {key: value for key, value in base.items() if key != "state_dict"}
    tuned["state_dict"] = {
        name: value + 0.125 if value.is_floating_point() else value.clone()
        for name, value in base["state_dict"].items()
    }
    tuned_path = tmp_path / "tuned.pt"
    save_payload(tuned_path, tuned)
    vector = TaskVector.from_checkpoints(base_path, tuned_path)
    restored = vector.apply_to(base_path, scale=1.0)
    for name, value in tuned["state_dict"].items():
        torch.testing.assert_close(restored["state_dict"][name], value)
    unchanged = vector.apply_to(base_path, scale=0.0)
    for name, value in base["state_dict"].items():
        torch.testing.assert_close(unchanged["state_dict"][name], value)
    vector_path = tmp_path / "vector.pt"
    vector.save(vector_path)
    torch.testing.assert_close(TaskVector.load(vector_path).delta, vector.delta)
    with pytest.raises(ValueError, match="original"):
        TaskVector.from_checkpoints(tuned_path, tuned_path)
    with pytest.raises(ValueError, match="pretrained base"):
        vector.apply_to(tuned_path)
    tuned["state_dict"]["test_integer_buffer"] += 1
    save_payload(tuned_path, tuned)
    with pytest.raises(ValueError, match="Non-floating"):
        TaskVector.from_checkpoints(base_path, tuned_path)
