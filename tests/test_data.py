"""Held-out partition invariants and label-order validation."""

import json
from pathlib import Path

import pytest
import torch

from task_arithmetic_clip.checkpoints import load_payload
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.data import ImagePreprocessor, build_dataset, stratified_partition
from task_arithmetic_clip.workflow import initialize


def test_stratified_partitions() -> None:
    """Check determinism, category coverage, and separation of held-out examples.

    Args:
        None.
    Returns:
        None.
    """
    labels = [0] * 10 + [1] * 10 + [2] * 10
    partitions = stratified_partition(labels, (0.7, 0.1, 0.2), 3)
    assert partitions == stratified_partition(labels, (0.7, 0.1, 0.2), 3)
    assert partitions != stratified_partition(labels, (0.7, 0.1, 0.2), 4)
    assert sorted(sum(partitions, [])) == list(range(len(labels)))
    for split in partitions:
        assert {labels[index] for index in split} == {0, 1, 2}
    assert not set(partitions[0]) & set(partitions[1])
    assert not set(partitions[0]) & set(partitions[2])


def test_manifest_preprocessing_and_changes(experiment: ExperimentConfig) -> None:
    """Check grayscale preprocessing and rejection of changed dataset assignments.

    Args:
        experiment: Local CPU experiment configuration.
    Returns:
        None.
    """
    base = load_payload(initialize(experiment), "backbone")
    bundle = build_dataset("ToyA", experiment, base["processor_config"])
    assert [len(bundle.train), len(bundle.val), len(bundle.test)] == [12, 4, 4]
    pixels, label = bundle.train[0]
    assert pixels.shape == (3, 16, 16) and label == 0
    from PIL import Image

    grayscale = ImagePreprocessor(base["processor_config"])(Image.new("L", (28, 28), 100))
    assert grayscale.shape == (3, 16, 16) and torch.isfinite(grayscale).all()
    path = Path(experiment.data_root) / "ToyA" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["train"][0]["label"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partitions changed"):
        build_dataset("ToyA", experiment, base["processor_config"])


def test_reject_partition_overlap(experiment: ExperimentConfig) -> None:
    """Reject explicit manifests that reuse an image across training and test splits.

    Args:
        experiment: Local CPU experiment configuration.
    Returns:
        None.
    """
    base = load_payload(initialize(experiment), "backbone")
    path = Path(experiment.data_root) / "ToyA" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["test"][0] = manifest["train"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="overlapping"):
        build_dataset("ToyA", experiment, base["processor_config"])
