"""Validated configuration for classification and task arithmetic experiments."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """ExperimentConfig class for storing model, dataset, and training settings.

    Attributes:
        model_id: Hugging Face model identifier or a local pretrained model directory.
        revision: Optional Hugging Face model revision.
        data_root: Directory containing the classification datasets.
        output_root: Directory containing checkpoints, heads, splits, and metrics.
        seed: Random seed for model training.
        split_seed: Independent random seed for dataset partitions.
        validation_fraction: Fraction reserved from training when validation is unavailable.
        test_fraction: Test fraction for datasets without an official test partition.
        download: Whether supported torchvision datasets may be downloaded.
        batch_size: Number of images per training microbatch.
        accumulation_steps: Number of microbatches per optimizer update.
        num_workers: Number of dataset loading workers.
        steps: Number of successful optimizer updates.
        warmup_steps: Number of linear learning rate warm-up updates.
        learning_rate: Peak AdamW learning rate.
        weight_decay: AdamW weight decay.
        grad_clip: Maximum gradient norm.
        precision: Computation precision, one of fp32, fp16, or bf16.
        device: PyTorch device, such as cuda, cuda:0, or cpu.
        gradient_checkpointing: Whether to recompute vision activations during backward.
        augment: Whether to use random crops and horizontal flips during training.
        eval_every: Interval between validation evaluations, in optimizer updates.
        save_every: Interval between resumable checkpoint saves, in optimizer updates.
        sun_split: Official SUN397 partition number.
        dtd_split: Official DTD partition number.
    """

    model_id: str = "openai/clip-vit-base-patch32"
    revision: str | None = None
    data_root: str = "./data"
    output_root: str = "./runs/vit-b32"
    seed: int = 0
    split_seed: int = 0
    validation_fraction: float = 0.1
    test_fraction: float = 0.2
    download: bool = True
    batch_size: int = 32
    accumulation_steps: int = 4
    num_workers: int = 4
    steps: int = 2000
    warmup_steps: int = 200
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "fp32"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    augment: bool = False
    eval_every: int = 200
    save_every: int = 200
    sun_split: int = 1
    dtd_split: int = 1

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the experiment configuration.
        Returns:
            An ExperimentConfig instance with validated settings.
        """
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError("The configuration must be a YAML mapping.")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        """Validate ranges and choices used by the experiment.

        Args:
            None.
        Returns:
            None.
        """
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if not 0 < self.test_fraction < 1 - self.validation_fraction:
            raise ValueError("test_fraction must leave room for training and validation.")
        for name in ("batch_size", "accumulation_steps", "steps", "eval_every", "save_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.num_workers < 0 or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("Invalid num_workers or warmup_steps.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid learning rate, weight decay, or gradient clipping.")
        if not 1 <= self.sun_split <= 10 or not 1 <= self.dtd_split <= 10:
            raise ValueError("SUN397 and DTD partition numbers must be in [1, 10].")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration into ordinary Python values.

        Args:
            None.
        Returns:
            A dictionary containing all experiment settings.
        """
        return asdict(self)
