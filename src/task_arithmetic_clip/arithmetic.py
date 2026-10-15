"""Task vectors for addition, subtraction, negation, scaling, and analogies."""

import math
from pathlib import Path
from typing import Any

import torch

from task_arithmetic_clip.checkpoints import (
    FORMAT_VERSION,
    load_payload,
    save_payload,
    state_identity,
    state_schema,
)


class TaskVector:
    """TaskVector class for changes to the shared vision encoder and projection.

    Floating parameters and buffers participate in arithmetic. Non-floating buffers
    must remain unchanged during fine-tuning and are copied from the base.

    Attributes:
        delta: CPU float32 parameter differences.
        base_id: Identity of the exact shared pretrained initialization.
        schema_id: Identity of the architecture and state schema.
        provenance: Descriptions of the vector's source and arithmetic operations.
    """

    def __init__(
        self,
        delta: dict[str, torch.Tensor],
        base_id: str,
        schema_id: str,
        provenance: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a task vector from parameter differences.

        Args:
            delta: Mapping from floating state names to parameter differences.
            base_id: Identity of the pretrained initialization.
            schema_id: Identity of the architecture and state schema.
            provenance: Optional records describing the vector.

        Returns:
            None.
        """
        # Error check
        if not delta or not base_id or not schema_id:
            raise ValueError("A task vector requires parameters, a base identity, and a schema.")

        # Class attributes
        self.base_id = base_id
        self.schema_id = schema_id
        self.provenance = provenance or []
        self.delta = {}

        # Clones the parameter difference between fine-tuned and pre-trained model
        for name, value in delta.items():
            # Error check
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Task vector entry {name} must be finite and floating-point.")

            # Clones the value to the related key
            self.delta[name] = value.detach().cpu().float().clone()



    @classmethod
    def from_checkpoints(cls, base_path: str | Path, finetuned_path: str | Path) -> "TaskVector":
        """Subtract the shared base from a fine-tuned backbone checkpoint.

        Args:
            base_path: Path to the original pretrained backbone checkpoint.
            finetuned_path: Path to a checkpoint fine-tuned from that exact base.

        Returns:
            A task vector containing all floating vision and projection differences.
        """
        # Loads the models
        base = load_payload(base_path, "backbone")
        tuned = load_payload(finetuned_path, "backbone")

        # Error checks
        if state_identity(base["state_dict"], base["vision_config"]) != base["base_id"]:
            raise ValueError("base_path must contain the original, unedited pretrained weights.")

        schema = state_schema(base["state_dict"], base["vision_config"])
        if base["base_id"] != tuned["base_id"] or schema != state_schema(tuned["state_dict"], tuned["vision_config"]):
            raise ValueError("Checkpoints must share the exact initialization and architecture.")

        # Dict for delta parameters
        delta = {}

        # Difference process between parameters
        for name, value in base["state_dict"].items():
            # Gets the parameter values from the fine-tuned version
            other = tuned["state_dict"][name]

            # Error check
            if value.is_floating_point():
                # Computes the difference
                delta[name] = other.float() - value.float()
            elif not torch.equal(value, other):
                raise ValueError(f"Non-floating buffer {name} changed during fine-tuning.")

        return cls(delta, base["base_id"], schema, [{"finetuned": str(finetuned_path)}])

    def _check_compatible(self, other: "TaskVector") -> None:
        """Verify that two vectors can be combined safely.

        Args:
            other: Task vector to compare with this instance.

        Returns:
            None.
        """
        # Error checks
        if self.base_id != other.base_id or self.schema_id != other.schema_id:
            raise ValueError("Task vectors have different pretrained bases or architectures.")

        if self.delta.keys() != other.delta.keys():
            raise ValueError("Task vectors contain different parameter names.")

        if any(self.delta[name].shape != other.delta[name].shape for name in self.delta):
            raise ValueError("Task vector parameter shapes differ.")

    def __add__(self, other: "TaskVector") -> "TaskVector":
        """Add compatible task vectors element-wise.

        Args:
            other: Task vector to add.

        Returns:
            A new task vector containing the sum.
        """
        # Error check
        if not isinstance(other, TaskVector):
            return NotImplemented

        # Checks if the 2 task vectors are compatible
        self._check_compatible(other)

        return TaskVector(
            {name: value + other.delta[name] for name, value in self.delta.items()},
            self.base_id,
            self.schema_id,
            self.provenance + other.provenance
        )

    def __radd__(self, other: int) -> "TaskVector":
        """Support summation of a sequence of task vectors.

        Args:
            other: Initial zero supplied by Python's sum function.

        Returns:
            This vector for an initial zero.
        """
        if isinstance(other, int) and other == 0:
            return self

        return NotImplemented

    def __neg__(self) -> "TaskVector":
        """Negate every parameter difference.

        Args:
            None.

        Returns:
            A task vector pointing away from the fine-tuned weights.
        """
        return self * -1.0

    def __sub__(self, other: "TaskVector") -> "TaskVector":
        """Subtract a compatible task vector.

        Args:
            other: Task vector to subtract.

        Returns:
            A task vector containing the element-wise difference.
        """
        return self + (-other)

    def __mul__(self, coefficient: float) -> "TaskVector":
        """Scale every parameter difference by a finite coefficient.

        Args:
            coefficient: Multiplicative coefficient.

        Returns:
            A new scaled task vector.
        """
        # Convert because it isn't guaranteed that it is a float
        coefficient = float(coefficient)

        # Error check
        if not math.isfinite(coefficient):
            raise ValueError("Task vector coefficients must be finite.")

        return TaskVector(
            {name: value * coefficient for name, value in self.delta.items()},
            self.base_id,
            self.schema_id,
            self.provenance + [{"scale": coefficient}]
        )

    def __rmul__(self, coefficient: float) -> "TaskVector":
        """Support coefficients placed before a task vector.

        Args:
            coefficient: Multiplicative coefficient.

        Returns:
            A new scaled task vector.
        """
        return self * coefficient

    def apply_to(self, base_path: str | Path, scale: float = 1.0) -> dict[str, Any]:
        """Apply this task vector to its exact pre-trained initialization.

        Args:
            base_path: Path to the original pre-trained backbone.
            scale: Global coefficient applied after combining vectors.

        Returns:
            An edited backbone checkpoint ready to save or load.
        """
        # Convert because it isn't guaranteed that it is a float
        scale = float(scale)

        # Error check
        if not math.isfinite(scale):
            raise ValueError("The global scale must be finite.")

        # Loads the pre-trained model
        base = load_payload(base_path, "backbone")

        # Error check
        state = base["state_dict"]
        if (
            state_identity(state, base["vision_config"]) != self.base_id
            or base["base_id"] != self.base_id
            or state_schema(state, base["vision_config"]) != self.schema_id
        ):
            raise ValueError("The application checkpoint is not this vector's pretrained base.")

        # Error check
        expected = {name for name, value in state.items() if value.is_floating_point()}
        if expected != self.delta.keys():
            raise ValueError("The task vector does not cover all floating backbone entries.")

        edited = {}
        for name, value in state.items():
            if value.is_floating_point():
                # Error check
                if value.shape != self.delta[name].shape:
                    raise ValueError(f"Task vector shape mismatch for {name}.")

                # Applies task vector parameter to the pre-trained parameter
                edited[name] = value.float() + scale * self.delta[name]
                if not torch.isfinite(edited[name]).all():
                    raise ValueError(f"Task arithmetic produced non-finite values for {name}.")
            else:
                edited[name] = value.clone()

        return {
            key: value
            for key, value in base.items()
            if key not in {"state_dict", "metadata", "training"}
        } | {
            "state_dict": edited,
            "metadata": {"operation": "task_arithmetic", "scale": scale, "terms": self.provenance},
        }

    def save(self, path: str | Path) -> None:
        """Save this vector as a portable tensor-only checkpoint.

        Args:
            path: Destination vector checkpoint.

        Returns:
            None.
        """
        save_payload(
            path,
            {
                "format_version": FORMAT_VERSION,
                "kind": "task_vector",
                "delta": self.delta,
                "base_id": self.base_id,
                "schema_id": self.schema_id,
                "provenance": self.provenance
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "TaskVector":
        """Load a previously saved task vector.

        Args:
            path: Source vector checkpoint.

        Returns:
            A TaskVector instance reconstructed in CPU memory.
        """
        payload = load_payload(path, "task_vector")

        return cls(payload["delta"], payload["base_id"], payload["schema_id"], payload["provenance"])

    def norm(self) -> float:
        """Compute the global Euclidean norm without concatenating parameters.

        Args:
            None.

        Returns:
            The square root of the sum of squared parameter differences.
        """
        return math.sqrt(sum(value.double().square().sum().item() for value in self.delta.values()))
