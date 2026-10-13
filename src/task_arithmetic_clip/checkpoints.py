"""Portable tensor-only checkpoints with architecture and base-weight identities."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from task_arithmetic_clip.backbone import CLIPViT

FORMAT_VERSION = 1


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy module parameters and persistent buffers to CPU memory.

    Args:
        model: Module whose state should be serialized.
    Returns:
        An independent CPU state dictionary, with floating tensors in float32.
    """
    return {
        name: value.detach().cpu().float().clone()
        if value.is_floating_point()
        else value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_identity(state: dict[str, torch.Tensor], config: dict[str, Any]) -> str:
    """Hash the exact parameter values and serialized architecture.

    Args:
        state: CPU backbone parameters and buffers.
        config: Serialized architecture configuration.
    Returns:
        A SHA-256 identity for the architecture and all state values.
    """
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_schema(state: dict[str, torch.Tensor], config: dict[str, Any]) -> str:
    """Hash the architecture and parameter schema without parameter values.

    Args:
        state: Backbone parameters and buffers.
        config: Serialized architecture configuration.
    Returns:
        A SHA-256 identity for parameter names, shapes, dtypes, and configuration.
    """
    entries = [[name, list(value.shape), str(value.dtype)] for name, value in sorted(state.items())]
    return hashlib.sha256(json.dumps([config, entries], sort_keys=True).encode()).hexdigest()


def save_payload(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically save a payload containing tensors and ordinary Python values.

    Args:
        path: Destination checkpoint path.
        payload: Checkpoint payload.
    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_payload(path: str | Path, kind: str) -> dict[str, Any]:
    """Load a versioned checkpoint without deserializing Python model objects.

    Args:
        path: Source checkpoint path.
        kind: Expected payload kind.
    Returns:
        A checkpoint dictionary loaded into CPU memory.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a checkpoint dictionary.")
    if payload.get("format_version") != FORMAT_VERSION or payload.get("kind") != kind:
        raise ValueError(f"{path} is not a supported {kind} checkpoint.")
    return payload


def snapshot(model: CLIPViT, template: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    """Create a backbone checkpoint using the original base metadata.

    Args:
        model: Vision encoder whose weights should be saved.
        template: Original base checkpoint, including configuration and preprocessing.
        metadata: Experiment provenance for the new checkpoint.
    Returns:
        A portable backbone checkpoint without optimizer state.
    """
    return {
        key: template[key]
        for key in (
            "format_version",
            "kind",
            "base_id",
            "vision_config",
            "processor_config",
            "model_id",
            "revision",
            "logit_scale",
        )
    } | {"state_dict": cpu_state(model), "metadata": metadata}


def load_backbone(path: str | Path) -> tuple[CLIPViT, dict[str, Any]]:
    """Reconstruct a vision encoder entirely from a local checkpoint.

    Args:
        path: Path to a backbone checkpoint produced by this project.
    Returns:
        The reconstructed backbone and its checkpoint metadata.
    """
    payload = load_payload(path, "backbone")
    model = CLIPViT.from_config(payload["vision_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write readable JSON data.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible values.
    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
