"""Device selection, computation precision, and reproducible random states."""

import random
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch


def resolve_device(name: str, precision: str) -> torch.device:
    """Validate and select the experiment device.

    Args:
        name: PyTorch device identifier.
        precision: Computation precision, one of fp32, fp16, or bf16.
    Returns:
        A usable CPU or CUDA device.
    """
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only Linux CPU and CUDA devices are supported.")
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Unsupported computation precision.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver or set device: cpu.")
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bf16. Use fp16 or fp32.")
    elif precision != "fp32":
        raise ValueError("CPU experiments use fp32.")
    return device


def autocast_context(device: torch.device, precision: str) -> AbstractContextManager:
    """Create the selected mixed-precision computation context.

    Args:
        device: Experiment device.
        precision: Requested computation precision.
    Returns:
        An autocast context for mixed precision, or an inactive context for fp32.
    """
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Training seed.
    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng() -> dict[str, Any]:
    """Capture random states using tensors and ordinary Python values.

    Args:
        None.
    Returns:
        Random states compatible with weights_only checkpoint loading.
    """
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(states: dict[str, Any]) -> None:
    """Restore saved random states after reconstructing model and optimizer objects.

    Args:
        states: Random states captured by capture_rng.
    Returns:
        None.
    """
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state(
        (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        )
    )
    torch.set_rng_state(states["torch"])
    if states["cuda"] and torch.cuda.is_available():
        if len(states["cuda"]) != torch.cuda.device_count():
            raise ValueError("The CUDA device count differs from the resumable checkpoint.")
        torch.cuda.set_rng_state_all(states["cuda"])


def to_cpu(value: Any) -> Any:
    """Recursively copy optimizer and scaler values to CPU memory.

    Args:
        value: Nested dictionaries, lists, tuples, tensors, or scalar values.
    Returns:
        A structure with the same layout and independent CPU tensors.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value
