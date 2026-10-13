"""Frozen classification prototypes produced once by the pretrained CLIP text encoder."""

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from task_arithmetic_clip.backbone import ClassificationHead
from task_arithmetic_clip.checkpoints import (
    FORMAT_VERSION,
    load_payload,
    save_payload,
    state_identity,
)

TEMPLATES = {
    "Cars": ("a photo of a {}.", "a photo of the {}.", "a picture of a {}."),
    "DTD": ("a photo of a {} texture.", "a close-up of a {} surface.", "{} texture."),
    "EuroSAT": ("a satellite photo of {}.", "an aerial image of {}.", "{} from above."),
    "RESISC45": ("a satellite photo of {}.", "an aerial image of {}.", "{} from above."),
    "GTSRB": ("a photo of a {} traffic sign.", "a {} traffic sign.", "a road sign for {}."),
    "MNIST": ("a handwritten digit {}.", "a photo of the number {}.", "the digit {}."),
    "SVHN": ("a house number {}.", "a photo of the number {}.", "the digit {}."),
    "SUN397": ("a photo of a {}.", "a picture of a {}.", "a scene showing a {}."),
    "ImageNet": ("a photo of a {}.", "a photo of the {}.", "a clear picture of a {}."),
}
DEFAULT_TEMPLATES = ("a photo of a {}.", "a photo of the {}.", "a picture of a {}.")


@torch.no_grad()
def build_head(
    task: str,
    class_names: list[str],
    text_encoder: CLIPTextModelWithProjection,
    tokenizer: CLIPTokenizer,
    logit_scale: float,
    device: torch.device,
    templates: tuple[str, ...] | None = None,
    batch_size: int = 64,
) -> ClassificationHead:
    """Create a frozen prompt ensemble for a classification task.

    Normalize each prompt embedding, average prompts within each class, then
    normalize the resulting class prototype before applying the fixed logit scale.

    Args:
        task: Classification task name used to select prompt templates.
        class_names: Class names in exact integer-label order.
        text_encoder: Pretrained CLIP text encoder and text projection.
        tokenizer: Tokenizer from the same pretrained CLIP source.
        logit_scale: Positive pretrained multiplier for cosine similarities.
        device: Device used for one-time text encoding.
        templates: Optional replacement prompt templates with one positional placeholder.
        batch_size: Number of prompts encoded at once.
    Returns:
        A frozen classification head containing CPU float32 prototypes.
    """
    templates = templates or TEMPLATES.get(task, DEFAULT_TEMPLATES)
    if batch_size < 1:
        raise ValueError("Prompt batch size must be positive.")
    for template in templates:
        if template.count("{}") != 1:
            raise ValueError("Each prompt template must contain exactly one {} placeholder.")
    prompts = [template.format(name) for name in class_names for template in templates]
    text_encoder.eval().to(device)
    embeddings = []
    for start in range(0, len(prompts), batch_size):
        inputs = tokenizer(
            prompts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=text_encoder.config.max_position_embeddings,
            return_tensors="pt",
        ).to(device)
        output = text_encoder(**inputs, return_dict=True).text_embeds.float()
        embeddings.append(F.normalize(output, dim=-1).cpu())
    prototypes = torch.cat(embeddings).reshape(len(class_names), len(templates), -1).mean(dim=1)
    return ClassificationHead(prototypes, logit_scale)


def save_head(
    path: str | Path,
    head: ClassificationHead,
    task: str,
    class_names: list[str],
    base_id: str,
    split_id: str,
    templates: tuple[str, ...],
) -> None:
    """Save a classifier and its exact dataset label mapping.

    Args:
        path: Destination classification head checkpoint.
        head: Frozen classification head.
        task: Classification task name.
        class_names: Class names in dataset label order.
        base_id: Identity of the shared pretrained vision model.
        split_id: Identity of the dataset partition assignments.
        templates: Prompt templates used to produce the class prototypes.
    Returns:
        None.
    """
    save_payload(
        path,
        {
            "format_version": FORMAT_VERSION,
            "kind": "classification_head",
            "task": task,
            "class_names": class_names,
            "base_id": base_id,
            "split_id": split_id,
            "templates": list(templates),
            "weight": head.weight.detach().cpu(),
            "logit_scale": float(head.logit_scale.item()),
            "head_id": state_identity(head.state_dict(), {"class_names": class_names}),
        },
    )


def load_head(
    path: str | Path,
    base_id: str,
    class_names: list[str],
    split_id: str,
) -> tuple[ClassificationHead, dict[str, Any]]:
    """Load a frozen classifier while verifying model and dataset compatibility.

    Args:
        path: Source classification head checkpoint.
        base_id: Expected pretrained model identity.
        class_names: Expected class names in exact label order.
        split_id: Expected dataset partition identity.
    Returns:
        The frozen classification head and its saved metadata.
    """
    payload = load_payload(path, "classification_head")
    if payload["base_id"] != base_id:
        raise ValueError("The classifier belongs to a different pretrained model.")
    if payload["class_names"] != class_names or payload["split_id"] != split_id:
        raise ValueError("The classifier label mapping or dataset partitions differ.")
    head = ClassificationHead(payload["weight"], payload["logit_scale"])
    # Preserve the exact stored prototypes rather than normalizing them a second time.
    head.load_state_dict(
        {
            "weight": payload["weight"],
            "logit_scale": torch.tensor(payload["logit_scale"], dtype=torch.float32),
        },
        strict=True,
    )
    if payload["head_id"] != state_identity(head.state_dict(), {"class_names": class_names}):
        raise ValueError("The classification head weights changed after preparation.")
    return head, payload
