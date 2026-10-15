"""Classification experiments with CLIP vision backbones and task vectors."""

from task_arithmetic_clip.arithmetic import TaskVector
from task_arithmetic_clip.backbone import ClassificationHead, CLIPViT

__all__ = [
    "CLIPViT",
    "ClassificationHead",
    "TaskVector"
]
