"""Hugging Face CLIP ViT image encoder with its visual projection."""

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel, CLIPVisionConfig, CLIPVisionModelWithProjection


class CLIPViT(nn.Module):
    """CLIPViT class for extracting projected CLIP image features without carrying the text encoder.

    Both the ViT and visual projection are fine-tuned and included in task vectors.

    Attributes:
        encoder: Hugging Face CLIP vision model with a visual projection layer.
    """

    def __init__(self, config: CLIPVisionConfig) -> None:
        """Initialize the CLIPViT instance.

        Args:
            config: Hugging Face vision configuration, including projection_dim.

        Returns:
            None.
        """
        super().__init__()

        self.encoder = CLIPVisionModelWithProjection(config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CLIPViT":
        """Construct the backbone from checkpoint metadata without network access.

        Hugging Face may modify its input dictionary while resolving attention settings.

        Args:
            config: Serialized CLIP vision configuration, including projection_dim.

        Returns:
            A CLIPViT instance with newly initialized parameters (checkpoint weights must be loaded separately).
        """
        return cls(CLIPVisionConfig.from_dict(deepcopy(config)))

    @classmethod
    def from_clip(cls, model: CLIPModel) -> "CLIPViT":
        """Copy the vision encoder and projection from a pretrained CLIP model.

        Args:
            model: Hugging Face CLIP model with pretrained vision and text parameters.

        Returns:
            A CLIPViT instance containing the pretrained vision parameters.
        """
        config = model.config.vision_config.to_dict()

        # The shared projection dimension belongs to the parent CLIP configuration
        config["projection_dim"] = model.config.projection_dim

        backbone = cls.from_config(config)
        backbone.encoder.vision_model.load_state_dict(model.vision_model.state_dict(), strict=True)
        backbone.encoder.visual_projection.load_state_dict(model.visual_projection.state_dict(), strict=True)

        return backbone

    @property
    def feature_dim(self) -> int:
        """Return the dimension of the projected image features.

        Args:
            None.

        Returns:
            The output feature dimension specified by projection_dim.
        """
        return self.encoder.config.projection_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract projected image features from preprocessed pixels.

        Args:
            inputs: Image tensor of shape [batch_size, 3, height, width].

        Returns:
            Projected image features of shape [batch_size, feature_dim].
        """
        return self.encoder(pixel_values=inputs, return_dict=True).image_embeds

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for the vision encoder.

        Recompute intermediate activations during the backward pass to reduce
        activation memory usage at the cost of additional computation.

        Args:
            None.

        Returns:
            None.
        """
        self.encoder.gradient_checkpointing_enable()


class ClassificationHead(nn.Module):
    """ClassificationHead class for a frozen, prompt-initialized CLIP classifier.

    Attributes:
        weight: Normalized text prototypes of shape [num_classes, feature_dim].
        logit_scale: Fixed positive multiplier applied to cosine similarities.
    """

    def __init__(self, weight: torch.Tensor, logit_scale: float) -> None:
        """Initialize the frozen classification head.

        Args:
            weight: Text prototypes in the exact dataset label order.
            logit_scale: Exponential of the pretrained CLIP logit scale.

        Returns:
            None.
        """
        super().__init__()

        # Error check
        if weight.ndim != 2 or not torch.isfinite(weight).all():
            raise ValueError("Classification weights must be a finite matrix.")
        if (weight.norm(dim=-1) == 0).any() or not 0 < logit_scale < float("inf"):
            raise ValueError("Prototypes must be nonzero and logit_scale finite and positive.")

        # Class attributes
        self.register_buffer("weight", F.normalize(weight.detach().float(), dim=-1))
        self.register_buffer("logit_scale", torch.tensor(logit_scale, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute class logits from projected image features.

        Args:
            features: Projected image features of shape [batch_size, feature_dim].

        Returns:
            Class logits of shape [batch_size, num_classes] computed in float32.
        """
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = F.normalize(features.float(), dim=-1)

            return self.logit_scale * F.linear(features, self.weight.float())
