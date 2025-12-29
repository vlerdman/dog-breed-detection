import logging

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class DogBreedClassifier(nn.Module):
    """Vision Transformer based classifier for dog breeds.

    Uses a pretrained ViT model from timm library with a custom classification head.
    Supports freezing backbone to train only the classification head (much faster).

    Args:
        config: Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.model.num_classes
        self.model_name = config.model.backbone
        self.freeze_backbone = config.model.get("freeze_backbone", True)

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=config.model.pretrained,
            num_classes=self.num_classes,
            drop_rate=config.model.dropout,
        )

        if self.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        head_names = {"head", "fc", "classifier", "heads"}

        for name, param in self.backbone.named_parameters():
            is_head = any(head_name in name for head_name in head_names)
            if not is_head:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Trainable parameters: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification head.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Feature tensor.
        """
        return self.backbone.forward_features(x)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen for fine-tuning.")
