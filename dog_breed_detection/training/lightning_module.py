import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import Accuracy, F1Score

from dog_breed_detection.models.classifier import DogBreedClassifier


class DogBreedLightningModule(pl.LightningModule):
    """Lightning module for training dog breed classifier.

    Args:
        config: Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = DogBreedClassifier(config)

        # Metrics
        num_classes = config.model.num_classes
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss (MultiClassLogLoss)."""
        return F.cross_entropy(logits, targets)

    def training_step(self, batch: tuple) -> torch.Tensor:
        """Training step.

        Args:
            batch: Tuple of (images, labels).
            batch_idx: Index of the batch.

        Returns:
            Training loss.
        """
        images, labels = batch
        logits = self(images)
        loss = self._compute_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Tuple of (images, labels).

        Returns:
            Validation loss for proper aggregation.
        """
        images, labels = batch
        logits = self(images)
        loss = self._compute_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)

        # Log loss with proper aggregation (mean across batches)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            reduce_fx="mean",
        )
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: tuple) -> None:
        """Test step.

        Args:
            batch: Tuple of (images, labels).
        """
        images, labels = batch
        logits = self(images)
        loss = self._compute_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)

        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Supports both SGD (as in reference solution) and AdamW.
        """
        optimizer_name = self.config.training.get("optimizer", "adamw").lower()

        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.training.learning_rate,
                momentum=self.config.training.get("momentum", 0.9),
                weight_decay=self.config.training.weight_decay,
            )
        else:  # adamw
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.epochs,
            eta_min=self.config.training.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
