import logging
import subprocess

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

import mlflow
from dog_breed_detection.data.datamodule import DogBreedDataModule
from dog_breed_detection.training.lightning_module import DogBreedLightningModule

logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:8]
    except subprocess.CalledProcessError:
        return "unknown"


def run_training(config: DictConfig) -> None:
    """Run model training.

    Args:
        config: Hydra configuration object.
    """
    from pathlib import Path

    from dog_breed_detection.utils.download import extract_dvc_archives

    data_dir = Path(config.data.data_dir)
    images_dir = data_dir / "images"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        logger.info("Data not found. Checking for archives...")

        images_archive = data_dir / "images.tar.gz"
        if not images_archive.exists():
            logger.info("Archives not found. Pulling from DVC...")
            try:
                subprocess.run(["dvc", "pull"], check=True)
                logger.info("DVC pull successful!")
            except subprocess.CalledProcessError:
                logger.error("DVC pull failed. Please run 'dog-breed download_data' first.")
                raise

        extract_dvc_archives(data_dir)

    pl.seed_everything(config.training.seed, workers=True)

    data_module = DogBreedDataModule(config)
    model = DogBreedLightningModule(config)

    mlflow_logger = MLFlowLogger(
        experiment_name=config.logging.experiment_name,
        tracking_uri=config.logging.mlflow_uri,
        run_name=config.logging.run_name,
    )

    mlflow_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))
    mlflow.log_param("git_commit", get_git_commit_id())

    callbacks = [
        ModelCheckpoint(
            dirpath=config.training.checkpoint_dir,
            filename="best-epoch{epoch:02d}-valloss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=config.training.early_stopping_patience,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        logger=mlflow_logger,
        callbacks=callbacks,
        deterministic=True,
        precision=config.training.precision,
    )

    trainer.fit(model, data_module)

    if config.training.run_test:
        trainer.test(model, data_module)

    best_model_path = callbacks[0].best_model_path
    logger.info(f"Training completed! Best model saved at: {best_model_path}")

    # Export to ONNX for Triton if enabled
    if config.training.get("export_onnx", False):
        from dog_breed_detection.utils.export import export_to_onnx

        onnx_path = Path(config.training.checkpoint_dir) / "model.onnx"
        export_to_onnx(
            checkpoint_path=Path(best_model_path),
            output_path=onnx_path,
            config=config,
        )
        mlflow.log_artifact(str(onnx_path))
