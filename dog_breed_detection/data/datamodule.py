from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dog_breed_detection.data.dataset import DogBreedDataset
from dog_breed_detection.data.transforms import get_train_transforms, get_val_transforms


class DogBreedDataModule(pl.LightningDataModule):
    """Lightning DataModule for dog breed dataset.

    Args:
        config: Hydra configuration object.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.data_dir = Path(config.data.data_dir)
        self.batch_size = config.training.batch_size
        self.num_workers = config.data.num_workers

        self.train_dataset: DogBreedDataset | None = None
        self.val_dataset: DogBreedDataset | None = None
        self.test_dataset: DogBreedDataset | None = None

    def prepare_data(self) -> None:
        # DVC pull is handled in commands.py
        pass

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        train_transform = get_train_transforms(self.config)
        val_transform = get_val_transforms(self.config)

        # Determine data directory structure
        # Stanford Dogs format: data_dir/images/n02085620-Chihuahua/...
        # or: data_dir/Images/n02085620-Chihuahua/...
        images_dir = None
        for possible_path in [
            self.data_dir / "images",
            self.data_dir / "Images",
            self.data_dir,
        ]:
            if possible_path.exists() and any(possible_path.iterdir()):
                images_dir = possible_path
                break

        if images_dir is None:
            raise ValueError(
                f"No images found in {self.data_dir}. "
                "Please run 'dog-breed download_data' first."
            )

        # Try to find labels file (optional - can use directory structure)
        labels_file = None
        for possible_path in [
            self.data_dir / "labels.csv",
            self.data_dir / "train_labels.csv",
        ]:
            if possible_path.exists():
                labels_file = possible_path
                break

        if stage == "fit" or stage is None:
            self.train_dataset = DogBreedDataset(
                data_dir=images_dir,
                labels_file=labels_file,
                transform=train_transform,
                split="train",
                val_split=self.config.data.val_split,
                test_split=self.config.data.test_split,
                random_seed=self.config.data.random_seed,
            )
            self.val_dataset = DogBreedDataset(
                data_dir=images_dir,
                labels_file=labels_file,
                transform=val_transform,
                split="val",
                val_split=self.config.data.val_split,
                test_split=self.config.data.test_split,
                random_seed=self.config.data.random_seed,
            )

            # Save class mapping after first dataset is created
            mapping_path = self.data_dir.parent / "class_mapping.json"
            if not mapping_path.exists():
                self.train_dataset.save_class_mapping(mapping_path)

        if stage == "test" or stage is None:
            self.test_dataset = DogBreedDataset(
                data_dir=images_dir,
                labels_file=labels_file,
                transform=val_transform,
                split="test",
                val_split=self.config.data.val_split,
                test_split=self.config.data.test_split,
                random_seed=self.config.data.random_seed,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_classes(self) -> int:
        if self.train_dataset is not None:
            return self.train_dataset.num_classes
        return self.config.model.num_classes
