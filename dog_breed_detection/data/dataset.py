import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class DogBreedDataset(Dataset):
    """Dataset for loading dog breed images.

    Supports two data formats:
    1. CSV file with columns: image_path, breed (or similar)
    2. Directory structure: data_dir/breed_name/image.jpg

    Args:
        data_dir: Path to the directory containing images.
        labels_file: Path to the CSV file with image labels (optional if using directory structure).
        transform: Optional transform to apply to images (albumentations).
        split: Dataset split ('train', 'val', 'test').
        val_split: Validation split ratio (used if split='val').
        test_split: Test split ratio (used if split='test').
        random_seed: Random seed for train/val/test split.
    """

    def __init__(
        self,
        data_dir: Path,
        labels_file: Path | None = None,
        transform=None,
        split: str = "train",
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.labels_file = Path(labels_file) if labels_file else None
        self.transform = transform
        self.split = split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}

        self._load_data()

    def _load_data(self) -> None:
        # Try to load from CSV file first
        if self.labels_file and self.labels_file.exists():
            self._load_from_csv()
        else:
            # Load from directory structure (breed folders)
            self._load_from_directory()

        # Apply train/val/test split
        self._apply_split()

    def _load_from_csv(self) -> None:
        df = pd.read_csv(self.labels_file)

        if "breed" in df.columns:
            breed_col = "breed"
        elif "label" in df.columns:
            breed_col = "label"
        else:
            raise ValueError("CSV must contain 'breed' or 'label' column")

        if "image_path" in df.columns:
            path_col = "image_path"
        elif "id" in df.columns:
            path_col = "id"
        else:
            raise ValueError("CSV must contain 'image_path' or 'id' column")

        unique_breeds = sorted(df[breed_col].unique())
        self.class_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
        self.idx_to_class = {idx: breed for breed, idx in self.class_to_idx.items()}

        # Build samples list
        all_samples = []
        for _, row in df.iterrows():
            image_path = self.data_dir / row[path_col]
            if not image_path.exists():
                # Try with .jpg extension if not found
                image_path = self.data_dir / f"{row[path_col]}.jpg"
            if image_path.exists():
                breed = row[breed_col]
                label = self.class_to_idx[breed]
                all_samples.append((image_path, label))

        self.samples = all_samples

    def _load_from_directory(self) -> None:
        # Find all breed directories
        breed_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if not breed_dirs:
            raise ValueError(
                f"No breed directories found in {self.data_dir}. "
                "Expected structure: data_dir/breed_name/image.jpg"
            )

        # Build class mapping
        breed_names = sorted([d.name for d in breed_dirs])
        self.class_to_idx = {breed: idx for idx, breed in enumerate(breed_names)}
        self.idx_to_class = {idx: breed for breed, idx in self.class_to_idx.items()}

        # Build samples list
        all_samples = []
        for breed_dir in breed_dirs:
            breed_name = breed_dir.name
            label = self.class_to_idx[breed_name]

            # Find all images in breed directory
            image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
            images = [
                img_path for img_path in breed_dir.iterdir() if img_path.suffix in image_extensions
            ]

            for image_path in images:
                all_samples.append((image_path, label))

        self.samples = all_samples

    def _apply_split(self) -> None:
        if self.split == "train":
            train_samples, temp = train_test_split(
                self.samples,
                test_size=self.val_split + self.test_split,
                random_state=self.random_seed,
                stratify=[label for _, label in self.samples],
            )
            self.samples = train_samples
        elif self.split == "val":
            train_samples, temp = train_test_split(
                self.samples,
                test_size=self.val_split + self.test_split,
                random_state=self.random_seed,
                stratify=[label for _, label in self.samples],
            )
            val_size = self.val_split / (self.val_split + self.test_split)
            val_samples, _ = train_test_split(
                temp,
                test_size=1 - val_size,
                random_state=self.random_seed,
                stratify=[label for _, label in temp],
            )
            self.samples = val_samples
        elif self.split == "test":
            train_samples, temp = train_test_split(
                self.samples,
                test_size=self.val_split + self.test_split,
                random_state=self.random_seed,
                stratify=[label for _, label in self.samples],
            )
            val_size = self.val_split / (self.val_split + self.test_split)
            _, test_samples = train_test_split(
                temp,
                test_size=1 - val_size,
                random_state=self.random_seed,
                stratify=[label for _, label in temp],
            )
            self.samples = test_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image tensor, label).
        """
        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor if no transform
            from torchvision import transforms

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            image = transform(image)

        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def save_class_mapping(self, output_path: Path) -> None:
        """Save class to index mapping to JSON file.

        Args:
            output_path: Path to save the mapping file.
        """
        mapping = {
            "class_to_idx": self.class_to_idx,
            "idx_to_class": {str(k): v for k, v in self.idx_to_class.items()},
        }
        with open(output_path, "w") as f:
            json.dump(mapping, f, indent=2)

    @classmethod
    def load_class_mapping(cls, mapping_path: Path) -> tuple[dict[str, int], dict[int, str]]:
        """Load class to index mapping from JSON file.

        Args:
            mapping_path: Path to the mapping file.

        Returns:
            Tuple of (class_to_idx, idx_to_class) dictionaries.
        """
        with open(mapping_path) as f:
            mapping = json.load(f)
        class_to_idx = mapping["class_to_idx"]
        idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
        return class_to_idx, idx_to_class
