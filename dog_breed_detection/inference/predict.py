import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image

from dog_breed_detection.data.transforms import get_val_transforms
from dog_breed_detection.training.lightning_module import DogBreedLightningModule


def load_class_mapping(mapping_path: Path) -> dict[int, str]:
    """Load class to index mapping from JSON file.

    Args:
        mapping_path: Path to the mapping file.

    Returns:
        Dictionary mapping class index to breed name.
    """
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping file not found: {mapping_path}. "
            "Please run training first to generate the mapping."
        )

    with open(mapping_path) as f:
        mapping = json.load(f)

    idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
    return idx_to_class


def predict_breed(
    image_path: Path,
    checkpoint_path: Path,
    config: DictConfig,
    mapping_path: Path | None = None,
) -> dict:
    """Predict dog breed from an image.

    Args:
        image_path: Path to the input image.
        checkpoint_path: Path to the model checkpoint.
        config: Hydra configuration object.
        mapping_path: Path to class mapping JSON file. If None, tries to find it automatically.

    Returns:
        Dictionary with predicted breed and confidence.
    """

    model = DogBreedLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        weights_only=False,
    )
    model.eval()

    transform = get_val_transforms(config)

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    if mapping_path is None:
        possible_paths = [
            Path(config.data.data_dir).parent / "class_mapping.json",
            Path("class_mapping.json"),
            Path("data") / "class_mapping.json",
        ]
        mapping_path = next((p for p in possible_paths if p.exists()), None)

    if mapping_path and mapping_path.exists():
        idx_to_class = load_class_mapping(mapping_path)
        breed_name = idx_to_class.get(predicted_idx.item(), f"breed_{predicted_idx.item()}")
    else:
        breed_name = f"breed_{predicted_idx.item()}"

    return {
        "breed": breed_name,
        "confidence": confidence.item(),
        "class_idx": predicted_idx.item(),
        "probabilities": probabilities.squeeze().tolist(),
    }
