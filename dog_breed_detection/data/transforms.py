import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def get_train_transforms(config: DictConfig) -> A.Compose:
    """Get training transforms with augmentations.

    Based on the reference solution:
    - Resize to 256, then CenterCrop to 224 (better than direct resize)
    - HorizontalFlip with p=0.6
    - RandomRotation up to 30 degrees

    Args:
        config: Hydra configuration object.

    Returns:
        Albumentations Compose object with training transforms.
    """
    image_size = config.data.image_size
    resize_size = int(image_size * 256 / 224)

    return A.Compose(
        [
            A.Resize(resize_size, resize_size),
            A.CenterCrop(image_size, image_size),
            A.HorizontalFlip(p=0.6),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms(config: DictConfig) -> A.Compose:
    """Get validation/test transforms (no augmentation).

    Uses same resize + center crop as training for consistency.

    Args:
        config: Hydra configuration object.

    Returns:
        Albumentations Compose object with validation transforms.
    """
    image_size = config.data.image_size
    resize_size = int(image_size * 256 / 224)

    return A.Compose(
        [
            A.Resize(resize_size, resize_size),
            A.CenterCrop(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
