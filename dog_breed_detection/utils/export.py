import logging
from pathlib import Path

import torch

from dog_breed_detection.training.lightning_module import DogBreedLightningModule

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    config,
    opset_version: int = 14,
    dynamic_batch: bool = True,
) -> Path:
    """Export model to ONNX format for Triton Inference Server.

    Args:
        checkpoint_path: Path to the Lightning checkpoint.
        output_path: Path to save the ONNX model.
        config: Hydra configuration object.
        opset_version: ONNX opset version (17 recommended for ViT).
        dynamic_batch: Enable dynamic batch size.

    Returns:
        Path to the exported ONNX model.
    """
    lightning_module = DogBreedLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location="cpu",
        weights_only=False,
    )
    lightning_module.eval()

    model = lightning_module.model
    model.eval()

    image_size = config.data.image_size
    dummy_input = torch.randn(1, 3, image_size, image_size)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    logger.info(f"Model exported to ONNX: {output_path}")
    logger.info(f"  Input shape: (batch_size, 3, {image_size}, {image_size})")
    logger.info(f"  Output shape: (batch_size, {config.model.num_classes})")

    return output_path


def export_to_torchscript(
    checkpoint_path: Path,
    output_path: Path,
    config,
    use_trace: bool = True,
) -> Path:
    """Export model to TorchScript format for Triton Inference Server.

    Args:
        checkpoint_path: Path to the Lightning checkpoint.
        output_path: Path to save the TorchScript model.
        config: Hydra configuration object.
        use_trace: Use tracing (True) or scripting (False).

    Returns:
        Path to the exported TorchScript model.
    """
    lightning_module = DogBreedLightningModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        map_location="cpu",
        weights_only=False,
    )
    lightning_module.eval()

    # Extract the inner model (DogBreedClassifier) to avoid Lightning trainer issues
    model = lightning_module.model
    model.eval()

    image_size = config.data.image_size
    dummy_input = torch.randn(1, 3, image_size, image_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scripted_model = torch.jit.trace(model, dummy_input) if use_trace else torch.jit.script(model)
    scripted_model.save(str(output_path))

    logger.info(f"Model exported to TorchScript: {output_path}")
    logger.info(f"  Input shape: (batch_size, 3, {image_size}, {image_size})")
    logger.info(f"  Output shape: (batch_size, {config.model.num_classes})")

    return output_path


def create_triton_model_repository(
    checkpoint_path: Path,
    repository_path: Path,
    config,
    model_name: str = "dog_breed_classifier",
    export_format: str = "onnx",
    max_batch_size: int = 8,
) -> Path:
    """Create a complete Triton model repository structure.

    Args:
        checkpoint_path: Path to the Lightning checkpoint.
        repository_path: Path to create the model repository.
        config: Hydra configuration object.
        model_name: Name of the model in Triton.
        export_format: Export format ("onnx" or "torchscript").
        max_batch_size: Maximum batch size for Triton.

    Returns:
        Path to the model repository.
    """
    repository_path = Path(repository_path)
    model_dir = repository_path / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    image_size = config.data.image_size
    num_classes = config.model.num_classes

    if export_format == "onnx":
        model_file = version_dir / "model.onnx"
        export_to_onnx(checkpoint_path, model_file, config)
        platform = "onnxruntime_onnx"
    else:
        model_file = version_dir / "model.pt"
        export_to_torchscript(checkpoint_path, model_file, config)
        platform = "pytorch_libtorch"

    # Create config.pbtxt
    config_content = f"""name: "{model_name}"
platform: "{platform}"
max_batch_size: {max_batch_size}

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, {image_size}, {image_size} ]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [ {num_classes} ]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_AUTO
  }}
]

dynamic_batching {{
  preferred_batch_size: [ 1, 2, 4, 8 ]
  max_queue_delay_microseconds: 100
}}
"""

    config_path = model_dir / "config.pbtxt"
    with open(config_path, "w") as f:
        f.write(config_content)

    logger.info(f"Triton model repository created: {repository_path}")
    logger.info(f"  Model name: {model_name}")
    logger.info(f"  Format: {export_format}")
    logger.info(f"  Config: {config_path}")

    return repository_path
