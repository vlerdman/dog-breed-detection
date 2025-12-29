import logging

import fire

from dog_breed_detection.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class Commands:
    """CLI commands for dog breed detection project."""

    def train(
        self,
        config_name: str = "config",
        overrides: list[str] | None = None,
    ) -> None:
        """Run model training.

        Args:
            config_name: Name of the config file (without .yaml extension).
            overrides: List of Hydra overrides (e.g., ["training.epochs=10"]).
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        from dog_breed_detection.training.train import run_training

        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name=config_name, overrides=overrides or [])
            run_training(cfg)

    def infer(
        self,
        image_path: str,
        checkpoint_path: str,
        config_name: str = "config",
        mapping_path: str | None = None,
    ) -> None:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image.
            checkpoint_path: Path to the model checkpoint.
            config_name: Name of the config file.
            mapping_path: Path to class mapping JSON file (optional).
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        from dog_breed_detection.inference.predict import predict_breed

        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name=config_name)
            mapping = Path(mapping_path) if mapping_path else None
            result = predict_breed(
                image_path=Path(image_path),
                checkpoint_path=Path(checkpoint_path),
                config=cfg,
                mapping_path=mapping,
            )
            logger.info(f"Predicted breed: {result['breed']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"Class index: {result['class_idx']}")

    def download_data(self, data_dir: str = "data") -> None:
        """Download and extract dataset.

        This command pulls archives from DVC and extracts them.

        Args:
            data_dir: Directory where data is stored.
        """
        from pathlib import Path

        from dog_breed_detection.utils.download import download_data

        download_data(Path(data_dir))

    def extract_data(self, data_dir: str = "data") -> None:
        """Extract archives without pulling from DVC.

        Use this if you already have archives locally.

        Args:
            data_dir: Directory containing the archives.
        """
        from pathlib import Path

        from dog_breed_detection.utils.download import extract_dvc_archives

        extract_dvc_archives(Path(data_dir))

    def export(
        self,
        checkpoint_path: str,
        output_path: str = "model_repository",
        config_name: str = "config",
        format: str = "onnx",
        model_name: str = "dog_breed_classifier",
    ) -> None:
        """Export model for NVIDIA Triton Inference Server.

        Creates a complete Triton model repository with config.pbtxt.

        Args:
            checkpoint_path: Path to the trained model checkpoint.
            output_path: Path to create the model repository.
            config_name: Name of the config file.
            format: Export format ("onnx" or "torchscript").
            model_name: Name of the model in Triton.
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        from dog_breed_detection.utils.export import create_triton_model_repository

        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name=config_name)
            create_triton_model_repository(
                checkpoint_path=Path(checkpoint_path),
                repository_path=Path(output_path),
                config=cfg,
                model_name=model_name,
                export_format=format,
            )

    def export_onnx(
        self,
        checkpoint_path: str,
        output_path: str = "model.onnx",
        config_name: str = "config",
    ) -> None:
        """Export model to ONNX format only (without Triton config).

        Args:
            checkpoint_path: Path to the trained model checkpoint.
            output_path: Path to save the ONNX model.
            config_name: Name of the config file.
        """
        from pathlib import Path

        from hydra import compose, initialize_config_dir

        from dog_breed_detection.utils.export import export_to_onnx

        config_dir = Path(__file__).parent.parent / "configs"

        with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base="1.3"):
            cfg = compose(config_name=config_name)
            export_to_onnx(
                checkpoint_path=Path(checkpoint_path),
                output_path=Path(output_path),
                config=cfg,
            )

    def infer_triton(
        self,
        image_path: str,
        triton_url: str = "localhost:8000",
        model_name: str = "dog_breed_classifier",
        mapping_path: str | None = None,
    ) -> None:
        """Run inference using Triton Inference Server.

        Args:
            image_path: Path to the input image.
            triton_url: Triton server URL (host:port).
            model_name: Name of the model in Triton.
            mapping_path: Path to class mapping JSON file.
        """
        import json
        from pathlib import Path

        import numpy as np
        import tritonclient.http as httpclient
        from PIL import Image

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        image = Image.open(image_path).convert("RGB")
        image_size = 224
        resize_size = int(image_size * 256 / 224)
        image = image.resize((resize_size, resize_size), Image.BILINEAR)

        left = (resize_size - image_size) // 2
        top = (resize_size - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))

        image_np = np.array(image, dtype=np.float32) / 255.0
        image_np = (image_np - mean) / std
        image_np = image_np.transpose(2, 0, 1)
        image_np = np.expand_dims(image_np, axis=0).astype(np.float32)

        client = httpclient.InferenceServerClient(url=triton_url)

        if not client.is_model_ready(model_name):
            logger.error(f"Model {model_name} is not ready on {triton_url}")
            return

        logger.info(f"Model {model_name} is ready")
        logger.info(f"Input shape: {image_np.shape}")

        inputs = [httpclient.InferInput("input", image_np.shape, "FP32")]
        inputs[0].set_data_from_numpy(image_np)
        outputs = [httpclient.InferRequestedOutput("output")]

        logger.info("Running inference...")
        response = client.infer(model_name, inputs, outputs=outputs)
        output_data = response.as_numpy("output")

        exp_scores = np.exp(output_data - np.max(output_data, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        predicted_idx = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(probabilities[0, predicted_idx])

        breed_name = f"class_{predicted_idx}"
        idx_to_class = {}
        mapping_file = Path(mapping_path) if mapping_path else Path("class_mapping.json")
        if mapping_file.exists():
            with open(mapping_file) as f:
                mapping = json.load(f)
            idx_to_class = {int(k): v for k, v in mapping["idx_to_class"].items()}
            breed_name = idx_to_class.get(predicted_idx, breed_name)

        top5_indices = np.argsort(probabilities[0])[::-1][:5]

        logger.info("=" * 50)
        logger.info(f"Predicted breed: {breed_name}")
        logger.info(f"Confidence: {confidence:.2%}")
        logger.info(f"Class index: {predicted_idx}")
        logger.info("Top 5 predictions:")
        for i, idx in enumerate(top5_indices, 1):
            idx = int(idx)
            name = idx_to_class.get(idx, f"class_{idx}")
            logger.info(f"  {i}. {name}: {probabilities[0, idx]:.2%}")
        logger.info("=" * 50)


def main() -> None:
    setup_logging()
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
