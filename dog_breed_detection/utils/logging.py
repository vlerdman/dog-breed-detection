import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger instance.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger("dog_breed_detection")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Name of the logger (usually __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"dog_breed_detection.{name}")
