import logging
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)


class TqdmUpTo(tqdm):
    """Progress bar for urlretrieve."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        """Update progress bar.

        Args:
            b: Number of blocks transferred so far.
            bsize: Size of each block.
            tsize: Total size of file.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar.

    Args:
        url: URL to download from.
        output_path: Path to save the file.
        desc: Description for progress bar.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=desc) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract archive (tar.gz or zip) to directory.

    Args:
        archive_path: Path to the archive file.
        extract_to: Directory to extract to.
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {archive_path.name} to {extract_to}...")

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffixes == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.suffixes == [".tar", ".bz2"]:
        with tarfile.open(archive_path, "r:bz2") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

    logger.info("Extraction complete!")


def download_stanford_dogs(data_dir: Path) -> None:
    """Download Stanford Dogs Dataset.

    Args:
        data_dir: Directory to save the dataset.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
    urls = {
        "images": f"{base_url}images.tar",
        "annotations": f"{base_url}annotation.tar",
        "lists": f"{base_url}lists.tar",
    }

    archives_dir = data_dir / "archives"
    archives_dir.mkdir(exist_ok=True)

    for name, url in urls.items():
        archive_path = archives_dir / f"{name}.tar"
        if not archive_path.exists():
            logger.info(f"Downloading {name}...")
            try:
                download_file(url, archive_path, desc=f"Downloading {name}")
            except Exception as e:
                logger.warning(f"Could not download {name} from {url}: {e}")
                logger.warning("You may need to download it manually.")
                continue

        extract_dir = data_dir / name
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            extract_archive(archive_path, data_dir)

    logger.info("Dataset download complete!")
    logger.info(f"Data saved to: {data_dir}")


def extract_dvc_archives(data_dir: Path | str = "data") -> None:
    """Extract archives pulled from DVC.

    This function extracts annotations.tar.gz and images.tar.gz
    that are stored in DVC.

    Args:
        data_dir: Directory containing the archives.
    """
    data_dir = Path(data_dir)

    archives = [
        ("annotations.tar.gz", "annotations"),
        ("images.tar.gz", "images"),
    ]

    for archive_name, extract_name in archives:
        archive_path = data_dir / archive_name
        extract_dir = data_dir / extract_name

        if archive_path.exists():
            if not extract_dir.exists() or not any(extract_dir.iterdir()):
                logger.info(f"Extracting {archive_name}...")
                extract_archive(archive_path, data_dir)
                logger.info(f"Extracted to {extract_dir}")
            else:
                logger.info(f"{extract_name} already extracted, skipping.")
        else:
            logger.warning(f"{archive_path} not found. Run 'dvc pull' first.")


def download_data(data_dir: Path | str = "data") -> None:
    """Download and extract dog breed dataset.

    This function first tries to use DVC to pull data, then extracts archives.
    If DVC fails, it attempts to download from Stanford Dogs Dataset.

    Args:
        data_dir: Directory to save the dataset.
    """
    import subprocess

    data_dir = Path(data_dir)
    logger.info(f"Preparing dataset in {data_dir}...")

    images_archive = data_dir / "images.tar.gz"
    annotations_archive = data_dir / "annotations.tar.gz"

    if not images_archive.exists() or not annotations_archive.exists():
        logger.info("Archives not found. Pulling from DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True)
            logger.info("DVC pull successful!")
        except subprocess.CalledProcessError:
            logger.warning("DVC pull failed. Trying to download from source...")
            download_stanford_dogs(data_dir)
            return

    extract_dvc_archives(data_dir)
    logger.info("Dataset ready!")
