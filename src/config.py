from dataclasses import dataclass
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = ROOT_DIR / "data"

# Output folders
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images"
MODELS_DIR = ROOT_DIR / "models"

# Create folders if missing
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Paths:
    data_path: Path = ROOT_DIR / "data" / "raw" / "insurance.csv"
    reports_dir: str = "reports"
    images_dir: str = "images"
    models_dir: str = "models"