"""Project configuration: paths, constants, and configs/config.yaml loading."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SCALER_PATH = MODELS_DIR / "scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"
SEGMENT_MAPPING_PATH = MODELS_DIR / "segment_mapping.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Model versioning: bump when the modeling contract changes.
MODEL_VERSION = "rfm-kmeans-v1"

# Raw transaction schema (UCI Online Retail).
RAW_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

_DEFAULTS: dict[str, Any] = {
    "dataset": {"path": "data/Online_Retail.xlsx"},
    "model": {"random_state": 42, "n_init": 10, "k_min": 2, "k_max": 8},
    "stability": {"seeds": [42, 7, 21, 52, 101], "reference_seed": 42},
    "k_selection": {"min_cluster_share": 0.02, "min_ari": 0.85, "min_business_k": 3},
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load configs/config.yaml, falling back to safe defaults per section."""
    config = {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in _DEFAULTS.items()
    }
    if Path(path).exists():
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        for section, values in loaded.items():
            if isinstance(values, dict) and section in config:
                config[section].update(values)
            else:
                config[section] = values
    return config


CONFIG = load_config()

DATASET_PATH = PROJECT_ROOT / CONFIG["dataset"]["path"]
RANDOM_STATE = int(CONFIG["model"]["random_state"])
N_INIT = int(CONFIG["model"]["n_init"])
K_MIN = int(CONFIG["model"]["k_min"])
K_MAX = int(CONFIG["model"]["k_max"])
STABILITY_SEEDS = list(CONFIG["stability"]["seeds"])
STABILITY_REFERENCE_SEED = int(CONFIG["stability"]["reference_seed"])
K_SELECTION = dict(CONFIG["k_selection"])
