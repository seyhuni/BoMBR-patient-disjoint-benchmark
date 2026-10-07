"""Dosya yolları, metadata ve yapılandırma için ortak yardımcılar."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

REQUIRED_METADATA_COLUMNS = {
    "filename", "mask_file", "split", "patient", "mf_grade", "hemorrhage"
}


def project_root() -> Path:
    """src/ klasöründen proje kökünü döndürür."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_root() / path


def set_global_seed(seed: int) -> None:
    """Tekrarlanabilirlik için temel rasgelelik kaynaklarını sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_torch() -> torch.device:
    """NVIDIA GPU varsa CUDA cihazını, yoksa CPU'yu döndürür."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("Using CPU")
    return torch.device("cpu")


def read_metadata(data_root: str | Path, metadata_name: str = "master_metadata.csv") -> pd.DataFrame:
    """Metadata'yı okur, sütunları doğrular ve tipleri standartlaştırır.

    Önemli: img_path/mask_path sütunlarındaki makineye özgü eski Windows yolları
    hiçbir aşamada kullanılmaz. Yollar filename ve mask_file sütunlarından türetilir.
    """
    data_root = Path(data_root)
    metadata_path = data_root / metadata_name
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata bulunamadı: {metadata_path}")

    df = pd.read_csv(metadata_path)
    absent = REQUIRED_METADATA_COLUMNS.difference(df.columns)
    if absent:
        raise ValueError(
            "Metadata zorunlu sütunları içermiyor: " + ", ".join(sorted(absent))
        )

    df = df.copy()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["patient"] = df["patient"].astype(str).str.strip()
    df["mf_grade"] = pd.to_numeric(df["mf_grade"], errors="raise").astype(int)
    df["hemorrhage"] = (
        df["hemorrhage"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    if df["hemorrhage"].isna().any():
        raise ValueError("hemorrhage sütununda yalnızca yes/no kullanılmalıdır.")
    df["hemorrhage"] = df["hemorrhage"].astype(int)

    legal_grades = set(df["mf_grade"].unique())
    if not legal_grades.issubset({0, 1, 2, 3}):
        raise ValueError(f"mf_grade 0-3 aralığında olmalı. Bulunan: {sorted(legal_grades)}")
    if not set(df["split"].unique()).issubset({"train", "test"}):
        raise ValueError("split sütununda yalnızca train/test değerleri kullanılmalıdır.")

    df["image_path"] = df.apply(
        lambda r: str(data_root / r["split"] / "images" / r["filename"]), axis=1
    )
    df["mask_path_resolved"] = df.apply(
        lambda r: str(data_root / r["split"] / "masks" / r["mask_file"]), axis=1
    )
    return df


def write_json(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"JSON'a dönüştürülemeyen tip: {type(value).__name__}")
