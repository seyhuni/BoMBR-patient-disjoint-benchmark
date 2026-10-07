"""Veri kümesi, augmentasyon ve maskeden morfometrik özellik çıkarımı."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from skimage.measure import label, regionprops
from torch.utils.data import Dataset


CLASS_NAMES = ["Bone", "Fat", "Cell", "Reticulin"]


def read_index_mask(path: str | Path) -> np.ndarray:
    """0-3 sınıf indeksleri içeren maskeyi güvenli biçimde okur."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Maske okunamadı: {path}")
    if mask.ndim == 3:
        if np.array_equal(mask[..., 0], mask[..., 1]) and np.array_equal(mask[..., 1], mask[..., 2]):
            mask = mask[..., 0]
        else:
            raise ValueError(f"Maske 3-kanallı ve indeksli değil: {path}")
    mask = mask.astype(np.int64)
    values = np.unique(mask)
    if not np.isin(values, [0, 1, 2, 3]).all():
        raise ValueError(f"Maskede geçersiz sınıf değerleri var ({values.tolist()}): {path}")
    return mask


def make_augmentation(image_size: int, train: bool) -> A.Compose:
    """Görüntü ve maskeye eş zamanlı uygulanabilen albumentations akışı."""
    if train:
        return A.Compose([
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.50),
            A.VerticalFlip(p=0.50),
            A.Rotate(limit=90, border_mode=cv2.BORDER_REFLECT_101, p=0.50),
            A.ElasticTransform(alpha=25, sigma=6, alpha_affine=6, border_mode=cv2.BORDER_REFLECT_101, p=0.20),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03, p=0.50),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.30),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class BoMBRDataset(Dataset):
    """Hasta-safe manifestten görüntü, maske, grade ve metadata sağlar."""

    def __init__(self, manifest: pd.DataFrame, image_size: int, train: bool, include_mask: bool = True):
        self.frame = manifest.reset_index(drop=True).copy()
        self.transform = make_augmentation(image_size, train=train)
        self.include_mask = include_mask

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        row = self.frame.iloc[index]
        image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {row.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        payload: dict[str, torch.Tensor | str | int]
        if self.include_mask:
            mask = read_index_mask(row.mask_path_resolved)
            transformed = self.transform(image=image, mask=mask)
            mask_out = torch.from_numpy(transformed["mask"].astype(np.int64))
        else:
            transformed = self.transform(image=image)
            mask_out = None
        image_out = torch.from_numpy(np.transpose(transformed["image"], (2, 0, 1)).astype(np.float32))
        payload = {
            "image": image_out,
            "grade": int(row.mf_grade),
            "patient": str(row.patient),
            "filename": str(row.filename),
        }
        if mask_out is not None:
            payload["mask"] = mask_out
        return payload


def extract_morphometrics(mask: np.ndarray) -> dict[str, float]:
    """Retikülin topolojisi ve sınıf oranlarından 12 açıklanabilir özellik çıkarır."""
    if mask.ndim != 2:
        raise ValueError("Morfometri için maske 2-boyutlu olmalı.")
    total = float(mask.size)
    features: dict[str, float] = {}
    for class_id, name in enumerate(["bone", "fat", "cell", "reticulin"]):
        features[f"pct_{name}"] = float((mask == class_id).sum() / total)

    ret = (mask == 3).astype(np.uint8)
    components = label(ret, connectivity=2)
    props = regionprops(components)
    areas = np.array([p.area for p in props], dtype=float)
    # major_axis/minor_axis ile retikülin liflerinin uzamışlığı tanımlanır.
    elongations = np.array([
        (p.major_axis_length / max(p.minor_axis_length, 1e-6)) for p in props if p.area >= 3
    ], dtype=float)
    perimeters = np.array([p.perimeter for p in props if p.area >= 3], dtype=float)

    features["ret_component_count"] = float(len(props))
    features["ret_mean_component_area"] = float(areas.mean()) if len(areas) else 0.0
    features["ret_median_elongation"] = float(np.median(elongations)) if len(elongations) else 0.0
    features["ret_elongation_sd"] = float(np.std(elongations)) if len(elongations) else 0.0
    features["ret_mean_perimeter"] = float(perimeters.mean()) if len(perimeters) else 0.0
    features["ret_to_cell_ratio"] = float(features["pct_reticulin"] / max(features["pct_cell"], 1e-6))
    features["ret_lacunarity_proxy"] = _lacunarity_proxy(ret)
    features["ret_fractal_proxy"] = _box_count_fractal_dimension(ret)
    return features


def _lacunarity_proxy(binary: np.ndarray, box: int = 16) -> float:
    """Kutu kütle dağılımının varyans/ortalama² oranı ile lacunarity yaklaşımı."""
    h, w = binary.shape
    h2, w2 = h - h % box, w - w % box
    if h2 == 0 or w2 == 0:
        return 0.0
    blocks = binary[:h2, :w2].reshape(h2 // box, box, w2 // box, box).sum(axis=(1, 3)).ravel()
    mean = blocks.mean()
    return float(blocks.var() / (mean * mean + 1e-8) + 1.0)


def _box_count_fractal_dimension(binary: np.ndarray) -> float:
    """Retikülin pikselleri için basit box-counting fraktal boyut yaklaşımı."""
    if binary.sum() == 0:
        return 0.0
    sizes = [2, 4, 8, 16, 32]
    counts: list[int] = []
    valid_sizes: list[int] = []
    h, w = binary.shape
    for size in sizes:
        h2, w2 = h - h % size, w - w % size
        if h2 < size or w2 < size:
            continue
        grid = binary[:h2, :w2].reshape(h2 // size, size, w2 // size, size)
        count = int((grid.sum(axis=(1, 3)) > 0).sum())
        if count > 0:
            valid_sizes.append(size)
            counts.append(count)
    if len(counts) < 2:
        return 0.0
    slope = np.polyfit(np.log(1 / np.asarray(valid_sizes)), np.log(np.asarray(counts)), 1)[0]
    return float(slope)


def feature_frame_from_masks(mask_paths: Iterable[str | Path], filenames: Iterable[str]) -> pd.DataFrame:
    records = []
    for path, filename in zip(mask_paths, filenames):
        metrics = extract_morphometrics(read_index_mask(path))
        metrics["filename"] = filename
        records.append(metrics)
    return pd.DataFrame(records)
