"""Dataset — CSV with image_path,lat,lon + torchvision transforms 224.

4050 constraints: image_size 224 by default, batch 32→16 fallback noted in train.py.
Windows paths safe via pathlib.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Transforms — 224 default (Section 3). 384 only if VRAM allows.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, train: bool = True) -> Callable:
    """Return torchvision transform pipeline.

    Train: Resize/CenterCrop or RandomResizedCrop + hflip + normalize.
    Eval:  Resize + CenterCrop + normalize.

    Args:
        image_size: target spatial size (224 default, 384 optional).
        train: whether to include augmentation.
    """
    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize(int(image_size * 256 / 224)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


# ---------------------------------------------------------------------------
# Core dataset — CSV: image_path,lat,lon
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"image_path", "lat", "lon"}


class GeoDataset(Dataset):
    """CSV-backed geolocation dataset.

    CSV header must contain: image_path,lat,lon
    image_path may be absolute or relative to the CSV file's directory.

    Returns:
        (image_tensor, label) where label is cell id if cells provided,
        otherwise (lat, lon) tuple — train.py handles both.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int = 224,
        train: bool = True,
        transform: Optional[Callable] = None,
        cell_ids: Optional[List[int]] = None,
    ) -> None:
        self.csv_path = Path(csv_path).resolve()
        self.root = self.csv_path.parent
        self.train = train
        self.image_size = image_size
        self.transform = transform if transform is not None else get_transforms(image_size, train=train)
        # cell_ids: pre-assigned cell index per row (len == num rows), optional
        self.cell_ids = cell_ids

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        # Use pandas for robustness; fallback to csv module if needed
        self.df = pd.read_csv(self.csv_path)

        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV {self.csv_path} missing columns: {missing}. Found: {list(self.df.columns)}")

        # Validate lat/lon ranges
        if not self.df["lat"].between(-90, 90).all():
            raise ValueError("CSV contains lat outside [-90, 90]")
        if not self.df["lon"].between(-180, 180).all():
            raise ValueError("CSV contains lon outside [-180, 180]")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        # Relative to CSV directory (Windows-safe)
        return (self.root / p).resolve()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor | int]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row["image_path"]))
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image not found: {img_path} (row {idx} in {self.csv_path})") from e

        image = self.transform(img)

        # Label: cell id if assigned, else raw lat/lon tensor
        if self.cell_ids is not None:
            label = int(self.cell_ids[idx])
            return image, label
        else:
            lat = float(row["lat"])
            lon = float(row["lon"])
            return image, torch.tensor([lat, lon], dtype=torch.float32)

    def get_coords(self) -> torch.Tensor:
        """Return Nx2 tensor of (lat, lon) for cell building."""
        return torch.tensor(self.df[["lat", "lon"]].values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers — subset loader + stratified sampler stub
# ---------------------------------------------------------------------------


def load_subset(
    csv_path: str | Path,
    max_samples: Optional[int] = None,
    image_size: int = 224,
    train: bool = True,
    seed: int = 42,
) -> GeoDataset | Subset:
    """Load dataset and optionally return a random subset.

    For stratified sampling by country/region, see `get_stratified_sampler`
    (stub in L0, implemented when cell labels exist).

    Args:
        csv_path: path to metadata.csv
        max_samples: if set, randomly sample at most this many rows (seeded).
        image_size: passed to GeoDataset
        train: train vs eval transforms
        seed: RNG seed for subset sampling
    """
    ds = GeoDataset(csv_path, image_size=image_size, train=train)
    if max_samples is None or max_samples >= len(ds):
        return ds

    import random

    random.seed(seed)
    # pandas sample is deterministic with random_state
    indices = ds.df.sample(n=max_samples, random_state=seed).index.tolist()
    # Return Subset that preserves GeoDataset.__getitem__
    # Subset will delegate transform etc. correctly.
    return Subset(ds, indices)


def get_stratified_sampler(
    dataset: GeoDataset,
    cell_ids: Optional[List[int]] = None,
) -> WeightedRandomSampler | None:
    """Stratified sampler stub — Section 8 mitigation (stratified by country).

    TODO L1/L2: implement true stratification by country or by cell.
    For now returns None (uniform sampling) so training runs without error.

    When cell_ids are available, compute per-class weights = 1 / count
    and return WeightedRandomSampler. Placeholder keeps import working.

    Args:
        dataset: GeoDataset (or Subset wrapping one)
        cell_ids: optional cell assignments aligned to dataset order

    Returns:
        WeightedRandomSampler or None (fallback to shuffle=True).
    """
    # TODO: L1 — compute weights from cell distribution or country metadata
    # Example (when ready):
    #   from collections import Counter
    #   counts = Counter(cell_ids)
    #   weights = [1.0 / counts[c] for c in cell_ids]
    #   return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return None


def build_dataloader(
    csv_path: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    train: bool = True,
    num_workers: int = 0,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """Convenience: CSV → DataLoader.

    Windows-safe: default num_workers=0 (avoids spawn issues on Windows).
    Set num_workers>0 only if needed; pin_memory=True only when CUDA available.
    """
    ds = load_subset(csv_path, max_samples=max_samples, image_size=image_size, train=train)
    sampler = None
    # Only attempt stratified sampler when underlying dataset is GeoDataset
    # (Subset case handled inside sampler stub as None)
    # Shuffle iff no sampler
    shuffle = sampler is None and train

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
