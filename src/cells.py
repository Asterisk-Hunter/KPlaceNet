"""Cells — quad-tree coarse K~300 stub + assign function.

L0: stubs only (no heavy deps). L1 builds quad-tree 200-500 from 10k subset.
L4 swaps to k-means / DBSCAN same K. Imports kept light for scaffold.

L2: JSON serialization/deserialization for fixed-cell fairness.

Constraints: K max ~300 for 4050 (Section 8).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import numpy as np


@dataclass
class Cell:
    """Single geographic cell metadata."""

    cell_id: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    centroid_lat: float
    centroid_lon: float
    count: int = 0

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


# ---------------------------------------------------------------------------
# L0 stubs — quad-tree coarse builder
# ---------------------------------------------------------------------------


def build_quad_tree_cells(
    coords: np.ndarray | torch.Tensor,
    K: int = 300,
    max_per_cell: Optional[int] = None,
) -> List[Cell]:
    """Build density-driven quad-tree cells targeting K cells (L1).

    Deterministic adaptive partitioning:
        - Start with one cell covering world [-90,90]x[-180,180].
        - While len(cells) < K: split the leaf cell with the highest point
          count at midpoint lat/lon into up to 4 children, reassign points.
        - A split is kept only if it yields >= 2 non-empty children
          (avoids empty classes from duplicate coords / degenerate bounds).
        - Stop at K or when no leaf is splittable.
        - Centroid = mean of assigned points, box-center fallback if empty.

    Args:
        coords: Nx2 array of (lat, lon), dtype float.
        K: target number of cells (200-500, ~300 default per Sections 3,4,8).
        max_per_cell: optional cap — unused (kept for API compatibility).

    Returns:
        List[Cell] of length min(K, n) at most; fewer if points are
        inseparable (e.g. all duplicates). Every input point falls in
        exactly one returned cell.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be Nx2, got {coords.shape}")

    n = len(coords)
    if n == 0:
        return []

    K = min(int(K), n)  # cannot have more cells than points
    if K <= 0:
        return []
    if K == 1 or n == 1:
        lat_c, lon_c = float(coords[0, 0]), float(coords[0, 1])
        return [
            Cell(
                cell_id=0,
                lat_min=-90.0,
                lat_max=90.0,
                lon_min=-180.0,
                lon_max=180.0,
                centroid_lat=float(coords[:, 0].mean()),
                centroid_lon=float(coords[:, 1].mean()),
                count=n,
            )
        ]

    _MIN_SPAN = 1e-9

    # Leaf: dict(bounds + point indices). Indices reference rows of coords.
    leaves: List[dict] = [
        {
            "lat_min": -90.0,
            "lat_max": 90.0,
            "lon_min": -180.0,
            "lon_max": 180.0,
            "indices": list(range(n)),
        }
    ]

    def _split_indices(
        idxs: List[int], mid_lat: float, mid_lon: float
    ) -> List[List[int]]:
        """Deterministic quadrant assignment: lat<=mid → south, lon<=mid → west."""
        quads: List[List[int]] = [[], [], [], []]  # SW, SE, NW, NE
        for i in idxs:
            lat = float(coords[i, 0])
            lon = float(coords[i, 1])
            south = lat <= mid_lat
            west = lon <= mid_lon
            if south and west:
                quads[0].append(i)
            elif south and not west:
                quads[1].append(i)
            elif not south and west:
                quads[2].append(i)
            else:
                quads[3].append(i)
        return quads

    def _try_split(leaf: dict) -> List[dict]:
        """Split leaf into non-empty children; [] if unsplittable."""
        if len(leaf["indices"]) < 2:
            return []
        if (leaf["lat_max"] - leaf["lat_min"]) <= _MIN_SPAN:
            return []
        if (leaf["lon_max"] - leaf["lon_min"]) <= _MIN_SPAN:
            return []
        mid_lat = (leaf["lat_min"] + leaf["lat_max"]) / 2.0
        mid_lon = (leaf["lon_min"] + leaf["lon_max"]) / 2.0
        quads = _split_indices(leaf["indices"], mid_lat, mid_lon)
        bounds = [
            (leaf["lat_min"], mid_lat, leaf["lon_min"], mid_lon),  # SW
            (leaf["lat_min"], mid_lat, mid_lon, leaf["lon_max"]),  # SE
            (mid_lat, leaf["lat_max"], leaf["lon_min"], mid_lon),  # NW
            (mid_lat, leaf["lat_max"], mid_lon, leaf["lon_max"]),  # NE
        ]
        children = []
        for q, (la0, la1, lo0, lo1) in zip(quads, bounds):
            if not q:
                continue  # drop empty quadrants → no empty classes
            children.append(
                {
                    "lat_min": la0,
                    "lat_max": la1,
                    "lon_min": lo0,
                    "lon_max": lo1,
                    "indices": q,
                }
            )
        return children if len(children) >= 2 else []

    # Cache splits per leaf index each pass; rank by count desc, index asc.
    while len(leaves) < K:
        room = K - len(leaves)
        # Rank candidates densest-first (deterministic tie-break by position).
        order = sorted(range(len(leaves)), key=lambda li: (-len(leaves[li]["indices"]), li))
        best = -1
        best_children: List[dict] = []
        for li in order:
            if len(leaves[li]["indices"]) < 2:
                continue
            children = _try_split(leaves[li])
            if len(children) < 2:
                continue  # inseparable (duplicates) — try next densest
            if len(children) - 1 <= room:  # split must fit without overshoot
                best = li
                best_children = children
                break
        if best < 0:
            break  # nothing fits / nothing splittable
        leaf = leaves.pop(best)
        # _try_split already validated; recompute deterministically.
        for offset, child in enumerate(best_children):
            leaves.insert(best + offset, child)

    cells: List[Cell] = []
    for cid, leaf in enumerate(leaves):
        idxs = leaf["indices"]
        if idxs:
            centroid_lat = float(coords[idxs, 0].mean())
            centroid_lon = float(coords[idxs, 1].mean())
        else:  # fallback (should not happen — empty quadrants dropped)
            centroid_lat = (leaf["lat_min"] + leaf["lat_max"]) / 2.0
            centroid_lon = (leaf["lon_min"] + leaf["lon_max"]) / 2.0
        cells.append(
            Cell(
                cell_id=cid,
                lat_min=float(leaf["lat_min"]),
                lat_max=float(leaf["lat_max"]),
                lon_min=float(leaf["lon_min"]),
                lon_max=float(leaf["lon_max"]),
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                count=len(idxs),
            )
        )
    return cells


def build_cells(
    coords: np.ndarray | torch.Tensor,
    K: int = 300,
    method: str = "quad_tree",
) -> List[Cell]:
    """Unified builder — dispatches by method.

    Args:
        coords: Nx2 (lat, lon)
        K: target cells
        method: "quad_tree" (L1), "kmeans" | "dbscan" (L4)

    Raises:
        NotImplementedError for L4 methods in L0.
    """
    if method == "quad_tree":
        return build_quad_tree_cells(coords, K=K)
    elif method in ("kmeans", "dbscan"):
        # TODO L4: implement k-means / DBSCAN vs quad-tree same K ~300
        # Keep stub raising so caller knows not to use before L4.
        raise NotImplementedError(
            f"Cell method '{method}' is L4 (Gap 1) — not implemented in L0. "
            f"Use method='quad_tree' with K~300."
        )
    else:
        raise ValueError(f"Unknown cell method: {method}. Expected quad_tree/kmeans/dbscan.")


def assign_cells(
    coords: np.ndarray | torch.Tensor,
    cells: List[Cell],
) -> List[int]:
    """Assign each (lat,lon) to nearest cell.

    For box cells (stub), assigns by containing box; fallback to nearest
    centroid (haversine or Euclidean in lat/lon for L0 — haversine in L1).

    Args:
        coords: Nx2 (lat, lon)
        cells: list from build_cells

    Returns:
        List[int] of length N with cell_id per point.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    coords = np.asarray(coords, dtype=np.float64)

    if not cells:
        raise ValueError("No cells provided to assign_cells")

    # Build centroid array for nearest search
    centroids = np.array([[c.centroid_lat, c.centroid_lon] for c in cells], dtype=np.float64)

    assignments: List[int] = []
    for lat, lon in coords:
        # First try: containing box (fast for grid stub)
        found = None
        for cell in cells:
            if cell.contains(float(lat), float(lon)):
                found = cell.cell_id
                break
        if found is not None:
            assignments.append(found)
            continue

        # Fallback: nearest centroid (Euclidean in lat/lon; L1 upgrades to haversine)
        # Good enough for stub; real scoring uses haversine in eval.py
        dists = np.sqrt(((centroids[:, 0] - lat) ** 2) + ((centroids[:, 1] - lon) ** 2))
        assignments.append(int(np.argmin(dists)))

    return assignments


def cells_to_centroids(cells: List[Cell]) -> np.ndarray:
    """Return Kx2 array of (lat, lon) centroids for model output mapping."""
    return np.array([[c.centroid_lat, c.centroid_lon] for c in cells], dtype=np.float64)


# ---------------------------------------------------------------------------
# L2: JSON serialization / deserialization for fixed-cell fairness
# ---------------------------------------------------------------------------


def save_cells_json(cells: List[Cell], path: str | Path) -> Path:
    """Serialize cells to a JSON file for reproducible fixed-cell experiments.

    Args:
        cells: list of Cell dataclasses
        path: output JSON file path

    Returns:
        Resolved Path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "num_cells": len(cells),
        "cells": [asdict(c) for c in cells],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[cells] saved {len(cells)} cells -> {path}")
    return path.resolve()


def load_cells_json(path: str | Path) -> List[Cell]:
    """Deserialize cells from a JSON file.

    Args:
        path: JSON file previously written by save_cells_json.

    Returns:
        List[Cell] in the same order as saved.

    Raises:
        FileNotFoundError: if path does not exist.
        KeyError/ValueError: if JSON structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cells JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cells_raw = data.get("cells")
    if cells_raw is None or not isinstance(cells_raw, list):
        raise ValueError(f"Invalid cells JSON at {path}: missing or non-list 'cells' key")
    cells: List[Cell] = []
    for i, c in enumerate(cells_raw):
        cells.append(
            Cell(
                cell_id=int(c["cell_id"]),
                lat_min=float(c["lat_min"]),
                lat_max=float(c["lat_max"]),
                lon_min=float(c["lon_min"]),
                lon_max=float(c["lon_max"]),
                centroid_lat=float(c["centroid_lat"]),
                centroid_lon=float(c["centroid_lon"]),
                count=int(c.get("count", 0)),
            )
        )
    print(f"[cells] loaded {len(cells)} cells from {path}")
    return cells
