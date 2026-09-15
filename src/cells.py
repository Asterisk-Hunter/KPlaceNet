"""Cells — quad-tree, k-means, and DBSCAN coarse cell builders + assign.

L0: quad-tree stub. L1: quad-tree 200-500 from 10k subset.
L2: JSON serialization for fixed-cell fairness.
L4: k-means and DBSCAN adaptive cell construction (Gap 1).

All builders produce List[Cell] with contiguous cell_ids 0..K-1.
assign_cells uses containment (quad-tree boxes) or nearest-centroid
(adaptive / overlapping clusters).  JSON-compatible for fixed-cell fairness.

Constraints: K max ~300 for 4050 (Section 8).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


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
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_np(coords: np.ndarray | torch.Tensor) -> np.ndarray:
    """Coerce to float64 numpy array of shape (N, 2)."""
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be Nx2, got {coords.shape}")
    return coords


def _haversine_vec(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in km between N pairs."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Quad-tree builder (L1, unchanged)
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
        - A split is kept only if it yields >= 2 non-empty children.
        - Stop at K or when no leaf is splittable.
        - Centroid = mean of assigned points, box-center fallback if empty.

    Args:
        coords: Nx2 array of (lat, lon), dtype float.
        K: target number of cells (200-500, ~300 default).
        max_per_cell: optional cap — unused (kept for API compat).

    Returns:
        List[Cell] of length min(K, n) at most; fewer if points are
        inseparable (e.g. all duplicates). Every input point falls in
        exactly one returned cell.
    """
    coords = _to_np(coords)
    n = len(coords)
    if n == 0:
        return []

    K = min(int(K), n)
    if K <= 0:
        return []
    if K == 1 or n == 1:
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
        quads: List[List[int]] = [[], [], [], []]
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
            (leaf["lat_min"], mid_lat, leaf["lon_min"], mid_lon),
            (leaf["lat_min"], mid_lat, mid_lon, leaf["lon_max"]),
            (mid_lat, leaf["lat_max"], leaf["lon_min"], mid_lon),
            (mid_lat, leaf["lat_max"], mid_lon, leaf["lon_max"]),
        ]
        children = []
        for q, (la0, la1, lo0, lo1) in zip(quads, bounds):
            if not q:
                continue
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

    while len(leaves) < K:
        room = K - len(leaves)
        order = sorted(range(len(leaves)), key=lambda li: (-len(leaves[li]["indices"]), li))
        best = -1
        best_children: List[dict] = []
        for li in order:
            if len(leaves[li]["indices"]) < 2:
                continue
            children = _try_split(leaves[li])
            if len(children) < 2:
                continue
            if len(children) - 1 <= room:
                best = li
                best_children = children
                break
        if best < 0:
            break
        leaf = leaves.pop(best)
        for offset, child in enumerate(best_children):
            leaves.insert(best + offset, child)

    cells: List[Cell] = []
    for cid, leaf in enumerate(leaves):
        idxs = leaf["indices"]
        if idxs:
            centroid_lat = float(coords[idxs, 0].mean())
            centroid_lon = float(coords[idxs, 1].mean())
        else:
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


# ---------------------------------------------------------------------------
# K-means builder (L4, Gap 1)
# ---------------------------------------------------------------------------


def build_kmeans_cells(
    coords: np.ndarray | torch.Tensor,
    K: int = 300,
    seed: int = 42,
    n_init: int = 10,
    max_iter: int = 300,
) -> List[Cell]:
    """Build cells via k-means clustering on (lat, lon).

    Uses sklearn MiniBatchKMeans for speed on 10k points.  Deterministic
    via seed=42.  Produces exactly K non-empty clusters when K <= N by
    splitting the largest cluster to fill any empty slots after fitting.

    Latitude is scaled by cos(mean_lat) to approximate equal-area projection
    before clustering, then centroids are un-projected back to (lat, lon).

    Args:
        coords: Nx2 (lat, lon).
        K: target number of cells.
        seed: RNG seed for reproducibility.
        n_init: number of k-means restarts (sklearn default 10).
        max_iter: max iterations per run.

    Returns:
        List[Cell] with cell_ids 0..K-1, all with count > 0.
    """
    coords = _to_np(coords)
    n = len(coords)
    if n == 0:
        return []
    K = min(int(K), n)
    if K <= 0:
        return []

    # Approximate equal-area: scale lon by cos(mean_lat)
    mean_lat_rad = np.radians(float(coords[:, 0].mean()))
    cos_lat = max(np.cos(mean_lat_rad), 1e-6)
    coords_proj = coords.copy()
    coords_proj[:, 1] *= cos_lat

    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=min(1024, n),
    )
    labels = km.fit_predict(coords_proj)

    # Guarantee all K clusters are non-empty.
    # MiniBatchKMeans can produce empty clusters; fix by iteratively
    # splitting the largest cluster: steal its farthest point and
    # assign it to the empty slot.
    empty_ids = [cid for cid in range(K) if (labels == cid).sum() == 0]
    for empty_id in empty_ids:
        # Find the largest cluster
        unique, counts = np.unique(labels, return_counts=True)
        largest_id = int(unique[np.argmax(counts)])
        largest_mask = labels == largest_id
        largest_pts = coords_proj[largest_mask]
        if len(largest_pts) < 2:
            continue  # cannot split a singleton
        # Centroid of largest cluster in projected space
        centroid = largest_pts.mean(axis=0)
        # Farthest point from centroid → assign to empty slot
        dists = np.linalg.norm(largest_pts - centroid, axis=1)
        farthest_local = int(np.argmax(dists))
        farthest_global = int(np.where(largest_mask)[0][farthest_local])
        labels[farthest_global] = empty_id

    # Un-project centroids from final assignments
    cells: List[Cell] = []
    for cid in range(K):
        mask = labels == cid
        pts = coords[mask]
        count = int(mask.sum())
        if count > 0:
            centroid_lat = float(pts[:, 0].mean())
            centroid_lon = float(pts[:, 1].mean())
            lat_min, lat_max = float(pts[:, 0].min()), float(pts[:, 0].max())
            lon_min, lon_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        else:
            # Should be unreachable after the fix above, but defensive
            centroid_lat = float(coords[:, 0].mean())
            centroid_lon = float(coords[:, 1].mean())
            lat_min = lat_max = centroid_lat
            lon_min = lon_max = centroid_lon
        cells.append(
            Cell(
                cell_id=cid,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                count=count,
            )
        )
    return cells


# ---------------------------------------------------------------------------
# DBSCAN builder (L4, Gap 1)
# ---------------------------------------------------------------------------


def _dbscan_eps_for_k(
    coords_proj: np.ndarray,
    K: int,
    min_samples: int = 5,
    lo: float = 0.001,
    hi: float = 10.0,
    iters: int = 60,
) -> float:
    """Binary-search for eps that yields approximately K clusters.

    DBSCAN's cluster count depends on eps.  Smaller eps → more
    fragmentation → more clusters.  Larger eps → more merging → fewer
    clusters.  We binary-search for the eps closest to K; ties broken
    by smaller eps (tighter clusters).

    Noise points (label -1) are NOT counted as clusters.
    """
    from sklearn.cluster import DBSCAN

    best_eps = lo
    best_diff = float("inf")

    for _ in range(iters):
        mid = (lo + hi) / 2.0
        db = DBSCAN(eps=mid, min_samples=min_samples, algorithm="ball_tree")
        db.fit(coords_proj)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
        diff = abs(n_clusters - K)
        if diff < best_diff or (diff == best_diff and mid < best_eps):
            best_diff = diff
            best_eps = mid
        if n_clusters > K:
            # too many clusters → increase eps to merge more
            lo = mid
        else:
            # too few or exact → try smaller eps to fragment more
            hi = mid
        if hi - lo < 1e-7:
            break

    return best_eps


def build_dbscan_cells(
    coords: np.ndarray | torch.Tensor,
    K: int = 300,
    seed: int = 42,
    min_samples: int = 5,
) -> List[Cell]:
    """Build cells via DBSCAN density-based clustering on (lat, lon).

    Algorithm:
        1. Scale lon by cos(mean_lat) for approximate equal-area.
        2. Binary-search for eps that yields approximately K non-noise clusters.
        3. If noise points exist (label == -1), assign each to the nearest
           non-noise cluster centroid (haversine in lat/lon space).
        4. After reassignment, if total clusters > K, merge smallest into
           nearest neighbor until exactly K.  If < K, split largest.
        5. Re-contiguify cell_ids to 0..K-1.

    Edge cases:
        - If K >= N unique points, each point gets its own cell.
        - If all points collapse to 1 cluster at any eps, we still produce K
          cells by iteratively splitting the largest cluster at its centroid
          midpoint along the longest axis.
        - Deterministic via seed=42 (DBSCAN is deterministic for ball_tree).

    Args:
        coords: Nx2 (lat, lon).
        K: target number of cells.
        seed: RNG seed (used for tie-breaking).
        min_samples: DBSCAN min_samples parameter.

    Returns:
        List[Cell] with contiguous cell_ids 0..K-1.
    """
    coords = _to_np(coords)
    n = len(coords)
    if n == 0:
        return []
    K = min(int(K), n)
    if K <= 0:
        return []

    mean_lat_rad = np.radians(float(coords[:, 0].mean()))
    cos_lat = max(np.cos(mean_lat_rad), 1e-6)
    coords_proj = coords.copy()
    coords_proj[:, 1] *= cos_lat

    # Step 1: Find eps that yields ~K clusters
    eps = _dbscan_eps_for_k(coords_proj, K, min_samples=min_samples)

    from sklearn.cluster import DBSCAN

    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit_predict(coords_proj)

    # Step 2: Reassign noise points to nearest non-noise centroid
    unique_labels = sorted(set(labels))
    non_noise_labels = [l for l in unique_labels if l != -1]

    if len(non_noise_labels) > 0:
        # Compute centroids of non-noise clusters (in original lat/lon)
        centroids_dict: Dict[int, np.ndarray] = {}
        for lab in non_noise_labels:
            mask = labels == lab
            centroids_dict[lab] = coords[mask].mean(axis=0)

        # Reassign noise
        noise_mask = labels == -1
        noise_idx = np.where(noise_mask)[0]
        if len(noise_idx) > 0 and len(centroids_dict) > 0:
            cent_labels = list(centroids_dict.keys())
            cent_arr = np.array([centroids_dict[l] for l in cent_labels])
            for idx in noise_idx:
                lat, lon = coords[idx]
                dists = _haversine_vec(
                    np.array([lat]), np.array([lon]),
                    cent_arr[:, 0], cent_arr[:, 1],
                )
                nearest = cent_labels[int(np.argmin(dists))]
                labels[idx] = nearest

    # Step 3: Recompute cluster set after noise reassignment
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    # Step 4: Merge or split to hit exactly K
    # Helper: compute centroids and counts for current labels
    def _cluster_stats(labs: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        cents: Dict[int, np.ndarray] = {}
        counts: Dict[int, int] = {}
        for lab in sorted(set(labs)):
            mask = labs == lab
            cents[lab] = coords[mask].mean(axis=0)
            counts[lab] = int(mask.sum())
        return cents, counts

    # Merge largest clusters if too many
    while n_clusters > K:
        cents, counts = _cluster_stats(labels)
        lab_list = sorted(cents.keys())
        # Find the smallest cluster and merge into nearest
        smallest_lab = min(lab_list, key=lambda l: counts[l])
        # Find nearest other centroid
        other_labs = [l for l in lab_list if l != smallest_lab]
        if not other_labs:
            break
        other_cents = np.array([cents[l] for l in other_labs])
        d = _haversine_vec(
            np.array([cents[smallest_lab][0]]),
            np.array([cents[smallest_lab][1]]),
            other_cents[:, 0], other_cents[:, 1],
        )
        nearest_lab = other_labs[int(np.argmin(d))]
        labels[labels == smallest_lab] = nearest_lab
        n_clusters -= 1

    # Split largest clusters if too few
    while n_clusters < K:
        cents, counts = _cluster_stats(labels)
        lab_list = sorted(cents.keys())
        if not lab_list:
            break
        largest_lab = max(lab_list, key=lambda l: counts[l])
        mask = labels == largest_lab
        pts = coords[mask]
        if len(pts) < 2:
            break  # can't split a singleton

        # Split along longest axis at centroid
        axis = 0 if (pts[:, 0].max() - pts[:, 0].min()) >= (pts[:, 1].max() - pts[:, 1].min()) else 1
        c = float(cents[largest_lab][axis])
        lo_mask = pts[:, axis] <= c
        hi_mask = ~lo_mask

        if lo_mask.sum() == 0 or hi_mask.sum() == 0:
            # Degenerate — split at median instead
            med = float(np.median(pts[:, axis]))
            lo_mask = pts[:, axis] <= med
            hi_mask = ~lo_mask
            if lo_mask.sum() == 0 or hi_mask.sum() == 0:
                break  # truly degenerate, give up

        # Assign split halves: keep largest_lab for one, new label for other
        new_lab = max(labels) + 1
        pt_indices = np.where(mask)[0]
        labels[pt_indices[hi_mask]] = new_lab
        n_clusters += 1

    # Step 5: Re-contiguify labels to 0..K-1
    unique_labels = sorted(set(labels))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    # Step 6: Build cells
    cents, counts = _cluster_stats(labels)
    cells: List[Cell] = []
    for cid in range(K):
        mask = labels == cid
        pts = coords[mask]
        count = int(mask.sum())
        centroid_lat = float(pts[:, 0].mean()) if count > 0 else 0.0
        centroid_lon = float(pts[:, 1].mean()) if count > 0 else 0.0
        if count > 0:
            lat_min, lat_max = float(pts[:, 0].min()), float(pts[:, 0].max())
            lon_min, lon_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        else:
            lat_min = lat_max = centroid_lat
            lon_min = lon_max = centroid_lon
        cells.append(
            Cell(
                cell_id=cid,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                centroid_lat=centroid_lat,
                centroid_lon=centroid_lon,
                count=count,
            )
        )
    return cells


# ---------------------------------------------------------------------------
# Unified builder — dispatches by method
# ---------------------------------------------------------------------------


def build_cells(
    coords: np.ndarray | torch.Tensor,
    K: int = 300,
    method: str = "quad_tree",
    **kwargs: Any,
) -> List[Cell]:
    """Unified builder — dispatches by method.

    Args:
        coords: Nx2 (lat, lon)
        K: target cells
        method: "quad_tree" | "kmeans" | "dbscan"
        **kwargs: forwarded to the specific builder (e.g. seed, n_init).

    Returns:
        List[Cell] with contiguous cell_ids 0..len(cells)-1.
    """
    if method == "quad_tree":
        return build_quad_tree_cells(coords, K=K)
    elif method == "kmeans":
        return build_kmeans_cells(coords, K=K, **kwargs)
    elif method == "dbscan":
        return build_dbscan_cells(coords, K=K, **kwargs)
    else:
        raise ValueError(
            f"Unknown cell method: {method}. Expected quad_tree/kmeans/dbscan."
        )


# ---------------------------------------------------------------------------
# Cell assignment — containment + nearest-centroid fallback
# ---------------------------------------------------------------------------


def assign_cells(
    coords: np.ndarray | torch.Tensor,
    cells: List[Cell],
) -> List[int]:
    """Assign each (lat,lon) to a cell.

    Strategy (handles both quad-tree boxes and adaptive/overlapping clusters):
        1. Collect all cells whose bounding box contains the point.
        2. Among containing cells, pick the one whose centroid is nearest
           (haversine).  This resolves ambiguity when clusters overlap.
        3. If no cell contains the point, pick the nearest centroid overall.

    Args:
        coords: Nx2 (lat, lon)
        cells: list from build_cells

    Returns:
        List[int] of length N with cell_id per point.

    Raises:
        ValueError: if cells is empty.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()
    coords = np.asarray(coords, dtype=np.float64)

    if not cells:
        raise ValueError("No cells provided to assign_cells")

    centroids = np.array(
        [[c.centroid_lat, c.centroid_lon] for c in cells], dtype=np.float64
    )
    # Precompute bounding boxes as arrays for vectorised containment
    lat_mins = np.array([c.lat_min for c in cells])
    lat_maxs = np.array([c.lat_max for c in cells])
    lon_mins = np.array([c.lon_min for c in cells])
    lon_maxs = np.array([c.lon_max for c in cells])

    assignments: List[int] = []
    for i in range(len(coords)):
        lat = float(coords[i, 0])
        lon = float(coords[i, 1])

        # 1. Find all cells whose box contains this point
        contains = (
            (lat_mins <= lat) & (lat <= lat_maxs)
            & (lon_mins <= lon) & (lon <= lon_maxs)
        )
        containing_ids = np.where(contains)[0]

        if len(containing_ids) > 0:
            # 2. Among containing cells, pick nearest centroid (haversine)
            c_lats = centroids[containing_ids, 0]
            c_lons = centroids[containing_ids, 1]
            dists = _haversine_vec(
                np.array([lat]), np.array([lon]),
                c_lats, c_lons,
            )
            assignments.append(int(cells[containing_ids[int(np.argmin(dists))]].cell_id))
        else:
            # 3. Nearest centroid overall
            dists = _haversine_vec(
                np.array([lat]), np.array([lon]),
                centroids[:, 0], centroids[:, 1],
            )
            assignments.append(int(cells[int(np.argmin(dists))].cell_id))

    return assignments


# ---------------------------------------------------------------------------
# Centroid extraction
# ---------------------------------------------------------------------------


def cells_to_centroids(cells: List[Cell]) -> np.ndarray:
    """Return Kx2 array of (lat, lon) centroids for model output mapping."""
    return np.array(
        [[c.centroid_lat, c.centroid_lon] for c in cells], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Cell balance metrics
# ---------------------------------------------------------------------------


def cell_balance_stats(cells: List[Cell]) -> Dict[str, float]:
    """Compute cell balance statistics from cell counts.

    Returns dict with mean, std, min, max, median, gini of per-cell counts.
    """
    counts = np.array([c.count for c in cells], dtype=np.float64)
    if len(counts) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0, "gini": 0}
    mean_val = float(counts.mean())
    std_val = float(counts.std())
    min_val = float(counts.min())
    max_val = float(counts.max())
    median_val = float(np.median(counts))
    # Gini coefficient
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    gini = float((2 * (index * sorted_counts).sum() / (n * sorted_counts.sum())) - (n + 1) / n) if sorted_counts.sum() > 0 else 0.0
    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "gini": gini,
    }


# ---------------------------------------------------------------------------
# L2: JSON serialization / deserialization for fixed-cell fairness
# ---------------------------------------------------------------------------


def save_cells_json(cells: List[Cell], path: str | Path) -> Path:
    """Serialize cells to a JSON file for reproducible fixed-cell experiments."""
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
    """Deserialize cells from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cells JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cells_raw = data.get("cells")
    if cells_raw is None or not isinstance(cells_raw, list):
        raise ValueError(f"Invalid cells JSON at {path}: missing or non-list 'cells' key")
    cells: List[Cell] = []
    for c in cells_raw:
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
