"""L4 Adaptive Cell Construction experiments — compare cell methods.

Matrix: 3 cell methods (quad_tree, kmeans, dbscan) x same K.
Keeps L2 winner backbone/data regime (ImageNet init, layer4 fine-tune,
100% data, 10 epochs).

For each method:
    1. Builds cells from the full training CSV and saves to JSON.
    2. Runs train.py with --cells-json for that method.
    3. Evaluates each checkpoint on the official OSV test CSV.
    4. Reports within-km, cell balance, regional breakdown.

Default: plan-only (prints commands, no execution). Use --run to execute.

Windows-safe: pathlib, sys.executable subprocess, no shell strings.

Usage:
    python scripts/run_l4_experiments.py
    python scripts/run_l4_experiments.py --run
    python scripts/run_l4_experiments.py --run --methods kmeans --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = ROOT / "data" / "osv5m_subset_10k" / "metadata.csv"
DEFAULT_TEST_CSV = ROOT / "data" / "osv5m_test" / "metadata.csv"
CHECKPOINTS_DIR = ROOT / "checkpoints"
METHODS = ["quad_tree", "kmeans", "dbscan"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L4 Adaptive Cell Construction experiments (plan-only by default)"
    )
    p.add_argument(
        "--run", action="store_true",
        help="Execute the runs (default is plan-only)",
    )
    p.add_argument(
        "--methods", nargs="+", default=METHODS,
        choices=["quad_tree", "kmeans", "dbscan"],
        help=f"Cell methods to compare (default: {METHODS})",
    )
    p.add_argument("--K", type=int, default=300,
                   help="Target number of cells (default: 300)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs per method (default: 10)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Training batch size (default: 16)")
    p.add_argument("--image-size", type=int, default=224,
                   help="Image size (default: 224)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default: 42)")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto|cuda|cpu (default: auto)")
    p.add_argument("--train-csv", type=str, default=str(DEFAULT_TRAIN_CSV),
                   help=f"Training CSV (default: {DEFAULT_TRAIN_CSV})")
    p.add_argument("--test-csv", type=str, default=str(DEFAULT_TEST_CSV),
                   help=f"Test CSV for evaluation (default: {DEFAULT_TEST_CSV})")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip methods whose checkpoint dir already has best.pt")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, only evaluate existing checkpoints")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_csv_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def cells_json_path(method: str) -> Path:
    return CHECKPOINTS_DIR / f"l4_cells_{method}.json"


def ckpt_dir_name(method: str) -> str:
    return f"l4_{method}"


def make_train_cmd(args: argparse.Namespace, method: str) -> list[str]:
    """Build the train.py command as a list (no shell string)."""
    ckpt_dir = CHECKPOINTS_DIR / ckpt_dir_name(method)
    cells_json = cells_json_path(method)

    return [
        sys.executable, "-m", "src.train",
        "--csv", str(args.train_csv),
        "--num-cells", str(args.K),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--checkpoint-dir", str(ckpt_dir),
        "--cells-json", str(cells_json),
        "--run-tag", f"l4_{method}",
        # L2 winner regime: ImageNet init, layer4 fine-tune
        "--pretrained",
        "--unfreeze-last-block",
    ]


def make_eval_cmd(args: argparse.Namespace, method: str) -> list[str]:
    """Build the eval.py command."""
    ckpt_dir = CHECKPOINTS_DIR / ckpt_dir_name(method)
    ckpt_file = ckpt_dir / "best.pt"
    return [
        sys.executable, "-m", "src.eval",
        "--csv", str(args.test_csv),
        "--checkpoint", str(ckpt_file),
        "--num-cells", str(args.K),
        "--batch-size", str(args.batch_size),
        "--device", str(args.device),
    ]


def build_cells_from_csv(
    csv_path: Path, method: str, K: int, seed: int
) -> None:
    """Build cells from full CSV and save to JSON."""
    jp = cells_json_path(method)
    if jp.exists():
        print(f"  cells JSON exists: {jp}")
        return

    sys.path.insert(0, str(ROOT))
    from src.cells import build_cells, save_cells_json
    from src.dataset import GeoDataset

    ds = GeoDataset(str(csv_path), image_size=224, train=True)
    coords = ds.get_coords()
    coords_np = coords.cpu().numpy() if isinstance(coords, np.ndarray) else np.asarray(coords)

    print(f"  building {method} cells (K={K}) from {csv_path} ...")
    cells = build_cells(coords_np, K=K, method=method, seed=seed)
    save_cells_json(cells, jp)
    print(f"  effective cells: {len(cells)}")


def compute_cell_balance(cells_json: Path) -> Dict[str, Any]:
    """Compute cell balance stats from a cells JSON."""
    sys.path.insert(0, str(ROOT))
    from src.cells import load_cells_json, cell_balance_stats

    cells = load_cells_json(cells_json)
    stats = cell_balance_stats(cells)
    return stats


def compute_regional_breakdown(
    cells_json: Path,
    train_csv: Path,
) -> Dict[str, Any]:
    """Compute regional breakdown: urban vs rural proxy.

    Urban/rural proxy: cells with count > median cell count are "urban"
    (dense training data), cells with count <= median are "rural" (sparse).
    This is a transparent, reproducible proxy since no urban/rural labels
    exist in OSV-5M.  We report this caveat explicitly.
    """
    sys.path.insert(0, str(ROOT))
    from src.cells import load_cells_json, assign_cells
    from src.dataset import GeoDataset

    cells = load_cells_json(cells_json)
    ds = GeoDataset(str(train_csv), image_size=224, train=True)
    coords = ds.get_coords()
    coords_np = coords.cpu().numpy() if isinstance(coords, np.ndarray) else np.asarray(coords)
    assignments = assign_cells(coords_np, cells)

    counts = Counter(assignments)
    cell_counts = np.array([c.count for c in cells])
    median_count = float(np.median(cell_counts)) if len(cell_counts) > 0 else 0

    urban_cells = [c.cell_id for c in cells if c.count > median_count]
    rural_cells = [c.cell_id for c in cells if c.count <= median_count]
    urban_set = set(urban_cells)
    rural_set = set(rural_cells)

    urban_pts = sum(1 for a in assignments if a in urban_set)
    rural_pts = sum(1 for a in assignments if a in rural_set)

    return {
        "proxy": "count > median => urban, count <= median => rural",
        "median_cell_count": median_count,
        "n_urban_cells": len(urban_cells),
        "n_rural_cells": len(rural_cells),
        "urban_points": urban_pts,
        "rural_points": rural_pts,
        "urban_fraction_points": urban_pts / max(1, urban_pts + rural_pts),
    }


# ---------------------------------------------------------------------------
# Haversine helpers for distance metrics
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a)))


def _haversine_km_vec(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _within_km_metrics(
    pred_lats: np.ndarray, pred_lons: np.ndarray,
    true_lats: np.ndarray, true_lons: np.ndarray,
) -> Dict[str, float]:
    """Compute within-1/25/200km + mean/median distance."""
    dists = _haversine_km_vec(pred_lats, pred_lons, true_lats, true_lons)
    return {
        "within_1km": float((dists <= 1).mean() * 100),
        "within_25km": float((dists <= 25).mean() * 100),
        "within_200km": float((dists <= 200).mean() * 100),
        "mean_km": float(dists.mean()),
        "median_km": float(np.median(dists)),
    }


# ---------------------------------------------------------------------------
# Internal eval (avoids subprocess for the eval step)
# ---------------------------------------------------------------------------


@torch.no_grad()
def internal_eval(
    args: argparse.Namespace,
    method: str,
) -> Optional[Dict[str, Any]]:
    """Evaluate an existing checkpoint using internal logic.

    Returns metrics dict or None if checkpoint not found.
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    sys.path.insert(0, str(ROOT))
    from src.cells import load_cells_json
    from src.dataset import GeoDataset
    from src.model import build_model

    ckpt_dir = CHECKPOINTS_DIR / ckpt_dir_name(method)
    ckpt_file = ckpt_dir / "best.pt"
    if not ckpt_file.exists():
        print(f"  [warn] checkpoint not found: {ckpt_file}")
        return None

    device_str = args.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    cells_data = ckpt.get("cells", [])
    num_cells = ckpt.get("num_cells", ckpt.get("args", {}).get("num_cells", len(cells_data)))

    model = build_model(num_cells=num_cells, pretrained=False, freeze_backbone=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()

    # Centroids from checkpoint
    centroids = np.array(
        [[c["centroid_lat"], c["centroid_lon"]] for c in sorted(cells_data, key=lambda x: x["cell_id"])],
        dtype=np.float64,
    )

    ds = GeoDataset(str(args.test_csv), image_size=args.image_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_pred_lats: list[float] = []
    all_pred_lons: list[float] = []
    all_true_lats: list[float] = []
    all_true_lons: list[float] = []

    for images, true_coords in tqdm(loader, desc=f"  eval {method}", unit="batch"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred_ids = logits.argmax(dim=1).cpu().numpy()
        if pred_ids.max(initial=0) >= len(centroids):
            pred_ids = np.clip(pred_ids, 0, len(centroids) - 1)

        pred_lats = centroids[pred_ids, 0]
        pred_lons = centroids[pred_ids, 1]

        if isinstance(true_coords, torch.Tensor):
            true_np = true_coords.cpu().numpy()
        else:
            true_np = np.asarray(true_coords)
        if true_np.ndim == 1:
            true_np = true_np[None, :]

        all_pred_lats.extend(pred_lats.tolist())
        all_pred_lons.extend(pred_lons.tolist())
        all_true_lats.extend(true_np[:, 0].tolist())
        all_true_lons.extend(true_np[:, 1].tolist())

    metrics = _within_km_metrics(
        np.array(all_pred_lats), np.array(all_pred_lons),
        np.array(all_true_lats), np.array(all_true_lons),
    )
    metrics["n_eval"] = len(all_pred_lats)
    return metrics


# ---------------------------------------------------------------------------
# Plan display
# ---------------------------------------------------------------------------


def print_plan(args: argparse.Namespace) -> None:
    """Print the L4 experiment plan without loading anything."""
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)

    print("=" * 72)
    print("L4 ADAPTIVE CELL CONSTRUCTION — EXPERIMENT PLAN")
    print("=" * 72)
    print()
    print(f"  Training CSV:  {train_csv}")
    print(f"  Test CSV:      {test_csv}")
    print(f"  Methods:       {args.methods}")
    print(f"  K (cells):     {args.K}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Image size:    {args.image_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Device:        {args.device}")
    print(f"  Seed:          {args.seed}")
    print()
    print("  For each method:")
    print("    1. Build cells from training CSV -> checkpoints/l4_cells_<method>.json")
    print("    2. Train: python -m src.train with --cells-json, --pretrained,")
    print("       --unfreeze-last-block, full data, 10 epochs")
    print("    3. Evaluate: python -m src.eval on OSV test CSV")
    print("    4. Report: within-km, cell balance, regional breakdown")
    print()
    print("  Urban/rural proxy: cells with count > median => 'urban',")
    print("  count <= median => 'rural'. No urban/rural labels in OSV-5M.")
    print()

    # Build commands
    print("  Commands:")
    for method in args.methods:
        jp = cells_json_path(method)
        ckpt_dir = CHECKPOINTS_DIR / ckpt_dir_name(method)
        print(f"\n  --- {method} ---")
        print(f"  Cells JSON: {jp}")
        print(f"  Checkpoint: {ckpt_dir}")
        print(f"  Train cmd: {' '.join(make_train_cmd(args, method))}")
        print(f"  Eval cmd:  {' '.join(make_eval_cmd(args, method))}")

    print()
    print("=" * 72)
    print("PLAN COMPLETE — no runs executed.")
    print("Re-run with --run to execute. Add --skip-existing to resume.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    run = args.run

    if not run:
        print_plan(args)
        return

    print("=" * 72)
    print("L4 ADAPTIVE CELL CONSTRUCTION — EXECUTING")
    print("=" * 72)

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    if not train_csv.exists():
        print(f"[error] Training CSV not found: {train_csv}")
        sys.exit(1)
    if not test_csv.exists():
        print(f"[error] Test CSV not found: {test_csv}")
        sys.exit(1)

    num_train_rows = count_csv_rows(train_csv)
    print(f"  Training rows: {num_train_rows}")
    print(f"  Test CSV: {test_csv}")
    print(f"  Methods: {args.methods}")
    print(f"  K={args.K}, epochs={args.epochs}, batch={args.batch_size}")
    print()

    # --- Step 1: Build cells for each method ---
    print("[1/3] Cell construction")
    for method in args.methods:
        print(f"  {method}:")
        build_cells_from_csv(train_csv, method, args.K, args.seed)
    print()

    # --- Step 2: Train or skip ---
    print("[2/3] Training")
    results: list[dict] = []
    for method in args.methods:
        ckpt_dir = CHECKPOINTS_DIR / ckpt_dir_name(method)
        best_pt = ckpt_dir / "best.pt"
        entry: Dict[str, Any] = {
            "method": method,
            "checkpoint_dir": str(ckpt_dir),
            "cells_json": str(cells_json_path(method)),
        }

        if args.skip_existing and best_pt.exists():
            print(f"  SKIP (exists): {method}")
            entry["skip"] = True
            results.append(entry)
            continue
        if args.eval_only:
            print(f"  EVAL-ONLY: {method}")
            entry["skip"] = True
            results.append(entry)
            continue

        cmd = make_train_cmd(args, method)
        entry["train_cmd"] = cmd
        print(f"  TRAINING: {method}")
        print(f"    cmd: {' '.join(cmd)}")
        t0 = time.time()
        try:
            result = subprocess.run(cmd, cwd=str(ROOT), check=False, timeout=3600 * 4)
            elapsed = time.time() - t0
            entry["train_returncode"] = result.returncode
            entry["train_elapsed_sec"] = elapsed
            if result.returncode != 0:
                print(f"    FAILED (returncode={result.returncode}, {elapsed:.0f}s)")
            else:
                print(f"    OK ({elapsed:.0f}s)")
        except subprocess.TimeoutExpired:
            entry["train_returncode"] = -1
            entry["train_elapsed_sec"] = 3600 * 4
            print("    TIMEOUT (4h)")
        except Exception as e:
            entry["train_returncode"] = -2
            entry["train_elapsed_sec"] = time.time() - t0
            print(f"    ERROR: {e}")
        results.append(entry)

    print()

    # --- Step 3: Evaluate ---
    print("[3/3] Evaluation on test set")
    all_eval_results: Dict[str, Any] = {}
    for entry in results:
        method = entry["method"]
        print(f"\n  --- {method} ---")

        # Cell balance
        jp = Path(entry["cells_json"])
        if jp.exists():
            balance = compute_cell_balance(jp)
            entry["cell_balance"] = balance
            print(f"  Cell balance: mean={balance['mean']:.1f}, std={balance['std']:.1f}, "
                  f"min={balance['min']:.0f}, max={balance['max']:.0f}, "
                  f"median={balance['median']:.1f}, gini={balance['gini']:.3f}")

            # Regional breakdown
            regional = compute_regional_breakdown(jp, train_csv)
            entry["regional"] = regional
            print(f"  Urban/rural ({regional['proxy']}):")
            print(f"    urban cells={regional['n_urban_cells']}, rural cells={regional['n_rural_cells']}")
            print(f"    urban points={regional['urban_points']}, rural points={regional['rural_points']} "
                  f"({regional['urban_fraction_points']:.1%} urban)")

        # Eval metrics
        if entry.get("skip") and not Path(entry["checkpoint_dir"], "best.pt").exists():
            print("  No checkpoint — skipping eval")
            continue

        try:
            eval_metrics = internal_eval(args, method)
        except Exception as e:
            print(f"  Eval error: {e}")
            eval_metrics = None

        if eval_metrics is not None:
            entry["eval"] = eval_metrics
            all_eval_results[method] = eval_metrics
            print(f"  Eval: {eval_metrics['n_eval']} samples")
            print(f"    within  1km: {eval_metrics['within_1km']:.2f}%")
            print(f"    within 25km: {eval_metrics['within_25km']:.2f}%")
            print(f"    within 200km: {eval_metrics['within_200km']:.2f}%")
            print(f"    mean dist:   {eval_metrics['mean_km']:.1f} km")
            print(f"    median dist: {eval_metrics['median_km']:.1f} km")

    # --- Summary table ---
    print()
    print("=" * 72)
    print("L4 RESULTS SUMMARY")
    print("=" * 72)
    header = f"{'Method':<12} {'@1km':>8} {'@25km':>8} {'@200km':>8} {'mean_km':>10} {'med_km':>10}"
    print(header)
    print("-" * 72)
    for method in args.methods:
        if method in all_eval_results:
            m = all_eval_results[method]
            print(f"{method:<12} {m['within_1km']:>7.2f}% {m['within_25km']:>7.2f}% "
                  f"{m['within_200km']:>7.2f}% {m['mean_km']:>9.1f} {m['median_km']:>9.1f}")
        else:
            print(f"{method:<12} {'(no result)':>8} {'':>8} {'':>8} {'':>10} {'':>10}")
    print("=" * 72)

    # Determine default (best @200km)
    if all_eval_results:
        best_method = max(all_eval_results, key=lambda m: all_eval_results[m]["within_200km"])
        print(f"\nRecommended default cell method: {best_method}")
        print(f"  (highest within-200km: {all_eval_results[best_method]['within_200km']:.2f}%)")
    print()

    # Save manifest/results
    manifest = {
        "experiment": "L4_adaptive_cells",
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "K": args.K,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "methods": args.methods,
        "results": results,
    }
    manifest_path = CHECKPOINTS_DIR / "l4_experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    print(f"Manifest: {manifest_path}")


def _json_default(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
