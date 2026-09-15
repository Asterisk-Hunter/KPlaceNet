"""L3 Uncertainty — temperature scaling, conformal prediction, abstention.

Loads the selected L2 winner checkpoint (l2_imagenet_1.0_layer4/last.pt),
splits the official OSV test set deterministically into calibration (first 1000)
and evaluation (remaining 2000), then reports:

  - Raw vs temperature-scaled ECE
  - Conformal prediction set coverage and mean set size
  - Abstention rate, accuracy-when-predicting, and distance metrics

Default mode is plan-only (no images loaded, no results written).  Use --run
to execute.

Windows-safe: pathlib, sys.executable, num_workers=0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cells import assign_cells, load_cells_json  # noqa: E402
from src.model import build_model  # noqa: E402
from src.uncertainty import (  # noqa: E402
    TemperatureScaler,
    abstention_mask,
    conformal_prediction_set,
    expected_calibration_error,
    fit_conformal_quantile,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "l2_imagenet_1.0_layer4" / "last.pt"
DEFAULT_CSV = ROOT / "data" / "osv5m_test" / "metadata.csv"
DEFAULT_CALIBRATION_SIZE = 1000
DEFAULT_ALPHA = 0.1
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
DEFAULT_IMAGE_SIZE = 224


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L3 Uncertainty — calibration + conformal + abstention"
    )
    p.add_argument(
        "--run", action="store_true",
        help="Execute the L3 evaluation (default is plan-only)",
    )
    p.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help=f"Path to L2 checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    p.add_argument(
        "--csv", type=str, default=str(DEFAULT_CSV),
        help=f"Official OSV test CSV (default: {DEFAULT_CSV})",
    )
    p.add_argument(
        "--calibration-size", type=int, default=DEFAULT_CALIBRATION_SIZE,
        help=f"Number of samples for calibration split (default: {DEFAULT_CALIBRATION_SIZE})",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Conformal miscoverage rate; coverage target = 1-alpha (default: {DEFAULT_ALPHA})",
    )
    p.add_argument(
        "--threshold", type=float, nargs="+", default=[DEFAULT_THRESHOLD],
        help=f"Abstention confidence threshold(s) (default: {DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="auto|cuda|cpu (default: auto)",
    )
    p.add_argument(
        "--image-size", type=int, default=DEFAULT_IMAGE_SIZE,
        help=f"Image size (default: {DEFAULT_IMAGE_SIZE})",
    )
    p.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers (default: 0 for Windows)",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"RNG seed for deterministic split (default: {DEFAULT_SEED})",
    )
    return p.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# Inference — collect logits
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_logits(
    model: torch.nn.Module,
    dataset,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference and return all logits + labels (cell ids).

    Args:
        model: GeoClassifier (eval mode, on device).
        dataset: GeoDataset or Subset with cell_ids assigned.
        batch_size: DataLoader batch size.
        device: torch device.
        num_workers: DataLoader workers.

    Returns:
        logits: (N, K) tensor of raw model outputs.
        labels: (N,) long tensor of ground-truth cell ids.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_logits: List[torch.Tensor] = []
    all_labels: List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        if isinstance(labels, torch.Tensor):
            all_labels.extend(labels.cpu().tolist())
        else:
            all_labels.extend(labels)

    return torch.cat(all_logits, dim=0), torch.tensor(all_labels, dtype=torch.long)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def eval_haversine(
    probs: torch.Tensor,
    labels: torch.Tensor,
    centroids: np.ndarray,
    true_lats: np.ndarray,
    true_lons: np.ndarray,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """Compute haversine distance metrics on predicted cell centroids.

    Args:
        probs: (N, K) probabilities.
        labels: (N,) not used directly (for ECE etc.).
        centroids: (K, 2) array of [lat, lon] per cell.
        true_lats: (N,) true latitudes.
        true_lons: (N,) true longitudes.
        mask: (N,) optional bool mask — only compute on where True.

    Returns:
        dict with mean_km, median_km, within_{1,25,200}km.
    """
    pred_ids = probs.argmax(dim=1).numpy()
    pred_lats = centroids[pred_ids, 0]
    pred_lons = centroids[pred_ids, 1]

    if mask is not None:
        idx = mask.numpy().astype(bool)
        pred_lats = pred_lats[idx]
        pred_lons = pred_lons[idx]
        true_lats = true_lats[idx]
        true_lons = true_lons[idx]

    if len(pred_lats) == 0:
        return {"mean_km": float("nan"), "median_km": float("nan"),
                "within_1km": 0.0, "within_25km": 0.0, "within_200km": 0.0,
                "n": 0}

    # Vectorized haversine
    phi1 = np.radians(pred_lats)
    phi2 = np.radians(true_lats)
    dphi = np.radians(true_lats - pred_lats)
    dlam = np.radians(true_lons - pred_lons)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    dists = 6371.0 * 2 * np.arcsin(np.sqrt(a))

    return {
        "mean_km": float(dists.mean()),
        "median_km": float(np.median(dists)),
        "within_1km": float((dists <= 1).mean() * 100),
        "within_25km": float((dists <= 25).mean() * 100),
        "within_200km": float((dists <= 200).mean() * 100),
        "n": len(dists),
    }


# ---------------------------------------------------------------------------
# Plan-only display
# ---------------------------------------------------------------------------


def print_plan(args: argparse.Namespace) -> None:
    """Print the L3 experiment plan without loading anything."""
    ckpt_path = Path(args.checkpoint)
    csv_path = Path(args.csv)

    print("=" * 72)
    print("L3 UNCERTAINTY — EXPERIMENT PLAN")
    print("=" * 72)
    print()
    print(f"  Checkpoint:     {ckpt_path}")
    print(f"  Test CSV:       {csv_path}")
    print(f"  Calibration:    first {args.calibration_size} rows (deterministic split)")
    print(f"  Evaluation:     remaining rows (withheld from calibration)")
    print(f"  Alpha:          {args.alpha} (target coverage = {1 - args.alpha:.1%})")
    print(f"  Thresholds:     {args.threshold}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Device:         {args.device}")
    print(f"  Image size:     {args.image_size}")
    print(f"  Seed:           {args.seed}")
    print()
    print("  Steps:")
    print("    1. Load checkpoint + cells metadata")
    print("    2. Load OSV test CSV (3000 rows)")
    print("    3. Deterministic split: calibration / evaluation")
    print("    4. Assign ground-truth lat/lon to checkpoint cells")
    print("    5. Run inference on both splits")
    print("    6. Fit temperature on calibration logits/labels")
    print("    7. Compare raw vs scaled ECE on evaluation")
    print("    8. Fit conformal threshold on calibration probabilities")
    print("    9. Report eval coverage, mean set size, abstention, distances")
    print("   10. Write checkpoints/l3_uncertainty_results.json")
    print()
    print("  NOTE: Calibration data (first 1000 rows) is withheld from")
    print("        final L3 evaluation.  Only the remaining 2000 rows are")
    print("        used for ECE, coverage, abstention, and distance metrics.")
    print()

    # Print example run command
    run_cmd = (
        f"python scripts/run_l3_uncertainty.py --run "
        f"--checkpoint {args.checkpoint} "
        f"--csv {args.csv} "
        f"--calibration-size {args.calibration_size} "
        f"--alpha {args.alpha} "
        f"--threshold {' '.join(str(t) for t in args.threshold)} "
        f"--batch-size {args.batch_size} "
        f"--device {args.device} "
        f"--seed {args.seed}"
    )
    print(f"  Run command:")
    print(f"    {run_cmd}")
    print()
    print("=" * 72)
    print("PLAN COMPLETE — no images loaded, no results written.")
    print("Re-run with --run to execute.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    run = args.run

    if not run:
        print_plan(args)
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    print("=" * 72)
    print("L3 UNCERTAINTY — EXECUTING")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"[1/10] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cells_raw = ckpt.get("cells", [])
    ckpt_args = ckpt.get("args", {})
    num_cells = ckpt.get("num_cells", ckpt_args.get("num_cells", len(cells_raw)))
    print(f"  num_cells={num_cells}, epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('loss', '?')}")

    # Build cells from checkpoint metadata
    from src.cells import Cell
    cells = [
        Cell(
            cell_id=c["cell_id"],
            lat_min=0, lat_max=0, lon_min=0, lon_max=0,
            centroid_lat=c["centroid_lat"],
            centroid_lon=c["centroid_lon"],
        )
        for c in sorted(cells_raw, key=lambda x: x["cell_id"])
    ]
    centroids = np.array(
        [[c.centroid_lat, c.centroid_lon] for c in cells], dtype=np.float64
    )
    print(f"  cells={len(cells)}")

    # ------------------------------------------------------------------
    # 2. Load model and weights
    # ------------------------------------------------------------------
    model = build_model(
        num_cells=num_cells,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    model.eval()
    print("  model loaded on device:", device)

    # ------------------------------------------------------------------
    # 3. Load OSV test CSV
    # ------------------------------------------------------------------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        sys.exit(1)

    from src.dataset import GeoDataset
    full_ds = GeoDataset(str(csv_path), image_size=args.image_size, train=False)
    n_total = len(full_ds)
    print(f"[2/10] OSV test set: {n_total} rows from {csv_path}")

    # ------------------------------------------------------------------
    # 4. Deterministic split
    # ------------------------------------------------------------------
    cal_size = min(args.calibration_size, n_total)
    eval_size = n_total - cal_size
    cal_indices = list(range(cal_size))
    eval_indices = list(range(cal_size, n_total))
    print(f"[3/10] Split: calibration={cal_size}, evaluation={eval_size}")
    print("  NOTE: First {0} rows are calibration (withheld from eval).".format(cal_size))

    from torch.utils.data import Subset
    cal_dataset = Subset(full_ds, cal_indices)
    eval_dataset = Subset(full_ds, eval_indices)

    # ------------------------------------------------------------------
    # 5. Assign ground-truth lat/lon to cells for both splits
    # ------------------------------------------------------------------
    # Get coords from full dataset, then split
    all_coords = full_ds.get_coords().numpy()
    all_cell_ids = assign_cells(all_coords, cells)
    cal_labels = torch.tensor([all_cell_ids[i] for i in cal_indices], dtype=torch.long)
    eval_labels = torch.tensor([all_cell_ids[i] for i in eval_indices], dtype=torch.long)

    # True lat/lon for distance metrics
    eval_true_lats = all_coords[eval_indices, 0]
    eval_true_lons = all_coords[eval_indices, 1]

    # ------------------------------------------------------------------
    # 6. Run inference on calibration split
    # ------------------------------------------------------------------
    print(f"[4/10] Inference on calibration split ({cal_size} samples)...")
    t0 = time.time()
    cal_logits, _ = collect_logits(
        model, cal_dataset, args.batch_size, device, args.num_workers
    )
    cal_probs_raw = F.softmax(cal_logits, dim=1)
    print(f"  done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 7. Run inference on evaluation split
    # ------------------------------------------------------------------
    print(f"[5/10] Inference on evaluation split ({eval_size} samples)...")
    t0 = time.time()
    eval_logits, _ = collect_logits(
        model, eval_dataset, args.batch_size, device, args.num_workers
    )
    eval_probs_raw = F.softmax(eval_logits, dim=1)
    print(f"  done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 8. Temperature scaling
    # ------------------------------------------------------------------
    print("[6/10] Fitting temperature scaling on calibration logits...")
    scaler = TemperatureScaler()
    T = scaler.fit(cal_logits, cal_labels)
    print(f"  fitted T = {T:.4f}")

    eval_probs_scaled = F.softmax(scaler(eval_logits), dim=1)
    cal_probs_scaled = F.softmax(scaler(cal_logits), dim=1)

    # ------------------------------------------------------------------
    # 9. ECE comparison
    # ------------------------------------------------------------------
    print("[7/10] ECE on evaluation split:")
    ece_raw = expected_calibration_error(eval_probs_raw, eval_labels)
    ece_scaled = expected_calibration_error(eval_probs_scaled, eval_labels)
    print(f"  Raw ECE:      {ece_raw:.4f}")
    print(f"  Scaled ECE:   {ece_scaled:.4f}")
    print(f"  Improvement:  {ece_raw - ece_scaled:+.4f}")

    # ------------------------------------------------------------------
    # 10. Conformal prediction
    # ------------------------------------------------------------------
    print(f"[8/10] Conformal prediction (alpha={args.alpha}, target coverage={1 - args.alpha:.1%}):")
    # Fit on calibration probabilities (scaled)
    conformal_q = fit_conformal_quantile(cal_probs_scaled, cal_labels, alpha=args.alpha)
    print(f"  Calibrated quantile q = {conformal_q:.4f}")

    # Eval coverage
    csets = conformal_prediction_set(eval_probs_scaled, alpha=args.alpha, quantile=conformal_q)
    # Coverage: does the true label fall in the prediction set?
    eval_pred_ids = eval_probs_scaled.argmax(dim=1)
    n_correct_in_set = 0
    set_sizes: List[int] = []
    for i in range(len(eval_labels)):
        true_id: int = int(eval_labels[i].item())
        in_set: bool = bool(csets[i, true_id].item())
        if in_set:
            n_correct_in_set += 1
        set_sizes.append(int(csets[i].sum().item()))

    coverage = n_correct_in_set / len(eval_labels)
    mean_set_size = np.mean(set_sizes)
    print(f"  Eval coverage:         {coverage:.4f} (target: {1 - args.alpha:.4f})")
    print(f"  Mean prediction set:   {mean_set_size:.2f} classes")
    print(f"  Max set size:          {max(set_sizes):.0f}")
    print(f"  Min set size:          {min(set_sizes):.0f}")

    # Also compute ECE with top-1 from conformal sets (for reference)
    # (not standard, but informative)
    conformal_top1_mask = csets  # (N, K)

    # ------------------------------------------------------------------
    # 11. Abstention analysis
    # ------------------------------------------------------------------
    print("[9/10] Abstention analysis:")
    abstention_results = {}
    for thr in args.threshold:
        should_pred = abstention_mask(eval_probs_scaled, threshold=thr)
        n_predict = should_pred.sum().item()
        n_abstain = len(should_pred) - n_predict
        abstention_rate = n_abstain / len(should_pred)

        # Accuracy when predicting
        pred_correct = eval_pred_ids.eq(eval_labels)
        correct_when_predicting = (pred_correct & should_pred).sum().item()
        acc_when_predicting = (
            correct_when_predicting / n_predict if n_predict > 0 else 0.0
        )

        # Coverage (fraction where we predict)
        coverage_thr = n_predict / len(should_pred)

        # Distance metrics when predicting
        dists_pred = eval_haversine(
            eval_probs_scaled, eval_labels, centroids,
            eval_true_lats, eval_true_lons,
            mask=should_pred,
        )

        abstention_results[thr] = {
            "threshold": thr,
            "n_predict": int(n_predict),
            "n_abstain": int(n_abstain),
            "abstention_rate": abstention_rate,
            "coverage": coverage_thr,
            "accuracy_when_predicting": acc_when_predicting,
            "distances": dists_pred,
        }

        print(f"  threshold={thr}:")
        print(f"    predict={n_predict}, abstain={n_abstain} "
              f"(abstention rate={abstention_rate:.1%})")
        print(f"    accuracy-when-predicting = {acc_when_predicting:.4f}")
        print(f"    coverage = {coverage_thr:.4f}")
        print(f"    mean_km = {dists_pred['mean_km']:.1f} km, "
              f"median_km = {dists_pred['median_km']:.1f} km")
        print(f"    within 1km={dists_pred['within_1km']:.1f}%, "
              f"25km={dists_pred['within_25km']:.1f}%, "
              f"200km={dists_pred['within_200km']:.1f}%")

    # ------------------------------------------------------------------
    # 12. Full-set distance metrics (no abstention)
    # ------------------------------------------------------------------
    dists_full = eval_haversine(
        eval_probs_scaled, eval_labels, centroids,
        eval_true_lats, eval_true_lons,
    )

    # ------------------------------------------------------------------
    # Build results dict
    # ------------------------------------------------------------------
    results = {
        "checkpoint": str(ckpt_path),
        "csv": str(csv_path),
        "seed": args.seed,
        "n_total": n_total,
        "n_calibration": cal_size,
        "n_evaluation": eval_size,
        "calibration_note": (
            f"First {cal_size} rows of the official OSV test CSV are used for "
            f"calibration (temperature scaling + conformal threshold fitting). "
            f"The remaining {eval_size} rows are used for final L3 evaluation."
        ),
        "num_cells": num_cells,
        "device": str(device),
        "temperature": T,
        "alpha": args.alpha,
        "conformal_quantile": conformal_q,
        "ece": {
            "raw": ece_raw,
            "scaled": ece_scaled,
            "improvement": ece_raw - ece_scaled,
        },
        "conformal": {
            "target_coverage": 1 - args.alpha,
            "eval_coverage": coverage,
            "mean_set_size": mean_set_size,
            "max_set_size": max(set_sizes),
            "min_set_size": min(set_sizes),
        },
        "abstention": {
            f"threshold_{thr}": abstention_results[thr]
            for thr in args.threshold
        },
        "full_eval": dists_full,
        "temperature_fitted_T": T,
    }

    # ------------------------------------------------------------------
    # Print compact summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("L3 UNCERTAINTY RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Checkpoint:    {ckpt_path.name}")
    print(f"  Calibration:   {cal_size} samples (withheld from eval)")
    print(f"  Evaluation:    {eval_size} samples")
    print(f"  Temperature:   T = {T:.4f}")
    print(f"  ECE (raw):     {ece_raw:.4f}")
    print(f"  ECE (scaled):  {ece_scaled:.4f}")
    print(f"  Conformal q:   {conformal_q:.4f}")
    print(f"  Coverage:      {coverage:.4f} (target {1 - args.alpha:.4f})")
    print(f"  Mean set size: {mean_set_size:.2f}")
    for thr in args.threshold:
        r = abstention_results[thr]
        print(f"  Abstain@{thr:.2f}:  rate={r['abstention_rate']:.1%}, "
              f"acc={r['accuracy_when_predicting']:.4f}, "
              f"mean_km={r['distances']['mean_km']:.1f}")
    print(f"  Full eval:     within 1km={dists_full['within_1km']:.1f}%, "
          f"25km={dists_full['within_25km']:.1f}%, "
          f"200km={dists_full['within_200km']:.1f}%")
    print(f"  Mean dist:     {dists_full['mean_km']:.1f} km")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Write results JSON
    # ------------------------------------------------------------------
    results_path = ROOT / "checkpoints" / "l3_uncertainty_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\nResults written to: {results_path}")
    print("[json]" + json.dumps(results, default=_json_default))


def _json_default(obj):
    """Handle numpy/torch types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return "NaN"
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
