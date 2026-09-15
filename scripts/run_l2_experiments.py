"""L2 Data-Efficient Training experiments — 12-run matrix runner.

Matrix (2 inits x 3 fractions x 2 regimes):
    init:     imagenet | places365
    fraction: 0.01 | 0.10 | 1.0
    regime:   frozen | layer4

For each combination:
    1. Builds cells ONCE from the full CSV and saves to JSON.
    2. Selects a deterministic seeded subset for the given fraction.
    3. Runs train.py with --cells-json (fixed cells) + --max-samples + regime flags.
    4. Saves checkpoint to checkpoints/l2_<init>_<fraction>_<regime>/.

Default: plan-only (prints commands, no execution). Use --run to execute.

Windows-safe: pathlib, sys.executable subprocess, no shell strings.

Usage:
    python scripts/run_l2_experiments.py --plan-only
    python scripts/run_l2_experiments.py --plan-only --fractions 0.01 --inits imagenet --regimes frozen
    python scripts/run_l2_experiments.py --run --fractions 0.01 --inits imagenet --regimes frozen --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "osv5m_subset_10k" / "metadata.csv"
CELLS_JSON = ROOT / "checkpoints" / "l2_cells_full.json"
PLACES365_CKPT = ROOT / "checkpoints" / "places365" / "resnet50_places365.pth.tar"
CHECKPOINTS_DIR = ROOT / "checkpoints"

FRACTIONS = [0.01, 0.10, 1.0]
INITS = ["imagenet", "places365"]
REGIMES = ["frozen", "layer4"]

BATCH_FROZEN = 32
BATCH_FINETUNE = 16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="L2 Data-Efficient Training matrix runner (plan-only by default)"
    )
    p.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                   help=f"Full 10k CSV for cell construction (default: {DEFAULT_CSV})")
    p.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS,
                   help=f"Data fractions to test (default: {FRACTIONS})")
    p.add_argument("--inits", type=str, nargs="+", default=INITS,
                   choices=["imagenet", "places365"],
                   help=f"Init types (default: {INITS})")
    p.add_argument("--regimes", type=str, nargs="+", default=REGIMES,
                   choices=["frozen", "layer4"],
                   help=f"Training regimes (default: {REGIMES})")
    p.add_argument("--epochs", type=int, default=10,
                   help="Epochs per run (default: 10)")
    p.add_argument("--image-size", type=int, default=224,
                   help="Image size (default: 224)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (default: 1e-3)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default: 42)")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: auto|cuda|cpu (default: auto)")
    p.add_argument("--plan-only", action="store_true", default=True,
                   help="Print plan and exit (default: True)")
    p.add_argument("--run", action="store_true",
                   help="Actually execute the runs (overrides --plan-only)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose checkpoint dir already has a best.pt")
    p.add_argument("--cells-json", type=str, default=str(CELLS_JSON),
                   help=f"Path for the fixed cells JSON (default: {CELLS_JSON})")
    p.add_argument("--places365-checkpoint", type=str, default=str(PLACES365_CKPT),
                   help=f"Places365 checkpoint path (default: {PLACES365_CKPT})")
    return p.parse_args()


def checkpoint_dir_name(init: str, fraction: float, regime: str) -> str:
    """Standardized checkpoint directory name for a single run."""
    frac_str = f"{fraction:.2f}" if fraction != int(fraction) else f"{fraction:.1f}"
    return f"l2_{init}_{frac_str}_{regime}"


def subset_csv_path(full_csv: Path, fraction: float, seed: int) -> Path:
    """Return a deterministic path for a subset CSV."""
    stem = full_csv.stem
    frac_str = str(fraction).replace(".", "p")
    return full_csv.parent / f"{stem}_sub{frac_str}_s{seed}.csv"


def make_subset_csv(full_csv: Path, fraction: float, seed: int, num_cells: int) -> tuple[Path, int]:
    """Create a deterministic seeded subset CSV from the full CSV.

    Returns (subset_csv_path, actual_num_rows).
    The subset CSV includes image_path, lat, lon (header) only,
    since cells are loaded from JSON and assigned by train.py.
    """
    import numpy as np

    out_path = subset_csv_path(full_csv, fraction, seed)
    if out_path.exists():
        # Count rows to report
        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            n = sum(1 for _ in reader)
        print(f"  subset CSV exists: {out_path} ({n} rows)")
        return out_path, n

    with open(full_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rng = np.random.RandomState(seed)
    n_keep = max(1, int(len(rows) * fraction))
    indices = rng.choice(len(rows), size=n_keep, replace=False)
    selected = [rows[i] for i in sorted(indices)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "lat", "lon"])
        writer.writeheader()
        for row in selected:
            writer.writerow({"image_path": row["image_path"], "lat": row["lat"], "lon": row["lon"]})

    print(f"  subset CSV created: {out_path} ({len(selected)} rows)")
    return out_path, len(selected)


def count_csv_rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def build_cells_from_csv(csv_path: Path, num_cells: int, cells_json_path: Path, seed: int) -> None:
    """Build cells from full CSV and save to JSON (done once)."""
    if cells_json_path.exists():
        print(f"  cells JSON exists: {cells_json_path}")
        return

    print(f"  building cells from {csv_path} ...")
    # We need to build cells using the same logic as train.py.
    # Import here to avoid heavy deps at top level.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from src.cells import build_cells, assign_cells, save_cells_json
    from src.dataset import GeoDataset

    ds = GeoDataset(str(csv_path), image_size=224, train=True)
    coords = ds.get_coords()
    coords_np = coords.cpu().numpy()

    cells = build_cells(coords_np, K=num_cells, method="quad_tree")
    save_cells_json(cells, cells_json_path)
    print(f"  effective cells: {len(cells)}")


def make_train_cmd(args: argparse.Namespace, csv_path: Path, ckpt_dir: Path,
                   init: str, fraction: float, regime: str,
                   batch_size: int, cells_json: str, num_cells: int,
                   subset_size: int) -> list[str]:
    """Build the train.py command as a list (no shell string)."""
    cmd = [
        sys.executable, "-m", "src.train",
        "--csv", str(csv_path),
        "--num-cells", str(num_cells),
        "--epochs", str(args.epochs),
        "--batch-size", str(batch_size),
        "--image-size", str(args.image_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--checkpoint-dir", str(ckpt_dir),
        "--cells-json", cells_json,
        "--max-samples", str(max(1, subset_size)),
        "--run-tag", checkpoint_dir_name(init, fraction, regime),
    ]

    if init == "imagenet":
        cmd.extend(["--pretrained"])
    elif init == "places365":
        cmd.extend(["--no-pretrained", "--places365-checkpoint", args.places365_checkpoint])

    if regime == "layer4":
        cmd.append("--unfreeze-last-block")

    return cmd


def main() -> None:
    args = parse_args()

    plan_only = not args.run

    if plan_only:
        print("=" * 72)
        print("L2 DATA-EFFICIENT TRAINING — EXPERIMENT PLAN")
        print("=" * 72)
    else:
        print("=" * 72)
        print("L2 DATA-EFFICIENT TRAINING — EXECUTING RUNS")
        print("=" * 72)

    # --- Step 1: Build cells from full CSV ---
    full_csv = Path(args.csv)
    if not full_csv.exists():
        print(f"\n[error] Full CSV not found: {full_csv}")
        print("  Run scripts/download_subset.py first to create the 10k training subset.")
        sys.exit(1)

    num_full_rows = count_csv_rows(full_csv)
    # The effective num_cells will be determined by build_cells; use --num-cells as target
    target_num_cells = 300

    print(f"\n[1/3] Cell construction from {full_csv}")
    print(f"  Full dataset rows: {num_full_rows}")
    print(f"  Target cells: {target_num_cells}")

    cells_json_path = Path(args.cells_json)
    if plan_only:
        if not cells_json_path.exists():
            print(f"  Will build cells -> {cells_json_path}")
        else:
            print(f"  Cells JSON exists: {cells_json_path}")
    else:
        build_cells_from_csv(full_csv, target_num_cells, cells_json_path, args.seed)
        # Read back actual num_cells
        with open(cells_json_path, encoding="utf-8") as f:
            data = json.load(f)
        target_num_cells = data["num_cells"]

    # --- Step 2: Generate subset CSVs ---
    print(f"\n[2/3] Deterministic subset CSVs (seed={args.seed})")
    for frac in args.fractions:
        sp = subset_csv_path(full_csv, frac, args.seed)
        n_expected = max(1, int(num_full_rows * frac))
        if plan_only:
            exists_tag = " [exists]" if sp.exists() else ""
            print(f"  fraction {frac:.2f}: {n_expected} rows -> {sp}{exists_tag}")
        else:
            _, n_actual = make_subset_csv(full_csv, frac, args.seed, target_num_cells)

    # --- Step 3: Build run commands ---
    print(f"\n[3/3] Experiment matrix ({len(args.inits)} inits x {len(args.fractions)} fractions x {len(args.regimes)} regimes)")
    total_runs = len(args.inits) * len(args.fractions) * len(args.regimes)
    print(f"  Total runs: {total_runs}")
    print(f"  Epochs per run: {args.epochs}")
    print(f"  Image size: {args.image_size}")
    print(f"  Device: {args.device}")
    print(f"  Cells JSON: {cells_json_path}")
    if args.inits == ["places365"] or set(args.inits) == {"imagenet", "places365"}:
        print(f"  Places365 checkpoint: {args.places365_checkpoint}")
    print()

    results: list[dict] = []
    run_idx = 0
    for init in args.inits:
        for frac in args.fractions:
            for regime in args.regimes:
                run_idx += 1
                ckpt_name = checkpoint_dir_name(init, frac, regime)
                ckpt_dir = CHECKPOINTS_DIR / ckpt_name
                sub_csv = subset_csv_path(full_csv, frac, args.seed)
                batch_size = BATCH_FROZEN if regime == "frozen" else BATCH_FINETUNE

                skip = False
                if args.skip_existing and (ckpt_dir / "best.pt").exists():
                    skip = True

                entry = {
                    "run": run_idx,
                    "init": init,
                    "fraction": frac,
                    "regime": regime,
                    "checkpoint_dir": str(ckpt_dir),
                    "batch_size": batch_size,
                    "skip": skip,
                }

                if skip:
                    print(f"  [{run_idx}/{total_runs}] SKIP (exists): {ckpt_name}")
                    results.append(entry)
                    continue

                cmd = make_train_cmd(
                    args,
                    sub_csv,
                    ckpt_dir,
                    init,
                    frac,
                    regime,
                    batch_size,
                    str(cells_json_path),
                    target_num_cells,
                    max(1, int(num_full_rows * frac)),
                )
                entry["cmd"] = cmd

                print(f"  [{run_idx}/{total_runs}] {ckpt_name}")
                print(f"    init={init} frac={frac:.2f} regime={regime} batch={batch_size}")
                print(f"    csv={sub_csv}")
                print(f"    ckpt={ckpt_dir}")

                if plan_only:
                    print(f"    cmd: {' '.join(cmd)}")
                results.append(entry)

    # --- Summary ---
    print()
    print("=" * 72)
    if plan_only:
        print("PLAN COMPLETE — no runs executed.")
        print("Re-run with --run to execute. Add --skip-existing to resume.")
        print()
        print("Example single-run smoke test (no Places365 download needed):")
        print(f"  python scripts/run_l2_experiments.py --run --fractions 0.01 --inits imagenet --regimes frozen --epochs 1")
    else:
        # Execute runs
        executed = 0
        failed = 0
        for entry in results:
            if entry.get("skip"):
                continue
            cmd = entry["cmd"]
            print(f"\n>>> Executing: {entry['run']}/{total_runs} — {checkpoint_dir_name(entry['init'], entry['fraction'], entry['regime'])}")
            print(f"    cmd: {' '.join(cmd)}")
            t0 = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    check=False,
                    timeout=3600 * 4,  # 4-hour safety timeout per run
                )
                elapsed = time.time() - t0
                entry["returncode"] = result.returncode
                entry["elapsed_sec"] = elapsed
                if result.returncode != 0:
                    failed += 1
                    print(f"    FAILED (returncode={result.returncode}, {elapsed:.0f}s)")
                else:
                    executed += 1
                    print(f"    OK ({elapsed:.0f}s)")
            except subprocess.TimeoutExpired:
                entry["returncode"] = -1
                entry["elapsed_sec"] = 3600 * 4
                failed += 1
                print(f"    TIMEOUT (4h)")
            except Exception as e:
                entry["returncode"] = -2
                entry["elapsed_sec"] = time.time() - t0
                failed += 1
                print(f"    ERROR: {e}")

        print()
        print("=" * 72)
        print(f"L2 EXPERIMENTS COMPLETE — {executed} ok, {failed} failed, "
              f"{sum(1 for e in results if e.get('skip'))} skipped")
        for entry in results:
            name = checkpoint_dir_name(entry["init"], entry["fraction"], entry["regime"])
            status = "SKIP" if entry.get("skip") else f"rc={entry.get('returncode', '?')}"
            print(f"  {name}: {status}")

    # Save plan/manifest
    manifest_path = CHECKPOINTS_DIR / "l2_experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "plan_only": plan_only,
            "fractions": args.fractions,
            "inits": args.inits,
            "regimes": args.regimes,
            "epochs": args.epochs,
            "seed": args.seed,
            "full_csv": str(full_csv),
            "cells_json": str(cells_json_path),
            "results": results,
        }, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
