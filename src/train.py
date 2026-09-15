"""Train — argparse, AMP, batch 32->16 fallback note, CE loss, checkpoint save.

L0 exit: train.py runs end-to-end on 4050 without OOM (smoke 50 images).
L2: fixed-cell fairness via --cells-json/--save-cells, --max-samples,
    per-run metrics JSON, optional Places365 init.

No heavy logic beyond stubs; uses src.dataset + src.model + src.cells.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow `python -m src.train` and `python src/train.py`
try:
    from src.dataset import GeoDataset, get_transforms
    from src.model import build_model, load_places365_checkpoint
    from src.cells import build_cells, assign_cells, save_cells_json, load_cells_json
except ImportError:
    # Fallback when running as script without package context
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.dataset import GeoDataset, get_transforms
    from src.model import build_model, load_places365_checkpoint
    from src.cells import build_cells, assign_cells, save_cells_json, load_cells_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KPlaceNet L0/L1/L2 train — 4050-safe")
    p.add_argument("--csv", type=str, required=True, help="Path to metadata.csv (image_path,lat,lon)")
    p.add_argument("--num-cells", type=int, default=300, help="Number of coarse cells (200-500, ~300)")
    p.add_argument("--epochs", type=int, default=2, help="Epochs (2 for smoke, more for L1)")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size — 32 default, 16/8 on OOM (Sec 3/8)")
    p.add_argument("--image-size", type=int, default=224, help="224 default, 384 only if VRAM allows")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for head")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained (default True)")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", help="Disable pretrained")
    p.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze backbone (L1 default)")
    p.add_argument("--unfreeze-last-block", action="store_true", help="Unfreeze layer4 (L2 regime)")
    p.add_argument("--amp", action="store_true", default=True, help="Enable AMP (default True)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 for Windows safety)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Where to save checkpoints")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu")
    # L2 additions
    p.add_argument("--cells-json", type=str, default=None,
                   help="Path to fixed cells JSON (loaded instead of building cells). "
                        "Ensures fair comparison across data fractions.")
    p.add_argument("--save-cells", type=str, default=None,
                   help="Path to save built cells as JSON (e.g. cells_l2_full.json). "
                        "Use with --cells-json on subsequent runs.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Deterministic subset: train on at most N samples (seeded). "
                        "When --cells-json is supplied, cells are NOT rebuilt.")
    p.add_argument("--places365-checkpoint", type=str, default=None,
                   help="Path to Places365 resnet50_places365.pth.tar for L2 init comparison.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Optional run identifier for metrics JSON filename (e.g. 'l2_places365_0.01_frozen')")
    return p.parse_args()


def resolve_device(pref: str) -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[train] device={device} | csv={args.csv} | cells={args.num_cells} | batch={args.batch_size} | amp={args.amp}")

    # ------------------------------------------------------------------
    # OOM mitigation note (Section 3/8) — not auto-fallback in L0 stub:
    # If CUDA OOM at batch 32, re-run with --batch-size 16 or 8.
    # Future: catch RuntimeError OOM and retry with half batch.
    # ------------------------------------------------------------------
    if args.batch_size > 32:
        print("[warn] batch-size >32 risks OOM on 4050 (6GB). Consider 32->16->8 per plan Sec 3.")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = GeoDataset(args.csv, image_size=args.image_size, train=True)
    full_len = len(ds)
    print(f"[train] dataset: {full_len} images from {args.csv}")

    # ------------------------------------------------------------------
    # Deterministic subset selection (L2)
    # ------------------------------------------------------------------
    subset_indices = None
    if args.max_samples is not None and args.max_samples < full_len:
        import numpy as np

        rng = np.random.RandomState(args.seed)
        subset_indices = sorted(rng.choice(full_len, size=args.max_samples, replace=False).tolist())
        print(f"[train] deterministic subset: {args.max_samples}/{full_len} (seed {args.seed})")
    elif args.max_samples is not None:
        print(f"[train] --max-samples {args.max_samples} >= dataset {full_len}; using full dataset")

    # ------------------------------------------------------------------
    # Cell construction or loading (L2 fixed-cell fairness)
    # ------------------------------------------------------------------
    if args.cells_json is not None:
        cells = load_cells_json(args.cells_json)
        print(f"[train] using fixed cells from {args.cells_json}: {len(cells)} cells")
        # Fixed cells: do NOT rebuild. Effective num_cells = len(cells).
        num_classes = len(cells)
        # Assign labels for the FULL dataset coords (cell assignment is on full,
        # then subset indices are used for DataLoader).
        coords = ds.get_coords()
        import numpy as np
        coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
        cell_ids = assign_cells(coords_np, cells)
        ds.cell_ids = cell_ids
    else:
        # Build cells from dataset coords (L1 density-driven quad-tree K~300)
        coords = ds.get_coords()
        cells = build_cells(coords, K=args.num_cells, method="quad_tree")
        print(f"[train] cells: {len(cells)} quad-tree cells (target {args.num_cells})")
        num_classes = len(cells)
        import numpy as np
        coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
        cell_ids = assign_cells(coords_np, cells)
        ds.cell_ids = cell_ids
        # Optionally save cells for later fixed-cell runs
        if args.save_cells is not None:
            save_cells_json(cells, args.save_cells)

    if num_classes != args.num_cells:
        print(f"[train] note: cells ({num_classes}) != --num-cells ({args.num_cells}); using head dim = {num_classes}")
    train_args = dict(vars(args))
    train_args["num_cells"] = num_classes

    # ------------------------------------------------------------------
    # Apply subset indices if needed
    # ------------------------------------------------------------------
    if subset_indices is not None:
        from torch.utils.data import Subset
        ds_for_loader = Subset(ds, subset_indices)
        # Override cell_ids on subset: we already set ds.cell_ids above,
        # Subset delegates __getitem__ to ds, so ds.cell_ids is used.
    else:
        ds_for_loader = ds

    # DataLoader — Windows-safe num_workers=0
    loader = DataLoader(
        ds_for_loader,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model — head dim follows actual cells (frozen ResNet-50 + one FC head)
    model = build_model(
        num_cells=num_classes,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )

    # Places365 checkpoint loading (L2 init comparison)
    places365_info = None
    if args.places365_checkpoint is not None:
        print(f"[train] loading Places365 checkpoint: {args.places365_checkpoint}")
        places365_info = load_places365_checkpoint(model, args.places365_checkpoint)
        if args.pretrained:
            print("[train] note: --pretrained was True but Places365 checkpoint loaded; "
                  "ImageNet weights were overwritten by Places365 backbone weights.")

    if args.unfreeze_last_block:
        model.unfreeze_backbone(last_block_only=True)
        print("[train] unfroze layer4 (L2 regime)")

    model.to(device)

    # Snapshot cell centroids alongside checkpoint for eval
    # Optim — only head (and possibly layer4) has requires_grad
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    print(f"[train] trainable params: {trainable_params:,} / {total_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # AMP — torch 2.x uses torch.amp; keep backward compat with torch.cuda.amp
    use_amp = args.amp and device.type == "cuda"
    scaler = None
    if use_amp:
        try:
            from torch.amp import GradScaler, autocast  # torch 2.x
            scaler = GradScaler("cuda")

            def autocast_ctx():
                return autocast("cuda")

        except ImportError:
            from torch.cuda.amp import GradScaler, autocast  # fallback

            scaler = GradScaler()

            def autocast_ctx():
                return autocast()

    else:

        def autocast_ctx():
            # No-op context when AMP off or on CPU
            from contextlib import nullcontext

            return nullcontext()

        # Keep names for uniform code below
        GradScaler = None  # type: ignore
        autocast = None  # type: ignore

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    epoch_metrics: list[dict] = []
    best_loss = float("inf")
    run_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            # labels is int cell id (from dataset) — ensure long tensor
            if isinstance(labels, torch.Tensor):
                labels = labels.to(device, non_blocking=True).long()
            else:
                labels = torch.tensor(labels, device=device, dtype=torch.long)

            optimizer.zero_grad()

            try:
                with autocast_ctx():
                    logits = model(images)
                    loss = criterion(logits, labels)
            except RuntimeError as e:
                # OOM hint — surface actionable message per Section 8
                if "out of memory" in str(e).lower():
                    print(
                        "\n[OOM] CUDA out of memory. Mitigation per plan Sec 3/8: "
                        "re-run with --batch-size 16 or 8, keep --image-size 224, "
                        "ensure --freeze-backbone, keep --amp. "
                        "Not auto-retrying in L0 stub."
                    )
                raise

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, pred = logits.max(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.1f}%" if total else "---")

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = 100 * correct / max(total, 1)
        elapsed = time.time() - start
        print(f"[train] epoch {epoch} -- loss {epoch_loss:.4f} | acc {epoch_acc:.2f}% | {elapsed:.1f}s")

        # Record per-epoch metrics
        epoch_metrics.append({
            "epoch": epoch,
            "loss": epoch_loss,
            "cell_accuracy_pct": epoch_acc,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "device": str(device),
            "elapsed_sec": elapsed,
            "num_classes": num_classes,
            "dataset_size": full_len,
            "subset_size": len(subset_indices) if subset_indices is not None else full_len,
        })

        # Checkpoint
        ckpt_path = ckpt_dir / "last.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "args": train_args,
                "cells": [{"cell_id": c.cell_id, "centroid_lat": c.centroid_lat, "centroid_lon": c.centroid_lon} for c in cells],
                "loss": epoch_loss,
                "num_cells": num_classes,
            },
            ckpt_path,
        )
        print(f"[train] saved checkpoint -> {ckpt_path}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "args": train_args,
                    "cells": [{"cell_id": c.cell_id, "centroid_lat": c.centroid_lat, "centroid_lon": c.centroid_lon} for c in cells],
                    "loss": epoch_loss,
                    "num_cells": num_classes,
                },
                best_path,
            )
            print(f"[train] new best -> {best_path}")

    # ------------------------------------------------------------------
    # Write machine-readable metrics JSON (L2)
    # ------------------------------------------------------------------
    total_elapsed = time.time() - run_start
    run_meta = {
        "csv": str(args.csv),
        "checkpoint_dir": str(ckpt_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "unfreeze_last_block": args.unfreeze_last_block,
        "amp": args.amp,
        "num_cells_requested": args.num_cells,
        "num_cells_effective": num_classes,
        "dataset_size": full_len,
        "subset_size": len(subset_indices) if subset_indices is not None else full_len,
        "cells_json": args.cells_json,
        "places365_checkpoint": args.places365_checkpoint,
        "run_tag": args.run_tag,
        "device": str(device),
        "total_elapsed_sec": total_elapsed,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    if places365_info is not None:
        run_meta["places365_loaded"] = places365_info["loaded"]
        run_meta["places365_skipped"] = places365_info["skipped"]
        run_meta["places365_mismatched"] = places365_info["mismatched"]

    metrics = {
        "run": run_meta,
        "epochs": epoch_metrics,
    }

    tag = args.run_tag or "run"
    metrics_path = ckpt_dir / f"metrics_{tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] metrics -> {metrics_path}")

    print("[train] done. Eval with: python -m src.eval --csv data/im2gps3k/metadata.csv --checkpoint checkpoints/last.pt")


if __name__ == "__main__":
    main()
