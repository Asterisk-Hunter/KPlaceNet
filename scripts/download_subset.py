"""Download subsets — Flickr-Geo tiny + OSV-5M 10k via huggingface_hub + IM2GPS3k note.

No actual large download by default. Use --dry-run for smoke, or run with
--dataset to fetch real (gated) data. 4050-safe: 10k max for OSV-5M subset.

Windows-safe: uses pathlib, no hardcoded absolute paths.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys


# Known dataset ids / notes (may require HF token / approval)
DATASETS = {
    "flickr_geo_tiny": {
        "hf_id": "flickr-geo-tiny",  # placeholder — replace with actual Flickr-Geo HF id if published
        "note": "Flickr-Geo debug subset 5-10k (iteration, OOM checks). If HF id unavailable, generate dummy CSV with --dry-run.",
    },
    "osv5m": {
        "hf_id": "osv5m/osv-5m",  # canonical OSV-5M on HF (requires approval)
        "note": "OSV-5M full is 5.1M. This script fetches 10k stratified subset only (Section 2).",
    },
    "im2gps3k": {
        "hf_id": "im2gps/im2gps3k",  # community mirror; some are gated
        "note": "IM2GPS3k — 3k eval only, never train (Section 2). If gated, download manually from authors and place under data/im2gps3k/.",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KPlaceNet subset downloader (L0 scaffold)")
    p.add_argument(
        "--dataset",
        type=str,
        default="flickr_geo_tiny",
        choices=["flickr_geo_tiny", "osv5m", "im2gps3k"],
        help="Which dataset to fetch",
    )
    p.add_argument("--output-dir", type=str, default="data/flickr_geo_tiny", help="Output directory (relative or absolute)")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to keep (e.g. 10000 for OSV-5M 10k)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for stratified sampling")
    p.add_argument("--dry-run", action="store_true", help="Do not download — create dummy metadata.csv for smoke test")
    p.add_argument("--hf-token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env)")
    return p.parse_args()


def make_dummy_csv(output_dir: Path, n: int = 50, seed: int = 42) -> Path:
    """Create a dummy metadata.csv with n rows (image_path,lat,lon).

    Also creates placeholder image files (1x1) so dataset.py can open them
    without needing real downloads — for smoke overfit of 50 images on 4050.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Try to create tiny valid images via PIL if available
    try:
        from PIL import Image

        has_pil = True
    except ImportError:
        has_pil = False

    rows = []
    for i in range(n):
        # Random lat/lon uniformly (not realistic distribution, just for scaffold)
        lat = rng.uniform(-60, 70)
        lon = rng.uniform(-180, 180)
        img_name = f"dummy_{i:05d}.jpg"
        img_path = images_dir / img_name

        if has_pil and not img_path.exists():
            # 64x64 random color — small but valid JPEG
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            Image.new("RGB", (64, 64), color).save(img_path, "JPEG")

        # Store relative path from output_dir (dataset.py resolves relative to CSV)
        rel = Path("images") / img_name
        rows.append((rel.as_posix(), f"{lat:.6f}", f"{lon:.6f}"))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "lat", "lon"])
        w.writerows(rows)

    # Create placeholder images as empty files if PIL not available
    if not has_pil:
        for rel, _, _ in rows:
            p = output_dir / rel
            if not p.exists():
                p.write_bytes(b"")  # will fail on open — user needs PIL; warn
        print("[warn] Pillow not installed — dummy images are empty. Install Pillow before smoke train.")

    print(f"[dry-run] wrote {n} rows → {csv_path}")
    return csv_path


def snapshot_download_subset(dataset: str, output_dir: Path, max_samples: int | None, token: str | None) -> None:
    """Download via huggingface_hub snapshot_download, then subsample.

    For OSV-5M 10k: downloads subset then keeps at most max_samples via
    stratified sampling (stub: random uniform in L0, stratified by cell in L1).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[error] huggingface_hub not installed. Run: pip install huggingface_hub")
        print("        Or use --dry-run for smoke test without download.")
        sys.exit(1)

    info = DATASETS[dataset]
    hf_id = info["hf_id"]
    print(f"[download] dataset={dataset} hf_id={hf_id}")
    print(f"           note: {info['note']}")

    # Allow filtering to reduce download size — use allow_patterns if known
    # L0: download repo then handle subset locally (HF filters are repo-specific)
    try:
        local_dir = snapshot_download(
            repo_id=hf_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
            token=token,
            # For OSV-5M, we could allow_patterns=["*.csv", "*.parquet"] to avoid images
            # but actual layout varies — so download all and subsample CSV.
        )
        print(f"[download] snapshot → {local_dir}")
    except Exception as e:
        print(f"[error] snapshot_download failed: {e}")
        print("        For OSV-5M you may need HF approval; for IM2GPS3k try manual download.")
        print("        Use --dry-run to create dummy data for smoke test.")
        sys.exit(1)

    # Post-process: if a metadata.csv / parquet exists, subsample to max_samples
    if max_samples is not None:
        print(f"[subset] max_samples={max_samples} — look for CSV/parquet to subsample (L0: no-op if none)")
        # Generic: find first CSV that looks like metadata
        for csv_path in output_dir.rglob("*.csv"):
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if {"image_path", "lat", "lon"} & set(df.columns):
                    if len(df) > max_samples:
                        df = df.sample(n=max_samples, random_state=42)
                        df.to_csv(csv_path, index=False)
                        print(f"[subset] truncated {csv_path} → {len(df)} rows")
                    break
            except Exception as e:
                print(f"[warn] could not subsample {csv_path}: {e}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    # Resolve relative to project root (where script is run from)
    # Keep Windows-safe: do not hardcode absolute
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    print(f"[info] KPlaceNet download_subset — 4050-safe, 10k cap")
    print(f"       dataset={args.dataset} → {output_dir} (dry_run={args.dry_run})")

    if args.dataset == "im2gps3k":
        print("[note] IM2GPS3k is eval-only, never train (plan Sec 2). 3k benchmark.")

    if args.dry_run:
        n = args.max_samples if args.max_samples is not None else 50
        make_dummy_csv(output_dir, n=n, seed=args.seed)
        print("[done] dry-run complete. Smoke train with:")
        print(f"       python -m src.train --csv {output_dir / 'metadata.csv'} --epochs 2 --batch-size 32 --amp")
        return

    # Real download (no large full OSV-5M)
    if args.dataset == "osv5m" and args.max_samples is None:
        # Default cap for OSV-5M per plan Section 2
        args.max_samples = 10000
        print(f"[info] OSV-5M default cap → --max-samples 10000 (override with explicit value)")

    token = args.hf_token or None
    snapshot_download_subset(args.dataset, output_dir, args.max_samples, token)
    print("[done] download complete.")


if __name__ == "__main__":
    main()
