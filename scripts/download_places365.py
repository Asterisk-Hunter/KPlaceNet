"""Download the official Places365 ResNet-50 checkpoint.

Default mode is plan-only: prints the exact download plan and exits.
Re-run with --yes to actually download (~97 MB) to checkpoints/places365/.

This downloads ONLY the pretrained weights checkpoint, NOT the Places365
image dataset. The checkpoint is used for L2 transfer-init comparison.

URL: http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
CHECKPOINT_DIR = Path("checkpoints/places365")
CHECKPOINT_FILENAME = "resnet50_places365.pth.tar"
EXPECTED_SIZE_MB = 97  # approximate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download official Places365 ResNet-50 checkpoint (~97 MB). "
                    "Plan-only by default; use --yes to download."
    )
    p.add_argument(
        "--output-dir", type=str, default=str(CHECKPOINT_DIR),
        help=f"Directory to save the checkpoint (default: {CHECKPOINT_DIR})"
    )
    p.add_argument(
        "--yes", action="store_true",
        help="Confirm download. Without this flag the script only prints the plan."
    )
    p.add_argument(
        "--url", type=str, default=CHECKPOINT_URL,
        help="Override the checkpoint URL (for mirrors or local copies)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / CHECKPOINT_FILENAME

    print("[plan] Places365 ResNet-50 checkpoint download")
    print(f"  URL:       {args.url}")
    print(f"  Dest:      {dest}")
    print(f"  Est size:  ~{EXPECTED_SIZE_MB} MB")
    print(f"  Purpose:   L2 transfer-init comparison (backbone weights only)")
    print(f"  NOTE:      This is NOT the Places365 image dataset.")

    if dest.exists():
        print(f"  Status:    CHECKPOINT ALREADY EXISTS at {dest}")
        print("  No download needed.")
        return

    if not args.yes:
        print("\n[gate] Re-run with --yes to download the checkpoint.")
        sys.exit(0)

    print(f"\n[download] Fetching from {args.url} ...")
    try:
        def _progress(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded / total_size * 100)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct:.0f}%)", end="", flush=True)

        urllib.request.urlretrieve(args.url, str(dest), reporthook=_progress)
        print()  # newline after progress

        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[done] Checkpoint saved: {dest} ({size_mb:.1f} MB)")
        print("[usage] Pass to train.py:")
        print(f"  python -m src.train --csv ... --places365-checkpoint {dest}")
    except Exception as e:
        print(f"\n[error] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
