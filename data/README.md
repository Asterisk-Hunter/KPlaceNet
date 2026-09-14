# data/ — gitignored subsets (L0)

This folder is **gitignored** (`data/` is in `.gitignore`). Only this README and `.gitkeep` are committed.

## Layout (after `scripts/download_subset.py`)

```
data/
  README.md
  .gitkeep
  flickr_geo_tiny/
    metadata.csv      # image_path,lat,lon  (dummy smoke set, --dry-run)
    images/           # actual images (not committed)
  osv5m_subset_10k/
    metadata.csv      # image_path,lat,lon,id,split(+country/region if present) — OSV TRAIN split
    images/
  osv5m_test/
    metadata.csv      # official OSV TEST split — eval-only, never train
    images/
  im2gps3k/
    metadata.csv      # 3k eval-only, never train (manual download)
    images/
```

## How to populate

```powershell
# Smoke (no download) — 50 dummy images
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny --max-samples 50 --dry-run

# OSV-5M train subset — SAFE downloader. WARNING: repo osv5m/osv5m is ~259GB
# in full; selected image shards are multi-GB. A token is optional for rate limits/resume.
# The script NEVER snapshot-downloads; it fetches only listed shard zips
# (images/train/NN.zip) + metadata, extracting only selected ids.
# Step 1 — plan only (no network, no files). Step 2 — re-run with --yes.
python scripts/download_subset.py --dataset osv5m --split train --output-dir data/osv5m_subset_10k --max-samples 10000 --plan-only
python scripts/download_subset.py --dataset osv5m --split train --output-dir data/osv5m_subset_10k --max-samples 10000 --yes
# Pin shards explicitly (otherwise sequential discovery from 00, up to --max-shards):
python scripts/download_subset.py --dataset osv5m --split train --output-dir data/osv5m_subset_10k --shards 00,01 --yes

# Official OSV test (eval-only, separate dir — never train on it):
python scripts/download_subset.py --dataset osv5m --split test --output-dir data/osv5m_test --max-samples 3000 --plan-only
python scripts/download_subset.py --dataset osv5m --split test --output-dir data/osv5m_test --max-samples 3000 --yes

# IM2GPS3k — manual only: download from the authors, place images + metadata.csv
# (header image_path,lat,lon) under data/im2gps3k/.
```

- CSV format: header `image_path,lat,lon` plus `id,split` provenance (+country/region if the source has them). `image_path` may be absolute or relative to the CSV file.
- OSV-5M train/test spatial separation (1km) is respected: train rows keep `split=train` and are NEVER re-split into eval; eval uses `--split test` output or IM2GPS3k.
- Full train.csv (~2.92GB) is never fetched unless you pass the explicit opt-in `--allow-full-train-csv`; otherwise train rows are found by a SINGLE capped Range streaming pass (`--csv-scan-cap-mb`, default 512MB) over a seeded shard-balanced candidate union (per-shard quotas summing to `--max-samples`), and the script aborts honestly instead of silently downloading huge files or falling back to first-rows/file-order sampling.
- Row counts are honest: if fewer than `--max-samples` are extracted the script says so and never claims '10k'.
- Full OSV-5M (5.1M) / YFCC100M are **out of scope** for this plan.

## .gitignore

`data/` is ignored except for this README and `.gitkeep`. Do not commit images or large CSVs.
