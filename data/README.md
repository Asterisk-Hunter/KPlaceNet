# data/ — gitignored subsets (L0)

This folder is **gitignored** (`data/` is in `.gitignore`). Only this README and `.gitkeep` are committed.

## Layout (after `scripts/download_subset.py`)

```
data/
  README.md
  .gitkeep
  flickr_geo_tiny/
    metadata.csv      # image_path,lat,lon  (5-10k debug, OOM checks)
    images/           # actual images (not committed)
  osv5m_subset_10k/
    metadata.csv      # 10k stratified subset of OSV-5M (plan Sec 2)
    images/
  im2gps3k/
    metadata.csv      # 3k eval-only, never train
    images/
```

## How to populate

```powershell
# Smoke (no download) — 50 dummy images
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny --max-samples 50 --dry-run

# Real (requires huggingface_hub + HF token/approval where gated)
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny
python scripts/download_subset.py --dataset osv5m --output-dir data/osv5m_subset_10k --max-samples 10000
python scripts/download_subset.py --dataset im2gps3k --output-dir data/im2gps3k
```

- CSV format: header `image_path,lat,lon`. `image_path` may be absolute or relative to the CSV file.
- OSV-5M train/test spatial separation (1km) is respected — never mix IM2GPS3k into train.
- Full OSV-5M (5.1M) / YFCC100M are **out of scope** for this plan.

## .gitignore

`data/` is ignored except for this README and `.gitkeep`. Do not commit images or large CSVs.
