# KPlaceNet / GeoNet

Evolutionary, layer-by-layer — PlaNet-style coarse geolocation on RTX 4050 (6GB VRAM).

> Locked plan: `docs/implementation_plan.md` — L0→L4, solo + 4050 constraints. Do not add new datasets/backbones without re-plan.

## Setup (RTX 4050, Windows + Python 3.10+, torch 2.x)

```powershell
# 1. Create env (example with venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install deps (no heavy extras)
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify imports (no GPU needed for import check)
python -c "import src.dataset, src.cells, src.model, src.train, src.eval, src.uncertainty; print('imports ok')"
```

> No `pip install` is run by the scaffold — install manually when ready.

## Data Layout (gitignored subsets)

```
data/
  README.md          # explains gitignored subsets
  flickr_geo_tiny/   # 5-10k debug — via scripts/download_subset.py --dataset flickr_geo_tiny
  osv5m_subset_10k/  # 10k stratified — via --dataset osv5m --max-samples 10000
  im2gps3k/          # 3k eval only — never train — via --dataset im2gps3k
  .gitkeep
```

- CSV format expected by `src/dataset.py`: `image_path,lat,lon` (header required). Paths may be absolute or relative to CSV location.
- OSV-5M spatial separation (1km) is respected by not mixing splits; IM2GPS3k is eval-only.
- Large datasets are **not** committed — see `data/README.md`.

## How to Run — Smoke Test (L0 exit gate)

```powershell
# 1) Create a tiny dummy CSV (50 rows) for overfit check
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny --max-samples 50 --dry-run

# 2) Real download (tiny only) — requires huggingface_hub login if gated
python scripts/download_subset.py --dataset flickr_geo_tiny --output-dir data/flickr_geo_tiny
python scripts/download_subset.py --dataset osv5m --output-dir data/osv5m_subset_10k --max-samples 10000
python scripts/download_subset.py --dataset im2gps3k --output-dir data/im2gps3k

# 3) Train smoke — overfit 50 images, 224px, AMP, batch 32->16 fallback note
python -m src.train --csv data/flickr_geo_tiny/metadata.csv --num-cells 300 --epochs 2 --batch-size 32 --image-size 224 --amp

# 4) Eval — within 1/25/200 km via haversine
python -m src.eval --csv data/im2gps3k/metadata.csv --checkpoint checkpoints/last.pt --num-cells 300
```

- `src/train.py` uses AMP (`torch.cuda.amp`) and CE loss; checkpoint saved to `checkpoints/last.pt`.
- If OOM on 4050: re-run with `--batch-size 16` or `--batch-size 8` (plan Section 3/8).
- `src/cells.py` provides `build_cells` / `assign_cells` stubs (quad-tree K~300 in L1; k-means/DBSCAN in L4).

## Repo Layout

```
KPlaceNet/
  docs/implementation_plan.md
  src/dataset.py  cells.py  model.py  train.py  eval.py  uncertainty.py
  scripts/download_subset.py
  notebooks/01_data_explore.ipynb
  data/  # gitignored
  requirements.txt
```

## Notes

- Windows paths safe — uses `pathlib.Path`, no hardcoded absolute paths.
- Imports use `src.*` absolute form; `src` is a package (`src/__init__.py`).
- Heavy work (training, large downloads) is stubbed/flagged — L0 only scaffolds.
