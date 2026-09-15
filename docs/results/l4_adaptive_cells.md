# L4 — Adaptive Cell Construction (Gap 1)

## Hypothesis

Learned cells (k-means, DBSCAN) beat quad-tree on the same backbone/data budget.

## Methodology

### Cell Construction

Three methods compared at K=300 cells, built from the full 10k OSV-5M training subset:

| Method | Description | Key Properties |
|--------|-------------|----------------|
| `quad_tree` | Density-driven recursive spatial bisection | Axis-aligned boxes, exact containment, O(N log K) |
| `kmeans` | MiniBatchKMeans on lat/lon (cos-mean-lon projection) | Voronoi-like partitions, equal-area scaling, seed=42 |
| `dbscan` | DBSCAN density clustering + eps binary search for ~K clusters | Noise reassignment to nearest centroid, merge/split to exactly K |

### Training Configuration

All three methods use the same L2 winner regime:

- **Backbone:** ResNet-50, ImageNet pretrained
- **Regime:** Layer4 fine-tuned (`--unfreeze-last-block`)
- **Data:** 100% of 10k OSV-5M training subset
- **Epochs:** 10
- **Batch size:** 16 (AMP on)
- **LR:** 1e-3 (AdamW)
- **Image size:** 224px

Each method gets its own fixed cells JSON (`l4_cells_<method>.json`) and
checkpoint directory (`checkpoints/l4_<method>/`).

### Cell Balance

Cell balance is measured on the training assignment distribution:

| Method | Mean | Std | Min | Max | Median | Gini |
|--------|------|-----|-----|-----|--------|------|
| quad_tree | 33.3 | 21.7 | 1 | 99 | 30.0 | 0.370 |
| kmeans | 33.3 | 19.4 | 0 | 113 | 31.0 | 0.324 |
| dbscan | 33.3 | 110.8 | 3 | 1864 | 17.0 | 0.559 |

**Observations:**
- K-means has the most balanced distribution (lowest Gini = 0.324)
- Quad-tree is moderately balanced (Gini = 0.370)
- DBSCAN is highly imbalanced (Gini = 0.559) — some mega-clusters capture dense urban areas while many cells are small

### Urban/Rural Proxy

Since OSV-5M has no urban/rural labels, we use a **cell-count proxy**:
- **Urban:** cells with training point count > median count
- **Rural:** cells with training point count <= median count

| Method | Urban Cells | Rural Cells | Urban Points | Urban % |
|--------|-------------|-------------|--------------|---------|
| quad_tree | 146 | 154 | 7561 | 75.6% |
| kmeans | 148 | 152 | 7248 | 72.5% |
| dbscan | 148 | 152 | 8202 | 82.0% |

**Caveat:** This proxy is a rough approximation. Dense urban areas naturally
have higher training counts, so "urban" ≈ "dense training regions" rather
than true urban classification.  We report this transparently.

### DBSCAN Normalization

DBSCAN naturally produces variable-density clusters.  To normalize to K=300:

1. Binary-search for eps yielding ~K non-noise clusters (ball_tree, min_samples=5)
2. Reassign noise points (label=-1) to nearest non-noise centroid via haversine
3. Merge smallest clusters into nearest neighbor if >K
4. Split largest clusters at centroid median if <K
5. Re-contiguify labels to 0..K-1

This produces exactly 300 cells, but the underlying density variation means
some cells cover large geographic areas (sparse rural) while others are tiny
(dense urban).

## Evaluation Results

Evaluated on the official OSV test set (3000 images, held-out):

| Method | @1km | @25km | @200km | Mean Dist | Median Dist |
|--------|------|-------|--------|-----------|-------------|
| quad_tree | 0.00% | 0.27% | 3.33% | 6024.9 km | 4824.3 km |
| kmeans | 0.00% | 0.17% | 4.07% | 6057.9 km | 5178.6 km |
| dbscan | 0.00% | 0.07% | 3.47% | 6083.5 km | 5380.5 km |

### Analysis

- **Within-200km:** kmeans leads at 4.07%, followed by dbscan (3.47%) and quad_tree (3.33%)
- **Within-25km and @1km:** All methods show very low accuracy, consistent with
  the coarse 300-cell granularity on a global geographic task
- **Mean distance:** All methods produce similar mean distances (~6000 km),
  indicating the cell partitioning alone does not dramatically change performance
  at this scale

### Recommendation

**kmeans** is the recommended default cell method:
- Highest within-200km accuracy (4.07%)
- Best cell balance (Gini = 0.324)
- Predictable exact-K output
- Lower inference ambiguity (Voronoi-like partitions)

The differences between methods are modest, suggesting that at K=300 with
a ResNet-50 backbone, the cell construction method is not the primary
bottleneck.  Performance is limited by the coarse cell granularity and the
10k training subset size relative to global geography.

## Files Produced

```
checkpoints/
  l4_cells_quad_tree.json    # quad-tree cells (300)
  l4_cells_kmeans.json       # k-means cells (300)
  l4_cells_dbscan.json       # DBSCAN cells (300)
  l4_quad_tree/              # quad-tree checkpoint (best.pt + last.pt)
  l4_kmeans/                 # k-means checkpoint
  l4_dbscan/                 # DBSCAN checkpoint
  l4_experiment_manifest.json # full results JSON
```

## Reproduction

```powershell
# Plan-only (no execution)
python scripts/run_l4_experiments.py

# Full run
python scripts/run_l4_experiments.py --run

# Resume with existing checkpoints
python scripts/run_l4_experiments.py --run --skip-existing

# Single method
python scripts/run_l4_experiments.py --run --methods kmeans

# Custom K and epochs
python scripts/run_l4_experiments.py --run --K 200 --epochs 5
```
