# L1 Baseline Results

**Run date:** 2026-09-14  
**Device:** CUDA, RTX 4050  
**Commit containing implementation:** `b2bc1e1`

## Configuration

- Dataset: OSV-5M train subset, 10,000 images
- Acquisition: official train shards `00, 01, 02, 03`
- Sampling: deterministic seed 42, 2,500 candidate IDs per shard
- Coverage: 161 countries
- Evaluation: official OSV-5M test subset, 3,000 images from all five test shards
- Test coverage: 168 countries
- Model: ImageNet-pretrained ResNet-50
- Backbone: frozen
- Head: one `Linear(2048, 300)` geographic-cell classifier
- Cells: density-driven quad-tree, target K=300
- Input: 224x224
- Optimizer: AdamW, learning rate 1e-3, weight decay 1e-4
- Loss: cross-entropy over geographic cells
- Training: 10 epochs, batch size 32, AMP enabled

## Training

```text
Final epoch loss: 1.8139
Final cell accuracy: 68.84%
```

## Metrics

| Evaluation set | Within 1km | Within 25km | Within 200km | Mean distance | Median distance |
|---|---:|---:|---:|---:|---:|
| Train subset sanity check | 0.07% | 3.30% | 38.44% | 3116.0 km | 466.6 km |
| Official OSV test subset | 0.00% | 0.07% | 2.53% | 6762.2 km | 6638.2 km |

## Interpretation

The pipeline is complete end-to-end: OSV data acquisition, density-driven cell construction, training, checkpointing, centroid decoding, and haversine evaluation all work on the RTX 4050.

The generalization result is weak and is not being presented as state of the art. The current experiment trains only a frozen ImageNet backbone's cell head on 10,000 images. The gap between the train sanity score and official spatially separated test score is the baseline to improve in later phases, especially Gap 4 (data-efficient training) and then Gap 3 (uncertainty).

The earlier one-shard run was discarded as the representative baseline because it covered only 75 countries and was strongly biased toward a few countries. The reported run uses four-shard balanced sampling and covers 161 countries.

## Reproduction commands

```powershell
# Train
python -X utf8 -m src.train `
  --csv data/osv5m_subset_10k/metadata.csv `
  --num-cells 300 --epochs 10 --batch-size 32 --image-size 224 --amp `
  --checkpoint-dir checkpoints/l1_osv5m_10k_balanced

# Evaluate official OSV test subset
python -X utf8 -m src.eval `
  --csv data/osv5m_test/metadata.csv `
  --checkpoint checkpoints/l1_osv5m_10k_balanced/last.pt `
  --num-cells 300 --batch-size 32 --image-size 224
```

Generated datasets and checkpoints are intentionally gitignored; this document records the reproducible configuration and observed metrics.
