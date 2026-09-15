# L2 Data-Efficiency Results

**Run date:** 2026-09-15  
**Device:** CUDA, RTX 4050  
**Implementation:** `8965f7c` plus Places365 key-mapping fix in the next commit

## Experimental controls

- Same fixed 300-cell quad-tree JSON for every run
- Same deterministic OSV-5M subsets: 100 / 1,000 / 10,000 images, seed 42
- Same official OSV test subset: 3,000 images from all five test shards
- Input: 224x224, AMP enabled
- Frozen regime: head only, 614,700 trainable parameters
- Layer4 regime: head + ResNet layer4, 15,579,436 trainable parameters
- 10 epochs per run
- Places365: official 92.8MB checkpoint; 265/318 backbone state entries loaded, 0 shape mismatches, classifier skipped

## Official-test results

Percentages are accuracy within the stated distance threshold.

| Init | Data | Regime | 1km | 25km | 200km | Mean km |
|---|---:|---|---:|---:|---:|---:|
| ImageNet | 1% | frozen | 0.00% | 0.00% | 0.27% | 8176.3 |
| ImageNet | 1% | layer4 | 0.00% | 0.00% | 0.50% | 8420.7 |
| ImageNet | 10% | frozen | 0.00% | 0.03% | 0.97% | 7452.8 |
| ImageNet | 10% | layer4 | 0.00% | 0.07% | 1.43% | 7108.5 |
| ImageNet | 100% | frozen | 0.00% | 0.07% | 2.53% | 6762.2 |
| ImageNet | 100% | layer4 | 0.00% | 0.27% | **3.33%** | 6029.7 |
| Places365 | 1% | frozen | 0.00% | 0.03% | 0.37% | 9259.9 |
| Places365 | 1% | layer4 | 0.00% | 0.00% | 0.43% | 8402.7 |
| Places365 | 10% | frozen | 0.00% | 0.03% | 1.13% | 7263.7 |
| Places365 | 10% | layer4 | 0.00% | 0.03% | 0.97% | 7324.7 |
| Places365 | 100% | frozen | 0.00% | 0.10% | 2.43% | 6931.4 |
| Places365 | 100% | layer4 | 0.00% | 0.20% | 2.87% | 5800.3 |

## Conclusion

The hypothesis is **partially supported**. More data improves official-test performance, and layer4 fine-tuning is consistently better than frozen ImageNet at 10% and 100%. The best run is ImageNet + 100% data + layer4 at 3.33% within 200km.

Places365 transfer is valid after mapping the legacy checkpoint's named keys into the numeric `Sequential` backbone. It performs below ImageNet at 1% and 100% within-200km, so ImageNet remains the selected initialization for L3 and L4. This is a useful negative result, not a failed experiment.

The experiment does **not** establish the planned 10x claim: 10% data does not match the 100% baseline. The next phase should use the selected ImageNet/layer4 checkpoint and add uncertainty calibration, as specified by the locked plan.

Generated checkpoints, logs, subset CSVs, and downloaded weights remain gitignored.
