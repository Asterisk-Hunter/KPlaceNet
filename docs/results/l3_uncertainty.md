# L3 Uncertainty Results

## Configuration

L3 was evaluated with the selected L2 checkpoint:

- ImageNet initialization
- 100% training data
- `layer4` fine-tuned
- 300 fixed geographic cells
- checkpoint: `checkpoints/l2_imagenet_1.0_layer4/last.pt`

The 3,000-row official OSV test CSV was split deterministically: the first
1,000 rows were used only for temperature and conformal calibration, and the
remaining 2,000 rows were used for evaluation. This makes the reported L3
metrics comparable within this experiment, but the evaluation is not an
entirely untouched 3,000-row test score because calibration consumed part of
the official test file.

## Results

| Metric | Result |
|---|---:|
| Fitted temperature | 3.9387 |
| Raw ECE | 0.4341 |
| Temperature-scaled ECE | **0.0218** |
| Conformal target coverage | 90.0% |
| Conformal evaluation coverage | **91.35%** |
| Mean conformal set size | 188.3 / 300 classes |
| Abstention rate at confidence 0.5 | 99.95% |
| Accuracy when predicting at 0.5 | 0.0% (1 prediction) |
| Evaluation within 25 km | 0.25% |
| Evaluation within 200 km | 2.90% |
| Mean evaluation distance | 6,041 km |

## Interpretation

Temperature scaling substantially improves calibration, meeting the L3 ECE
target. Split-conformal prediction reaches the requested 90% coverage, but
the prediction sets are large because the underlying coarse-cell classifier
has low accuracy on this spatially separated test subset. Likewise, a 0.5
confidence abstention threshold is too strict after calibration: nearly every
prediction is rejected. This is a valid uncertainty result and indicates that
future work should report a coverage/accuracy curve or tune thresholds for a
desired operating point rather than use 0.5 as a universal threshold.

The machine-readable artifact is written to the gitignored path
`checkpoints/l3_uncertainty_results.json`.
