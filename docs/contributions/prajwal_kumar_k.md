# Fork contribution: conformal correctness and auditable L3 evaluation

**Contributor:** Prajwal Kumar K.

**Scope:** L3 conformal correctness, split/provenance, and regional evaluation diagnostics.

## Baseline and audit

The upstream L3 implementation computed a fractional quantile using PyTorch's
default linear interpolation after applying the finite-sample conformal rank
correction. That does not select the required discrete order statistic and
clamped an out-of-range rank to the sample maximum. The experiment also used
the same held-out labels to fit temperature and fit the conformal threshold,
and assigned split membership by CSV row order despite exposing a seed.

Existing L1-L4 model results and features are upstream work; this contribution
does not claim them. The historical L3 result table is retained but marked as
descriptive because its protocol predates this correction. The checkout had no
OSV test metadata/images or model checkpoint, so no new geolocation or
real-data calibration result could be measured without large data downloads.

## Changes

- Select `k = ceil((n + 1)(1 - alpha))` by discrete order statistic; return
  `+inf` when `k > n`. Validate labels, probabilities, shapes, and alpha, and
  align labels to the probability tensor's device.
- Fit temperature on one seeded subset, conformal scores on a second subset,
  and report metrics on a third. The default 3,000-row setup uses 500 / 1,000 /
  1,500 rows, respectively.
- Record exact row indices and stable sample IDs for each split, the split
  seed, CSV and checkpoint SHA-256 hashes, dependency/runtime versions, and
  repository revision, working-tree state, and inference configuration in
  the result artifact.
- Make geodesic evaluation robust to empty retained sets, shape errors, CPU or
  accelerator tensors, and floating-point roundoff. An empty abstention
  subset now records missing distances as JSON `null`, not nonstandard `NaN`.
- Add equal-area spatial stratification to L3 output. Each occupied region
  reports sample count, top-1 accuracy, conformal coverage and a Wilson
  interval, mean set size, and distance-threshold metrics. Sparse regions remain
  visible with metrics withheld below a configurable sample-count floor.
- Add deterministic unit tests and a small synthetic exchangeable-score
  diagnostic. Update the README and annotate the historical L3 report.

## Validation and results

Run the focused checks from the repository root:

```sh
python -m unittest discover -s tests -v
python -m py_compile src/uncertainty.py src/reproducibility.py scripts/run_l3_uncertainty.py
python scripts/run_l3_uncertainty.py
python scripts/simulate_conformal_coverage.py --trials 20000 --seed 2026
```

The unit tests use manually specified scores to verify the selected rank,
ties, small calibration sets, extreme alpha, invalid inputs, and deterministic
disjoint split membership. Geodesic helper tests verify a one-degree
longitude example against its known distance and the empty-abstention JSON
case. The synthetic diagnostic uses 20,000 iid Uniform
score trials with `n=9` and `alpha=0.01`: the valid rank is 10, so the discrete
threshold is infinite (coverage 1.0000); the previous clamped maximum-score
cutoff covered 0.9037 in this simulation. This checks a finite-sample
mathematical edge case; it is not a geolocation result.

Plan-only smoke output confirms the CLI reports all three partitions without
loading images or writing results in the earlier fork revision. For the added
regional evaluator, four deterministic standard-library tests and Python
byte-compilation pass. The conformal (9 tests) and split/provenance (6 tests)
suites also pass using the cached PyTorch install. The full test discovery and
the updated plan-only CLI could not run in this environment because NumPy is
missing; no dependency packages were installed for this check. No training or
real-data evaluation was run because the checkout contains neither data nor a
checkpoint. No new claim is made about ECE, geographic accuracy, regional
performance, or model superiority. Regional diagnostics are implemented and
synthetically tested, but have not been run on real OSV-5M evaluation data.

## Guarantees and limitations

The usual split-conformal statement is marginal coverage over a future example
when calibration and future examples are exchangeable and the score function
is fixed independently of conformal calibration data. The three-way split
addresses temperature fitting sharing labels with conformal calibration.
Random row splitting does not prove exchangeability: nearby or otherwise
dependent locations, dataset collection biases, and geographic distribution
shift can invalidate the guarantee. No conditional or region-wise coverage is
promised.

The upstream repository has no license file at the audited revision. This
contribution preserves its history and attribution and adds no license; the
maintainer should clarify the project's licensing before downstream reuse.

## Exact contribution

Prajwal Kumar K. implemented the discrete conformal rank correction, the
three-way deterministic split and provenance manifest, spatially stratified
evaluation diagnostics, focused tests, the synthetic diagnostic, and the
documentation updates in this fork. The
upstream's model, datasets, checkpoints, and published historical metrics are
not claimed as this contributor's work.
