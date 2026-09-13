# KPlaceNet / GeoNet — Implementation Plan (Locked)

> Evolutionary, layer-by-layer. Solo builder + RTX 4050. No scope creep.

## 0. Constraints (why this plan looks like this)

- **Compute:** RTX 4050 laptop (~6GB VRAM). No 26k-cell classifier, no full OSV-5M (5.1M), no CLIP ViT-L/14 full fine-tune, no 126M-image PlaNet reproduction.
- **Team:** 4-person on paper, effectively solo. All critical-path work must be solo-doable. Teammates get isolated, non-blocking tasks only (viz, docs, eval notebooks).
- **Goal:** First a working PlaNet-style reproduction (coarse + subset), then gaps in fixed order. No deviation without explicit re-plan.

## 1. Locked Scope

### KEEP (in build order)
1. **L1 Baseline reproduction** (PlaNet-style, coarse)
2. **Gap 4 Data-Efficient Training** — do first. Tier 1, ★★★★★ fit. Proves 10x data saving. Medium / Medium.
3. **Gap 3 Uncertainty-Aware Geolocation** — do second. Tier 1, high novelty, cheap compute. Temperature scaling + conformal + abstention. Medium / High.
4. **Gap 1 Adaptive Cell Construction** — do third. k-means / DBSCAN vs quad-tree on same backbone/data. Medium / Medium.

### REMOVED (deferred, not in build)
- **Gap 2 Multi-Modal Fusion — REMOVED.** Needs YFCC100M captions/tags + temporal + CLIP contrastive training. High difficulty, OOM risk on 4050, 3-4 weeks with full team. Revisit only after L1-L4 done.
- **Gap 5 Real-Time Deployment — REMOVED.** Needs pruning/quant/distillation + ONNX/TensorRT + mobile lab. Low novelty for research. We only log inference ms on 4050 as a free stat, no compression track.

**Anti-deviation rule:** No new dataset, no new backbone, no text/temporal fusion, no mobile/ONNX, no cross-view satellite, no full OSV-5M until L1-L4 exit gates are met.

## 2. Dataset Strategy (4050-safe)

| Phase | Dataset | Size | Purpose |
|-------|---------|------|---------|
| Debug | Flickr-Geo subset | 5-10k | iteration, OOM checks |
| Train | OSV-5M subset | 10k (stratified) | main training |
| Eval | IM2GPS3k | 3k | benchmark only, never train |
| Transfer init | Places365 / ImageNet | pretrained weights only | backbone init |

- OSV-5M train/test spatial separation (1km) respected. Never mix IM2GPS3k into train.
- Full OSV-5M / YFCC100M explicitly out of scope for this plan.

## 3. Model Strategy (4050-safe)

- **Backbone:** ResNet-50 pretrained (ImageNet first, Places365 comparison in L2). Frozen initially, last block fine-tune only in L2.
- **Head:** Single FC → 200-500 coarse cells (not 26k). Mixed precision (AMP) on.
- **Batch:** Start 32, drop to 16/8 if OOM. Image size 224 first, 384 only if VRAM allows.
- **Output:** cell softmax → centroid lat/lon + confidence. Region (continent/country) derived from cell metadata.
- **Pipeline:** `photo → resize/norm/augment → ResNet-50 → adaptive/coarse cell classifier → lat/lon + confidence + region (+ uncertainty from L3 onward)`

## 4. Layers + Exit Gates

### L0 Foundation
**Do:**
- Scaffold `src/ data/ notebooks/`, `requirements.txt` (torch, torchvision, sklearn, huggingface_hub, numpy, pandas, matplotlib), README setup
- `scripts/download_subset.py` for Flickr-Geo tiny + OSV-5M 10k + IM2GPS3k eval
- `src/eval.py` with within 1/25/200km (haversine)
- Smoke test: overfit 50 images on 4050

**Exit:** `train.py` runs end-to-end on 4050 without OOM. Eval script outputs numbers.

### L1 Baseline Reproduction
**Do:**
- Quad-tree coarse 200-500 cells from 10k subset locations
- ResNet-50 frozen + FC head, CE loss, AMP
- Train on 10k, eval on IM2GPS3k

**Exit:** Loss drops, eval table produced (even if low, e.g. 5-15% @200km) + 1 demo prediction image → lat/lon + map. No tuning beyond this. L1 code frozen as reference.

### L2 Gap 4 — Data Efficiency (FIRST GAP)
**Hypothesis:** Transfer init reduces data needs by 10x.
**Do (vary one thing at a time, same L1 code):**
- Data fractions: 1% / 10% / 100% of 10k subset
- Init: ImageNet vs Places365
- Regime: frozen vs last-block fine-tune
- Log: learning curves, convergence steps, final acc @1/25/200km

**Exit:** Table + curves showing winner. Decision recorded: which init + regime goes into L3/L4. This justifies everything after.

### L3 Gap 3 — Uncertainty (SECOND GAP)
**Hypothesis:** Calibrated uncertainty makes the system usable (knows when to abstain).
**Do (no arch change):**
- Temperature scaling on L2 winner
- Conformal prediction for prediction sets / intervals
- Abstention threshold + region-wise reliability
- Metrics add: ECE (target <0.1), coverage vs accuracy tradeoff, accuracy-when-predicting

**Exit:** ECE reported, coverage-accuracy curve, demo with "confident" vs "abstain" examples. Model can say don't-know on OOD.

### L4 Gap 1 — Adaptive Cells (THIRD GAP)
**Hypothesis:** Learned cells beat quad-tree on same budget.
**Do (swap only cell construction, keep L2 winner backbone/data):**
- quad-tree coarse vs k-means vs DBSCAN (same K ~300)
- Compare: acc @1/25/200km + cell balance (images/cell std) + regional breakdown (urban vs rural)
- Visualize cell maps

**Exit:** Comparison table + cell maps + regional analysis. Pick final cell method as project default.

## 5. Evaluation Protocol (locked)

- **Primary:** within X km (1 / 25 / 200km) via haversine, on IM2GPS3k + OSV-5M-subset test split
- **L3 adds:** ECE, coverage-accuracy, abstention rate
- **L4 adds:** cell balance, regional split
- No new metrics without re-plan.

## 6. Repo Layout (target)

```
KPlaceNet/
  docs/
    PROJECT.md
    literature_review.md
    Research_Gaps.md
    implementation_plan.md  # this file
  src/
    dataset.py
    cells.py        # quad-tree + k-means/DBSCAN (L1, expanded L4)
    model.py        # ResNet-50 + cell head
    train.py
    eval.py         # within-km + (L3) ECE/coverage
    uncertainty.py  # L3: temp scaling + conformal
  scripts/
    download_subset.py
    make_cells.py
  notebooks/
    01_data_explore.ipynb
    02_baseline_eval.ipynb
    03_data_efficiency.ipynb
    04_uncertainty.ipynb
    05_cells.ipynb
  data/  # gitignored, subsets only
```

## 7. Teammate Lane (non-blocking)

To keep solo progress unblocked:
- **You:** L0→L4 critical path, train/eval code
- **Others (pick any, async):** data viz notebook, README/API docs, failure-case gallery, presentation slides, regional analysis plots
- Rule: nobody touches `model.py` / `train.py` without your review. No dependency on them for exit gates.

## 8. Risks + Mitigations (4050-specific)

| Risk | Mitigation |
|------|------------|
| OOM on 4050 | 224px, batch 32→16→8, frozen backbone first, AMP on, K=300 max |
| Subset too small / biased | stratified sampling by country, log distribution, keep test separate |
| Results weak vs published PlaNet/GeoCLIP | expected — we report coarse-subset numbers honestly + focus on relative gains (L2/L3/L4 deltas), not SOTA chase |
| Teammates idle | non-blocking lanes only, solo can hit all exits alone |
| Scope creep to Gap 2/5 | blocked by Section 1 rule — requires explicit re-plan |

## 9. Success Criteria (project-level)

| Criterion | Target | Stretch |
|-----------|--------|---------|
| Baseline | End-to-end train + eval on 4050, demo prediction | Stable 224 + 384 configs |
| Data efficiency | Show best init/regime + curves, 10x claim tested | 10x holds at 1% data |
| Uncertainty | ECE <0.1, abstention working | ECE <0.05 |
| Cells | Learned vs quad-tree table + maps | Rural gain demonstrated |
| Output | Report + reproducible scripts + notebooks | Paper draft outline from Research_Gaps.md Sec 6 |

## 10. Immediate Next Actions

1. Scaffold L0 (folders + requirements + download_subset + eval stub)
2. Smoke overfit 50 images on 4050
3. Build quad-tree-300 + train L1 on 10k
4. Freeze L1, start L2 fractions

*Source: PROJECT.md (spec/timeline), literature_review.md (9 papers, PlaNet→GeoCLIP), Research_Gaps.md (5 gaps, Tier 1 = Gap 4 + Gap 3). Gaps 2 and 5 deferred per 4050/solo constraint.*
