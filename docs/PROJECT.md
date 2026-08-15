# Project Overview

## Project Name Options

### Top Pick: **GeoNet**
- Short, memorable, professional
- Doesn't sound like AI slop
- Easy to say and remember

### Alternatives:
- **WherePix** — Playful but professional
- **LocateAI** — Direct but clean
- **PlaceFinder** — Functional name
- **GeoScope** — Slightly more technical

---

## Project Description

### One-Liner
> A deep learning system that predicts where a photo was taken using only its visual content.

### Full Description
GeoNet is a research project exploring efficient and uncertainty-aware photo geolocation using convolutional neural networks. The system takes a single photograph as input and predicts its geographic coordinates (latitude, longitude) without relying on GPS metadata or EXIF data.

Building on the foundational PlaNet architecture (Weyand et al., 2016), our project investigates three key research directions:

1. **Adaptive Geographic Cells** — Learning optimal spatial partitions instead of using fixed quad-trees
2. **Data-Efficient Training** — Leveraging transfer learning and self-supervision to reduce training data requirements
3. **Uncertainty Quantification** — Providing calibrated confidence estimates for geolocation predictions

### Applications
- **Travel & Tourism**: Automatic photo organization by location
- **Content Moderation**: Verifying location claims in social media
- **Forensics**: Geolocating images for investigative purposes
- **Autonomous Systems**: Visual localization for robots and vehicles
- **Cultural Heritage**: Documenting and preserving location context

### Technical Approach
```
Input: Single photograph (any aspect ratio)
  ↓
Preprocessing: Resize, normalize, augment
  ↓
CNN Backbone: ResNet-50 / EfficientNet (pre-trained)
  ↓
Geographic Head: Adaptive cell classifier
  ↓
Output: 
  - Predicted coordinates (lat, lon)
  - Confidence score
  - Uncertainty estimate
  - Geographic region (continent, country)
```

---

## Datasets

### Primary Datasets (Recommended for Your Project)

#### 1. OpenStreetView-5M (OSV-5M) ⭐ **RECOMMENDED**
| Property | Details |
|----------|---------|
| **Source** | CVPR 2024 paper |
| **Size** | 5.1 million geo-referenced street view images |
| **Coverage** | 225 countries and territories |
| **Access** | Open, free download |
| **URL** | https://huggingface.co/datasets/osv5m/osv5m |
| **Code** | https://github.com/gastruc/osv5m |

**Why use this:**
- Latest dataset (2024), modern and well-curated
- Global coverage, balanced distribution
- Train/test spatial separation (1km) prevents data leakage
- Active community and benchmark

**Download:**
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", repo_type='dataset')
```

---

#### 2. YFCC100M (Multimedia Commons)
| Property | Details |
|----------|---------|
| **Source** | Yahoo/Flickr |
| **Size** | 100 million images (48 million geotagged) |
| **Access** | Free with Flickr account |
| **URL** | https://multimediacommons.wordpress.com/yfcc100m-core-dataset/ |

**Why use this:**
- Massive scale for training
- Rich metadata (tags, captions, timestamps)
- Used by PlaNet and many other papers

**Note:** Large download (12TB total), need to filter geotagged subset

---

#### 3. IM2GPS / IM2GPS3k (Benchmark)
| Property | Details |
|----------|---------|
| **Source** | CMU (Hays & Efros, 2008) |
| **Size** | 237 test images (IM2GPS), 2,997 test images (IM2GPS3k) |
| **Access** | Free download |
| **URL** | http://graphics.cs.cmu.edu/projects/im2gps/ |

**Why use this:**
- Standard benchmark for evaluation
- Compare against published results
- Small, easy to evaluate

---

### Secondary Datasets (For Specific Experiments)

#### 4. Places365
| Property | Details |
|----------|---------|
| **Source** | MIT (Zhou et al., 2017) |
| **Size** | 1.8 million images, 365 scene categories |
| **Access** | Free |
| **URL** | https://github.com/csailvision/places365 |
| **Use Case** | Transfer learning initialization |

#### 5. CVUSA / CVACT (Cross-View)
| Property | Details |
|----------|---------|
| **Source** | Ground-satellite image pairs |
| **Size** | CVUSA: 35k pairs, CVACT: 44k pairs |
| **Access** | Free |
| **URL** | https://mvrl.cse.wustl.edu/datasets/cvusa/ |
| **Use Case** | Cross-view geolocation experiments |

#### 6. Flickr-Geo (HuggingFace)
| Property | Details |
|----------|---------|
| **Source** | HuggingFace |
| **Size** | Subset of Flickr geotagged images |
| **Access** | Free |
| **URL** | https://huggingface.co/datasets/do-me/Flickr-Geo |
| **Use Case** | Quick experiments, prototyping |

---

### Dataset Strategy for Your Project

| Phase | Dataset | Purpose |
|-------|---------|---------|
| **Prototyping** | Flickr-Geo (small subset) | Quick iteration, debugging |
| **Training** | OSV-5M (full) | Main training data |
| **Evaluation** | IM2GPS3k + OSV-5M test | Benchmark comparison |
| **Transfer Learning** | Places365 | Pre-training backbone |

---

## Feasibility Assessment: Can 4 UG Students Do This?

### Short Answer: **YES, absolutely.**

### Why This Is Feasible

| Factor | Assessment | Notes |
|--------|------------|-------|
| **Technical Complexity** | Medium | CNNs are well-understood; many tutorials exist |
| **Compute Requirements** | Low-Medium | Single GPU sufficient; Google Colab available |
| **Data Availability** | Excellent | All datasets are open and free |
| **Code Availability** | Good | PlaNet, GeoCLIP have implementations |
| **Literature Support** | Excellent | Clear progression from IM2GPS → PlaNet → GeoCLIP |
| **Novelty Potential** | High | Several gaps identified in literature |

### Realistic Timeline (8-10 Weeks)

| Week | Milestone | Effort |
|------|-----------|--------|
| 1-2 | Environment setup, dataset download, literature review | Low |
| 3-4 | Implement baseline PlaNet, reproduce results | Medium |
| 5-6 | Implement adaptive cell construction | Medium |
| 7-8 | Implement data efficiency experiments | Medium |
| 9-10 | Uncertainty quantification + paper writing | Medium |

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **GPU** | Google Colab (free) | RTX 3060+ or cloud GPU |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 100 GB | 500 GB |
| **Time** | 10 hrs/week | 15-20 hrs/week |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset too large | Medium | Medium | Use subsets, cloud storage |
| Training too slow | Medium | Low | Use Colab Pro, smaller models |
| Results not novel | Low | High | Focus on gaps, proper ablation |
| Integration issues | Medium | Medium | Clear API design, regular sync |

### What Makes This Project Strong

1. **Clear Research Gaps** — We identified 5 specific, actionable gaps
2. **Incremental Approach** — Can deliver baseline + improvements
3. **Publication Potential** — Novel contributions in cell construction, uncertainty
4. **Practical Impact** — Real-world applications
5. **Learning Value** — Covers CV, DL, research methodology

---

## Team Roles (Suggested)

| Role | Responsibility | Skills Needed |
|------|----------------|---------------|
| **ML Lead** | Model architecture, training pipeline | PyTorch, CNNs, training |
| **Data Lead** | Dataset processing, loading, augmentation | Data pipelines, visualization |
| **Research Lead** | Literature review, experiment design, writing | Paper writing, analysis |
| **Systems Lead** | Infrastructure, evaluation, deployment | Software engineering, DevOps |

---

## Deliverables

### Code
- [ ] Clean, documented PyTorch implementation
- [ ] Reproducible training scripts
- [ ] Evaluation pipeline with standard metrics
- [ ] Jupyter notebooks for analysis

### Research
- [ ] Literature review document
- [ ] Research gaps analysis
- [ ] Experimental results with ablation studies
- [ ] Conference paper draft (optional)

### Documentation
- [ ] README with setup instructions
- [ ] API documentation
- [ ] Result visualizations
- [ ] Presentation slides

---

## Success Criteria

| Criterion | Target | Stretch Goal |
|-----------|--------|--------------|
| Baseline accuracy | Match PlaNet | Exceed PlaNet |
| Data efficiency | 10x less data | 100x less data |
| Uncertainty calibration | ECE < 0.1 | ECE < 0.05 |
| Code quality | Working pipeline | Deployable API |
| Research output | Report | Conference submission |

---

*Generated for GeoNet Project*
