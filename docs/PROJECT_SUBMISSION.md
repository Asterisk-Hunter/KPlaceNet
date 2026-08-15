# Project Submission Materials

---

## Title Options

### Top Picks (Ranked)

| Rank | Title | Why It Works |
|------|-------|--------------|
| 1 | **GeoNet** | Clean, professional, memorable. 5 letters. |
| 2 | **WhereLens** | Evocative — "looking through a lens to find where" |
| 3 | **PlaceNet** | Direct, technical, easy to understand |
| 4 | **LocateAI** | Simple, clear purpose |
| 5 | **GeoScope** | Technical but approachable |

### Creative Options

| Title | Vibe |
|-------|------|
| **SnapLocal** | Casual, app-like |
| **PhotoPin** | Simple, memorable |
| **GeoGuessr AI** | Fun reference to the game |
| **ViewFinder** | Classic photography term |

### Formal/Academic Options

| Title | Use Case |
|-------|----------|
| **Visual Geo-Localization Network** | Research paper |
| **CNN-Based Photo Geolocation** | Technical report |
| **Adaptive Geographic Cell Learning** | If focusing on cell construction |

---

## Project Description (For Form)

### Option 1: Concise (50 words)

> GeoNet is a deep learning system that predicts the geographic location of any photograph using only its visual content. By combining convolutional neural networks with adaptive spatial partitioning, the model learns to identify visual cues—landmarks, vegetation, architecture, and terrain—to estimate where an image was captured without relying on GPS metadata.

---

### Option 2: Standard (100 words)

> GeoNet explores photo geolocation through deep learning, answering the question: "Where was this photo taken?" using only pixel information. Building on PlaNet (Weyand et al., 2016), our system classifies images into learned geographic regions using convolutional neural networks trained on millions of geotagged photographs. We investigate three research directions: adaptive geographic cell construction that learns optimal spatial partitions, data-efficient training via transfer learning that reduces required training data by 10x, and uncertainty quantification that provides calibrated confidence estimates for predictions. The project demonstrates that deep networks can achieve superhuman geolocation accuracy while remaining practical for real-world deployment.

---

### Option 3: Technical (150 words)

> GeoNet is a research project advancing visual geo-localization through deep convolutional neural networks. The system takes a single photograph as input and predicts its geographic coordinates (latitude, longitude) without GPS metadata, relying entirely on learned visual features. We address three limitations of current approaches: (1) static geographic cell boundaries that poorly represent real-world image distributions, (2) massive data requirements exceeding 100M training images, and (3) lack of calibrated uncertainty estimates for predictions. Our methodology employs adaptive cell construction via learned clustering, transfer learning from CLIP and Places365 for data efficiency, and conformal prediction for uncertainty quantification. Evaluated on the OpenStreetView-5M benchmark (CVPR 2024), our approach targets improved accuracy in under-represented regions while providing reliable confidence scores essential for safety-critical applications.

---

### Option 4: Accessible (For non-technical audience)

> Have you ever looked at a photo and wondered where it was taken? GeoNet does exactly that—automatically. Our system analyzes visual clues in any photograph—the style of buildings, types of vegetation, road signs, terrain, and even the quality of light—and predicts its location anywhere on Earth. Think of it as giving a computer the same intuition humans have about recognizing places, but at global scale. Unlike GPS, which requires hardware sensors, GeoNet works with any existing photo, making it useful for organizing photo collections, verifying image authenticity, and helping travelers rediscover places they've visited.

---

## Complete Project Description (Copy-Paste Ready)

```
PROJECT TITLE: GeoNet

PROJECT DESCRIPTION:

GeoNet is a deep learning system that predicts the geographic location of photographs using only their visual content. The project addresses a fundamental computer vision challenge: given an arbitrary image, estimate where on Earth it was captured without relying on GPS metadata or EXIF data.

Building on the PlaNet architecture (Weyand et al., 2016), our system classifies images into learned geographic regions using convolutional neural networks trained on millions of geotagged photographs from open datasets. The core innovation lies in three research directions that improve upon existing approaches:

1. ADAPTIVE GEOGRAPHIC CELLS: Instead of using fixed quad-tree spatial partitions that poorly represent real-world image distributions, we learn optimal cell boundaries through clustering algorithms that adapt to actual data density, improving accuracy in both urban and rural regions.

2. DATA-EFFICIENT TRAINING: Current methods require 100M+ training images. We investigate transfer learning from pre-trained vision models (CLIP, Places365) and self-supervised pre-training to achieve comparable performance with 10x less labeled data, making the approach accessible to resource-constrained environments.

3. UNCERTAINTY QUANTIFICATION: Real-world applications require knowing when predictions are unreliable. We implement conformal prediction to provide calibrated confidence estimates, enabling the system to abstain from predictions when confidence is low—essential for safety-critical deployment.

EVALUATION: We benchmark on OpenStreetView-5M (CVPR 2024), a dataset of 5.1M geo-referenced street view images spanning 225 countries, with standard metrics including within-distance accuracy at 1km, 25km, 200km, and continental scales.

IMPACT: The system has practical applications in travel and tourism (automatic photo organization), content verification (geolocating social media images), forensics (investigative geolocation), and autonomous systems (visual localization). By improving data efficiency and adding uncertainty estimates, we make photo geolocation more accessible and reliable for real-world deployment.

TEAM: 4 undergraduate students collaborating on machine learning, computer vision, and research methodology.

TECHNOLOGIES: PyTorch, CNNs (ResNet/EfficientNet), transfer learning, conformal prediction, geospatial data processing.
```

---

## Dataset Links (Direct Access)

### Primary Datasets

| Dataset | Direct Link | Size | Access Method |
|---------|-------------|------|---------------|
| **OpenStreetView-5M** | https://huggingface.co/datasets/osv5m/osv5m | 5.1M images | HuggingFace download |
| **YFCC100M** | https://multimediacommons.wordpress.com/yfcc100m-core-dataset/ | 100M images | AWS S3 (free) |
| **IM2GPS** | http://graphics.cs.cmu.edu/projects/im2gps/ | 237 test images | Direct download |
| **IM2GPS3k** | https://github.com/TIBHannover/GeoEstimation | 2,997 test images | GitHub |

### Benchmark Datasets

| Dataset | Direct Link | Size | Use Case |
|---------|-------------|------|----------|
| **Places365** | https://github.com/csailvision/places365 | 1.8M images | Transfer learning |
| **CVUSA** | https://mvrl.cse.wustl.edu/datasets/cvusa/ | 35K pairs | Cross-view matching |
| **CVACT** | https://github.com/YujiaoShi/cross_view_localization_CVFT | 44K pairs | Cross-view matching |

### Easy-Access Datasets

| Dataset | Direct Link | Size | Notes |
|---------|-------------|------|-------|
| **Flickr-Geo** | https://huggingface.co/datasets/do-me/Flickr-Geo | Variable | Quick experiments |
| **Kaggle Geotagged** | https://www.kaggle.com/datasets/ifeanyichukwunwobodo/tokyo-geotagged-flickr-images | Sample | Tokyo subset |

### Quick Start Commands

```bash
# Download OSV-5M (recommended primary dataset)
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('osv5m/osv5m', local_dir='datasets/osv5m')"

# Download IM2GPS test set
wget http://graphics.cs.cmu.edu/projects/im2gps/data/im2gps_testset.zip

# Clone GeoEstimation (includes IM2GPS3k)
git clone https://github.com/TIBHannover/GeoEstimation.git
```

---

## Form Submission Summary

**Project Title:** GeoNet

**One-Line Description:** A deep learning system that predicts where a photo was taken using only its visual content.

**Key Technologies:** PyTorch, Convolutional Neural Networks, Transfer Learning, Geospatial Analysis

**Datasets:** OpenStreetView-5M (5.1M images, CVPR 2024), YFCC100M, IM2GPS3k

**Team Size:** 4 undergraduate students

**Expected Duration:** 8-10 weeks

---

*Generated for Project Submission*
