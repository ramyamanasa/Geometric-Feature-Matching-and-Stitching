# Classical Feature Matching and Stitching

A production-quality implementation of a classical geometric computer-vision pipeline for automatic feature detection, description, matching, and panoramic image stitching — built entirely from first principles using NumPy, scikit-image, and SciPy.

---

## Technical Overview

This project demonstrates end-to-end geometric vision engineering: given a pair of overlapping images, the system automatically identifies corresponding points, estimates the projective transformation relating them, and composites a seamless panorama. Every stage is implemented from scratch to expose the underlying mathematics rather than delegating to high-level library calls.

### Pipeline Stages

```
Input images
     │
     ▼
1. Harris Corner Detection   – structure-tensor eigenvalue analysis
     │
     ▼
2. Non-Maximal Suppression   – local-maximum filtering to reduce clustering
     │
     ▼
3. Descriptor Extraction     – 192-dim bias/gain-normalised RGB patch vectors
     │
     ▼
4. NNDR Feature Matching     – Lowe's ratio test on squared-Euclidean distances
     │
     ▼
5. RANSAC Homography         – robust DLT estimation with outlier rejection
     │
     ▼
6. Inverse Warping + Blend   – bilinear interpolation, average-blend composite
     │
     ▼
Panorama output
```

### Key Technical Concepts

**Harris Corner Detection** – The second-moment (structure) tensor captures local gradient behaviour. Corners correspond to regions where both tensor eigenvalues are large, indicating strong intensity variation in multiple directions. The Harris approximation avoids explicit eigendecomposition via the `det(M) − k·tr(M)²` response.

**Non-Maximal Suppression** – Dense Harris detections cluster around high-contrast edges. A sliding-window local-maximum filter retains only the single strongest response per neighbourhood, distributing keypoints more uniformly across the image.

**MOPS-Style Descriptors** – A 40×40 RGB patch is sampled around each keypoint, downsampled to 8×8×3 (192-dimensional), and bias/gain normalised to achieve invariance to affine illumination changes. Using RGB rather than grayscale preserves colour-discriminative information, improving matching in chromatic scenes.

**NNDR Matching (Lowe's Ratio Test)** – For each descriptor, the two nearest neighbours in the other image are found. A match is accepted only when `d(1NN) / d(2NN) < τ` (τ = 0.7), ensuring the best match is distinctively better than the runner-up and reducing false positives in repetitive or ambiguous regions.

**Direct Linear Transform (DLT) Homography** – Four point correspondences yield eight linear equations in nine homography unknowns. The system is assembled into a matrix `A` and the null-space solution is extracted as the last right-singular vector via SVD.

**RANSAC** – Random Sample Consensus iteratively draws minimal four-point subsets, estimates candidate homographies, and counts geometrically consistent inliers. The hypothesis maximising the inlier count survives; remaining matches are classified as outliers and discarded. Convergence is fast because the inlier fraction for well-overlapping image pairs is typically above 85 %.

**Inverse Warping with Bilinear Interpolation** – To avoid holes in the output, each destination pixel is mapped back through `H⁻¹` to find its source location. Sub-pixel samples are resolved by bilinear interpolation over the four nearest source pixels.

---

## Installation

**Prerequisites:** Python 3.8+

```bash
git clone https://github.com/your-username/classical-feature-matching-and-stitching.git
cd classical-feature-matching-and-stitching

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

All outputs are written to `results/<scene>/`.

### Command-line options

```bash
# Custom config
python run_pipeline.py --config configs/default.yaml

# Process a subset of scenes
python run_pipeline.py --scenes mural teddy

# Skip panorama stitching (run only stages 1–4)
python run_pipeline.py --no-panorama
```

### Using your own images

1. Add your images to the `examples/` directory.
2. Edit `configs/default.yaml` to add a new scene entry:

```yaml
scenes:
  - name: my_scene
    img1: "examples/my_scene_left.jpg"
    img2: "examples/my_scene_right.jpg"
```

3. Run `python run_pipeline.py --scenes my_scene`.

> **Capture tips:** For reliable stitching, rotate the camera about its optical centre (avoid translation), lock the exposure, and ensure 30–40 % overlap between frames.

---

## Repository Structure

```
classical-feature-matching-and-stitching/
├── run_pipeline.py          # CLI entrypoint – runs the full pipeline
├── requirements.txt         # Python dependencies
├── .gitignore
│
├── configs/
│   └── default.yaml         # All tunable parameters
│
├── src/
│   ├── detectors/
│   │   └── harris.py        # Harris corner detection + SSD distance
│   ├── suppression/
│   │   └── nms.py           # Non-maximal suppression
│   ├── descriptors/
│   │   └── mops.py          # RGB patch descriptor extraction
│   ├── matching/
│   │   └── nndr.py          # Lowe's ratio-test matcher
│   ├── geometry/
│   │   ├── homography.py    # DLT homography estimation
│   │   └── ransac.py        # RANSAC robust fitting
│   ├── stitching/
│   │   └── panorama.py      # Inverse warping + blend compositing
│   └── utils/
│       ├── image_io.py      # Image loading / colour conversion
│       └── visualization.py # All matplotlib figure generation
│
├── examples/                # Input image pairs (compressed JPEGs)
└── results/                 # Pipeline outputs (git-ignored)
```

---

## Results

Pipeline performance across three test scenes (640×480 images, default parameters):

| Scene | Harris (img1/2) | After NMS (img1/2) | NNDR Matches | RANSAC Inliers | Inlier Rate |
|-------|----------------|--------------------|--------------|----------------|-------------|
| Mural | 26 371 / 26 230 | 4 898 / 4 887 | 1 825 | 1 623 | **88.9 %** |
| Teddy | 26 686 / 26 931 | 4 856 / 4 792 | 2 321 | ~2 150 | **92.6 %** |
| North | 5 630 / 5 310   | 1 364 / 1 194 |   755 |   ~689 | **91.3 %** |

High inlier rates (>88 %) demonstrate that the NNDR matcher produces clean correspondences before geometric verification, and that RANSAC correctly identifies the dominant homography even when a small fraction of outliers remain.

### Output artefacts per scene

| File | Description |
|------|-------------|
| `step1_harris.jpg` | Original images annotated with detected Harris corners |
| `step2_nms.jpg` | Before / after NMS comparison |
| `step4_nndr.jpg` | NNDR ratio histogram with acceptance threshold |
| `step4_top5.jpg` | Top-5 match triplets: query patch · 1-NN · 2-NN |
| `step4_matches.jpg` | Side-by-side image pair with top-50 match lines |
| `panorama.jpg` | Stitched panorama |

---

## Configuration Reference

All parameters are exposed in `configs/default.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `harris.edge_discard` | 20 | Border strip (px) excluded from detection |
| `nms.window_size` | 11 | NMS suppression window side length (px) |
| `descriptor.patch_size` | 40 | Raw patch size around each keypoint (px) |
| `descriptor.desc_size` | 8 | Downsampled descriptor grid (→ 192-dim) |
| `matching.nndr_threshold` | 0.7 | Ratio test acceptance threshold |
| `ransac.num_iterations` | 2000 | Maximum RANSAC iterations |
| `ransac.inlier_threshold` | 5.0 | Reprojection error for inlier classification (px) |
| `ransac.min_inliers` | 10 | Minimum inliers required to attempt stitching |

---

## Potential Extensions

- **Rotation-invariant descriptors** – rotate the patch to align with the dominant gradient orientation before sampling (full SIFT-style invariance).
- **Adaptive NMS** – enforce a minimum radius between retained corners to guarantee spatial coverage.
- **Multi-image mosaicking** – chain pairwise homographies to stitch sequences of three or more frames into a wide panorama.
- **Laplacian / feathering blending** – replace simple averaging with multi-band blending to suppress seams at exposure boundaries.
- **Learned feature backbone** – swap the MOPS descriptor for a CNN embedding (e.g. SuperPoint) while retaining the NNDR + RANSAC matching stack.
- **GPU acceleration** – vectorise the inner loops in the warping and descriptor extraction stages with CuPy for real-time throughput.

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| NumPy | ≥ 1.24 | Array computation throughout |
| Pillow | ≥ 10.0 | Image I/O |
| scikit-image | ≥ 0.21 | Harris response, `peak_local_max`, `resize` |
| SciPy | ≥ 1.11 | Pairwise distance computation |
| Matplotlib | ≥ 3.7 | Result visualisation |
| PyYAML | ≥ 6.0 | Configuration loading |
