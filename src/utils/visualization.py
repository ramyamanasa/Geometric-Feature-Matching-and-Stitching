"""
Visualization utilities for the feature-matching and stitching pipeline.

All functions save figures to disk rather than displaying them interactively,
making the module suitable for headless execution.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Step 1 – Harris corner detection
# ---------------------------------------------------------------------------

def save_harris_corners(img1: np.ndarray, img2: np.ndarray,
                         corners1: np.ndarray, corners2: np.ndarray,
                         scene: str, out_dir: str) -> None:
    """Save a 2x2 grid showing input images and detected Harris corners."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img1); axes[0, 0].set_title(f"{scene} – Image 1"); axes[0, 0].axis("off")
    axes[0, 1].imshow(img2); axes[0, 1].set_title(f"{scene} – Image 2"); axes[0, 1].axis("off")

    axes[1, 0].imshow(img1)
    axes[1, 0].plot(corners1[1], corners1[0], "r+", markersize=6, markeredgewidth=1)
    axes[1, 0].set_title(f"Harris corners ({corners1.shape[1]} detected)"); axes[1, 0].axis("off")

    axes[1, 1].imshow(img2)
    axes[1, 1].plot(corners2[1], corners2[0], "r+", markersize=6, markeredgewidth=1)
    axes[1, 1].set_title(f"Harris corners ({corners2.shape[1]} detected)"); axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, scene, "step1_harris.jpg"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 2 – Non-maximal suppression
# ---------------------------------------------------------------------------

def save_nms_comparison(img1: np.ndarray, img2: np.ndarray,
                         pre1: np.ndarray, pre2: np.ndarray,
                         post1: np.ndarray, post2: np.ndarray,
                         scene: str, out_dir: str, window_size: int) -> None:
    """Save a 2x2 grid comparing corners before and after NMS."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(img1)
    axes[0, 0].plot(pre1[1], pre1[0], "r+", markersize=4)
    axes[0, 0].set_title(f"Before NMS ({pre1.shape[1]} pts)"); axes[0, 0].axis("off")

    axes[0, 1].imshow(img2)
    axes[0, 1].plot(pre2[1], pre2[0], "r+", markersize=4)
    axes[0, 1].set_title(f"Before NMS ({pre2.shape[1]} pts)"); axes[0, 1].axis("off")

    axes[1, 0].imshow(img1)
    axes[1, 0].plot(post1[1], post1[0], "g+", markersize=8, markeredgewidth=2)
    axes[1, 0].set_title(f"After NMS ({post1.shape[1]} pts, w={window_size})"); axes[1, 0].axis("off")

    axes[1, 1].imshow(img2)
    axes[1, 1].plot(post2[1], post2[0], "g+", markersize=8, markeredgewidth=2)
    axes[1, 1].set_title(f"After NMS ({post2.shape[1]} pts, w={window_size})"); axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, scene, "step2_nms.jpg"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Step 4 – Feature matching
# ---------------------------------------------------------------------------

def save_nndr_histogram(all_ratios: np.ndarray, threshold: float,
                         scene: str, out_dir: str) -> None:
    """Save a histogram of NNDR values with the acceptance threshold marked."""
    plt.figure(figsize=(10, 5))
    plt.hist(all_ratios, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2,
                label=f"Threshold = {threshold}")
    plt.xlabel("NNDR (distance ratio)")
    plt.ylabel("Count")
    plt.title(f"{scene} – NNDR distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, scene, "step4_nndr.jpg"), dpi=150, bbox_inches="tight")
    plt.close()


def save_top_matches(desc1: np.ndarray, desc2: np.ndarray,
                      matches: list, scene: str, out_dir: str, n: int = 5) -> None:
    """Save a grid visualising the top-N matches as 8x8 RGB descriptor patches."""
    sorted_matches = sorted(matches, key=lambda x: x[2])[:n]
    if not sorted_matches:
        return

    fig, axes = plt.subplots(len(sorted_matches), 3,
                              figsize=(10, 3 * len(sorted_matches)))
    if len(sorted_matches) == 1:
        axes = axes.reshape(1, -1)

    for row, (i, j, ratio) in enumerate(sorted_matches):
        patch1 = np.clip(desc1[i].reshape(8, 8, 3), 0, 1)
        patch2 = np.clip(desc2[j].reshape(8, 8, 3), 0, 1)

        dists = cdist(desc1[i:i + 1], desc2, metric="sqeuclidean")[0]
        nn2_idx = np.argsort(dists)[1]
        patch_nn2 = np.clip(desc2[nn2_idx].reshape(8, 8, 3), 0, 1)

        axes[row, 0].imshow(patch1); axes[row, 0].set_title(f"Match {row+1}: Image 1"); axes[row, 0].axis("off")
        axes[row, 1].imshow(patch2); axes[row, 1].set_title(f"1-NN (ratio={ratio:.3f})"); axes[row, 1].axis("off")
        axes[row, 2].imshow(patch_nn2); axes[row, 2].set_title("2-NN"); axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, scene, "step4_top5.jpg"), dpi=150, bbox_inches="tight")
    plt.close()


def save_match_lines(img1: np.ndarray, img2: np.ndarray,
                      corners1: np.ndarray, corners2: np.ndarray,
                      matches: list, scene: str, out_dir: str,
                      n: int = 50) -> None:
    """Save a side-by-side image with lines connecting the top-N matches."""
    sorted_matches = sorted(matches, key=lambda x: x[2])[:n]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1 + w2] = img2

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(combined)

    for idx, (i, j, ratio) in enumerate(sorted_matches):
        y1, x1 = corners1[:, i]
        y2, x2 = corners2[:, j]
        ax.plot([x1, x2 + w1], [y1, y2], "g-", linewidth=1, alpha=0.6)
        ax.plot(x1, y1, "ro", markersize=5)
        ax.plot(x2 + w1, y2, "ro", markersize=5)
        if idx < 20:
            kw = dict(color="yellow", fontsize=8, weight="bold",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))
            ax.text(x1, y1 - 10, str(idx + 1), **kw)
            ax.text(x2 + w1, y2 - 10, str(idx + 1), **kw)

    ax.set_title(f"Feature matches – top {len(sorted_matches)} of {len(matches)}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, scene, "step4_matches.jpg"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Panorama
# ---------------------------------------------------------------------------

def save_panorama(panorama: np.ndarray, scene: str, out_dir: str,
                   n_inliers: int, n_matches: int) -> None:
    """Save the stitched panorama image."""
    rate = 100.0 * n_inliers / n_matches if n_matches else 0.0
    plt.figure(figsize=(20, 10))
    plt.imshow(panorama)
    plt.title(f"{scene} panorama  |  {n_inliers}/{n_matches} inliers ({rate:.1f}%)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, scene, "panorama.jpg"), dpi=150, bbox_inches="tight")
    plt.close()
