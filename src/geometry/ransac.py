"""
RANSAC-based robust homography estimation.

Random Sample Consensus (RANSAC) handles outliers in feature matches by
repeatedly drawing minimal subsets, fitting a homography, and counting
geometrically consistent inliers.  The hypothesis with the highest inlier
count is returned.
"""

import numpy as np
from src.geometry.homography import compute_homography, apply_homography


def count_inliers(H: np.ndarray, corners1: np.ndarray, corners2: np.ndarray,
                  matches: list, threshold: float = 5.0) -> list:
    """Count feature matches consistent with homography *H*.

    A match is an inlier when the reprojection error
    ``||H * p1 - p2||₂`` is below *threshold* pixels.

    Parameters
    ----------
    H : np.ndarray
        3 x 3 homography matrix.
    corners1, corners2 : np.ndarray
        2 x N corner coordinate arrays for images 1 and 2.
    matches : list of (int, int, float)
        Match tuples produced by the NNDR matcher.
    threshold : float
        Maximum allowed reprojection error (pixels) for an inlier.

    Returns
    -------
    list of (int, int)
        Index pairs of inlier matches.
    """
    inliers = []
    for idx1, idx2, _ in matches:
        p1 = corners1[:, idx1:idx1 + 1]
        p2_pred = apply_homography(H, p1)
        p2_actual = corners2[:, idx2:idx2 + 1]
        dist = np.sqrt(np.sum((p2_pred - p2_actual) ** 2))
        if dist < threshold:
            inliers.append((idx1, idx2))
    return inliers


def ransac_homography(corners1: np.ndarray, corners2: np.ndarray,
                      matches: list, num_iterations: int = 2000,
                      threshold: float = 5.0):
    """Estimate a robust homography via RANSAC.

    Parameters
    ----------
    corners1, corners2 : np.ndarray
        2 x N corner coordinate arrays.
    matches : list of (int, int, float)
        Putative matches to draw samples from.
    num_iterations : int
        Maximum number of RANSAC iterations.
    threshold : float
        Inlier reprojection error threshold (pixels).

    Returns
    -------
    best_H : np.ndarray or None
        Best-scoring 3 x 3 homography, or *None* if estimation failed.
    best_inliers : list of (int, int)
        Inlier match index pairs for *best_H*.
    """
    print(f"  Running RANSAC ({num_iterations} iterations, "
          f"threshold={threshold}px)...")

    best_H = None
    best_inliers = []
    matches_array = [(m[0], m[1]) for m in matches]

    for _ in range(num_iterations):
        if len(matches_array) < 4:
            break

        sample_idx = np.random.choice(len(matches_array), 4, replace=False)
        sample = [matches_array[i] for i in sample_idx]

        pts1 = np.array([[corners1[0, m[0]], corners1[1, m[0]]]
                          for m in sample]).T
        pts2 = np.array([[corners2[0, m[1]], corners2[1, m[1]]]
                          for m in sample]).T

        try:
            H = compute_homography(pts1, pts2)
        except Exception:
            continue

        inliers = count_inliers(H, corners1, corners2, matches, threshold)

        if len(inliers) > len(best_inliers):
            best_H = H
            best_inliers = inliers

        # Early exit when a dominant consensus is found
        if len(best_inliers) > 0.8 * len(matches):
            break

    n_matches = len(matches)
    n_in = len(best_inliers)
    rate = 100.0 * n_in / n_matches if n_matches else 0.0
    print(f"  Best H: {n_in} inliers / {n_matches} matches ({rate:.1f}%)")

    return best_H, best_inliers
