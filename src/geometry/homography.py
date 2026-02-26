"""
Homography estimation from point correspondences.

A planar homography (projective transformation) maps points in one image to
corresponding points in another when the scene is planar or the camera
undergoes pure rotation.  The 3x3 matrix is estimated via Direct Linear
Transform (DLT) and solved with SVD.
"""

import numpy as np


def compute_homography(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """Estimate a 3x3 homography from four point correspondences.

    Uses the Direct Linear Transform (DLT): each correspondence contributes
    two linear equations in the nine homography entries.  With four points
    the system is exactly determined (up to scale) and solved via SVD.

    Parameters
    ----------
    pts1 : np.ndarray
        2 x 4 array of (row, col) source coordinates.
    pts2 : np.ndarray
        2 x 4 array of (row, col) destination coordinates.

    Returns
    -------
    H : np.ndarray
        3 x 3 homography matrix (normalised so H[2, 2] == 1) such that
        ``pts2 ≈ H @ pts1`` in homogeneous coordinates.
    """
    A = []
    for i in range(4):
        x1, y1 = pts1[1, i], pts1[0, i]   # col → x, row → y
        x2, y2 = pts2[1, i], pts2[0, i]

        A.append([-x1, -y1, -1,  0,   0,  0, x2 * x1, x2 * y1, x2])
        A.append([ 0,   0,  0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])

    A = np.array(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to a set of (row, col) coordinates.

    Parameters
    ----------
    H : np.ndarray
        3 x 3 homography matrix.
    points : np.ndarray
        2 x N array of (row, col) coordinates.

    Returns
    -------
    np.ndarray
        2 x N array of transformed (row, col) coordinates.
    """
    ones = np.ones((1, points.shape[1]))
    # Stack as [x, y, 1]  (x = col, y = row)
    homog = np.vstack([points[1:2, :], points[0:1, :], ones])

    transformed = H @ homog
    transformed = transformed / transformed[2:3, :]

    # Return as [row, col]
    return np.vstack([transformed[1:2, :], transformed[0:1, :]])
