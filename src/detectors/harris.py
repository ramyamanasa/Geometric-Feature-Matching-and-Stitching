"""
Harris corner detector and SSD distance utility.

Detects corners by analysing the structure tensor: regions where image
gradients change strongly in multiple directions yield large eigenvalues of
the tensor and are marked as corners.
"""

import numpy as np
from skimage.feature import corner_harris, peak_local_max


def get_harris_corners(im, edge_discard: int = 20):
    """Detect Harris corners in a grayscale image.

    Parameters
    ----------
    im : np.ndarray
        Grayscale image (H x W), float in [0, 1].
    edge_discard : int
        Width of the border strip (pixels) from which corners are excluded.
        Must be >= 20.

    Returns
    -------
    h : np.ndarray
        Corner response map, same shape as *im*.
    coords : np.ndarray
        2 x N array of (row, col) coordinates for the surviving corners.
    """
    assert edge_discard >= 20, "edge_discard must be at least 20 pixels"

    # Compute Harris corner response
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # Discard corners within the border strip
    mask = (
        (coords[:, 0] > edge_discard) &
        (coords[:, 0] < im.shape[0] - edge_discard) &
        (coords[:, 1] > edge_discard) &
        (coords[:, 1] < im.shape[1] - edge_discard)
    )
    coords = coords[mask].T
    return h, coords


def dist_ssd(x, c):
    """Compute squared Euclidean (SSD) distances between two sets of vectors.

    Parameters
    ----------
    x : np.ndarray
        M x D matrix of query vectors.
    c : np.ndarray
        L x D matrix of reference vectors.

    Returns
    -------
    np.ndarray
        M x L matrix of squared distances.
    """
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Vector dimensions do not match"

    return (
        (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T +
        np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0) -
        2 * np.inner(x, c)
    )
