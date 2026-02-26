"""
Nearest-Neighbour Distance Ratio (NNDR) feature matching.

Implements Lowe's ratio test: a match is accepted only when the distance to
the best candidate is significantly smaller than the distance to the second-
best candidate, ensuring distinctiveness.
"""

import numpy as np
from scipy.spatial.distance import cdist


def match_features(desc1: np.ndarray, desc2: np.ndarray,
                   threshold: float = 0.7):
    """Match descriptors from two images using Lowe's ratio test.

    For each descriptor in *desc1*, the two nearest neighbours in *desc2* are
    found.  The match is kept only if
    ``dist(1NN) / dist(2NN) < threshold``.

    Parameters
    ----------
    desc1 : np.ndarray
        M x D descriptor matrix for image 1.
    desc2 : np.ndarray
        N x D descriptor matrix for image 2.
    threshold : float
        NNDR ratio threshold in (0, 1).  Lower values are more selective.

    Returns
    -------
    matches : list of (int, int, float)
        Each entry is ``(idx1, idx2, ratio)`` — indices into *desc1* /
        *desc2* and the acceptance ratio.
    all_ratios : np.ndarray
        NNDR ratios for every descriptor in *desc1* (useful for diagnostics).
    """
    # Pairwise squared-Euclidean distance matrix  (M x N)
    distances = cdist(desc1, desc2, metric='sqeuclidean')

    matches = []
    all_ratios = []

    for i in range(desc1.shape[0]):
        sorted_idx = np.argsort(distances[i])

        nn1_idx = sorted_idx[0]
        nn2_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]

        dist1 = distances[i, nn1_idx]
        dist2 = distances[i, nn2_idx]

        ratio = dist1 / (dist2 + 1e-8)
        all_ratios.append(ratio)

        if ratio < threshold:
            matches.append((i, nn1_idx, ratio))

    return matches, np.array(all_ratios)
