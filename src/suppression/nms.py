"""
Non-maximal suppression (NMS) for corner response maps.

Retains only corners that are a strict local maximum within a square
neighbourhood, reducing dense clusters to single representative points.
"""

import numpy as np


def non_maximal_suppression(h: np.ndarray, corners: np.ndarray,
                             window_size: int = 11) -> np.ndarray:
    """Filter corners so only local maxima within a sliding window survive.

    Parameters
    ----------
    h : np.ndarray
        2-D corner response map (same shape as the source image).
    corners : np.ndarray
        2 x N array of (row, col) corner coordinates to filter.
    window_size : int
        Side length of the square suppression window (pixels).

    Returns
    -------
    np.ndarray
        2 x M array of surviving (row, col) corner coordinates (M <= N).
    """
    filtered = []
    half = window_size // 2

    for i in range(corners.shape[1]):
        y, x = corners[0, i], corners[1, i]

        y_min = max(0, y - half)
        y_max = min(h.shape[0], y + half + 1)
        x_min = max(0, x - half)
        x_max = min(h.shape[1], x + half + 1)

        window = h[y_min:y_max, x_min:x_max]

        if h[y, x] == np.max(window):
            filtered.append([y, x])

    if filtered:
        return np.array(filtered).T
    return np.empty((2, 0), dtype=int)
