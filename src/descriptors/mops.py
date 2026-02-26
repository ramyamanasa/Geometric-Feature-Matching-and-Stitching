"""
Multi-scale Oriented Patch (MOPS-style) descriptor extraction.

For each detected corner, a 40x40 RGB patch is sampled, downsampled to
8x8x3 with anti-aliasing, flattened to a 192-dimensional vector, and
bias/gain normalised to achieve invariance to illumination changes.
"""

import numpy as np
from skimage.transform import resize


def extract_descriptors(img: np.ndarray, corners: np.ndarray,
                         patch_size: int = 40, desc_size: int = 8):
    """Extract normalised RGB patch descriptors for a set of keypoints.

    Parameters
    ----------
    img : np.ndarray
        H x W x 3 uint8 RGB image.
    corners : np.ndarray
        2 x N array of (row, col) corner coordinates.
    patch_size : int
        Side length (pixels) of the raw patch extracted around each corner.
    desc_size : int
        Side length of the downsampled descriptor grid.

    Returns
    -------
    descriptors : np.ndarray
        M x (desc_size * desc_size * 3) float64 descriptor matrix.
    valid_corners : np.ndarray
        2 x M array of corner coordinates for which descriptors were computed.
        Corners too close to the image border are dropped.
    """
    descriptors = []
    valid_corners = []
    half = patch_size // 2

    for i in range(corners.shape[1]):
        y, x = int(corners[0, i]), int(corners[1, i])

        # Skip keypoints whose patch would extend outside the image
        if (y - half < 0 or y + half > img.shape[0] or
                x - half < 0 or x + half > img.shape[1]):
            continue

        # Extract raw RGB patch
        patch = img[y - half:y + half, x - half:x + half]

        # Downsample with anti-aliasing → desc_size x desc_size x 3
        desc = resize(patch, (desc_size, desc_size), anti_aliasing=True)

        # Flatten and bias/gain normalise
        desc = desc.flatten()
        desc = (desc - np.mean(desc)) / (np.std(desc) + 1e-8)

        descriptors.append(desc)
        valid_corners.append([y, x])

    if valid_corners:
        return np.array(descriptors), np.array(valid_corners).T
    return np.empty((0, desc_size * desc_size * 3)), np.empty((2, 0))
