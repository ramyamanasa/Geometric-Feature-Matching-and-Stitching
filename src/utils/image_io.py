"""
Image I/O helpers.

Thin wrappers around PIL and scikit-image for consistent image loading,
colour conversion, and output directory management across the pipeline.
"""

import os
import numpy as np
from PIL import Image
from skimage.color import rgb2gray


def load_image_pair(path1: str, path2: str):
    """Load a stereo image pair as uint8 RGB arrays.

    Parameters
    ----------
    path1, path2 : str
        File paths to the two images.

    Returns
    -------
    img1, img2 : np.ndarray
        H x W x 3 uint8 arrays.
    """
    img1 = np.array(Image.open(path1).convert("RGB"))
    img2 = np.array(Image.open(path2).convert("RGB"))
    return img1, img2


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB image to a float64 grayscale image in [0, 1].

    Parameters
    ----------
    img : np.ndarray
        H x W x 3 uint8 image.

    Returns
    -------
    np.ndarray
        H x W float64 image.
    """
    return rgb2gray(img)


def ensure_output_dirs(scenes: list, base: str = "results") -> None:
    """Create output subdirectories for each scene name.

    Parameters
    ----------
    scenes : list of str
        Scene identifiers (one subdirectory is created per scene).
    base : str
        Root output directory.
    """
    for scene in scenes:
        os.makedirs(os.path.join(base, scene), exist_ok=True)
