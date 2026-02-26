"""
Panorama construction via inverse warping and alpha blending.

Given a homography that maps image 1 into the coordinate frame of image 2,
the canvas bounds are computed analytically, image 1 is inverse-warped with
bilinear interpolation, and the two images are composited with simple
averaging in the overlap region.
"""

import numpy as np


def warp_image(img: np.ndarray, H: np.ndarray, output_shape: tuple) -> np.ndarray:
    """Warp *img* into a new canvas using inverse homography mapping.

    Inverse warping iterates over every output pixel, computes the
    corresponding source location via H⁻¹, and samples the source image
    with bilinear interpolation.  Pixels that map outside the source
    image are left as zero (black).

    Parameters
    ----------
    img : np.ndarray
        H x W x 3 uint8 source image.
    H : np.ndarray
        3 x 3 homography mapping *img* coordinates to output coordinates.
    output_shape : tuple of (int, int)
        (height, width) of the destination canvas.

    Returns
    -------
    np.ndarray
        Warped image with the same dtype as *img*, shape (*output_shape*, 3).
    """
    h_out, w_out = output_shape
    warped = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    H_inv = np.linalg.inv(H)

    for y_out in range(h_out):
        for x_out in range(w_out):
            p_out = np.array([x_out, y_out, 1.0])
            p_in = H_inv @ p_out
            p_in = p_in / p_in[2]
            x_in, y_in = p_in[0], p_in[1]

            if 0 <= x_in < img.shape[1] - 1 and 0 <= y_in < img.shape[0] - 1:
                x0, y0 = int(x_in), int(y_in)
                dx = x_in - x0
                dy = y_in - y0

                warped[y_out, x_out] = (
                    img[y0,     x0    ] * (1 - dx) * (1 - dy) +
                    img[y0,     x0 + 1] *      dx  * (1 - dy) +
                    img[y0 + 1, x0    ] * (1 - dx) *      dy  +
                    img[y0 + 1, x0 + 1] *      dx  *      dy
                )

    return warped


def create_panorama(img1: np.ndarray, img2: np.ndarray,
                    H: np.ndarray) -> np.ndarray:
    """Stitch *img1* and *img2* into a single panorama.

    The homography *H* maps pixel coordinates in *img1* to the coordinate
    frame of *img2*.  The output canvas is sized to contain both images;
    overlapping pixels are averaged.

    Parameters
    ----------
    img1, img2 : np.ndarray
        H x W x 3 uint8 images.
    H : np.ndarray
        3 x 3 homography mapping img1 → img2 coordinate frame.

    Returns
    -------
    np.ndarray
        Stitched panorama as a uint8 RGB image.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    print("  Computing panorama canvas bounds...")

    # Project all four corners of img1 into img2's coordinate frame
    corners_img1 = np.array([
        [0,  0,  1],
        [w1, 0,  1],
        [w1, h1, 1],
        [0,  h1, 1],
    ], dtype=float).T

    warped_corners = H @ corners_img1
    warped_corners /= warped_corners[2, :]

    all_x = np.concatenate([warped_corners[0, :], [0, w2, w2, 0]])
    all_y = np.concatenate([warped_corners[1, :], [0, 0,  h2, h2]])

    x_min = int(np.floor(np.min(all_x)))
    x_max = int(np.ceil(np.max(all_x)))
    y_min = int(np.floor(np.min(all_y)))
    y_max = int(np.ceil(np.max(all_y)))

    # Translate everything so all coordinates are positive
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0,  1    ],
    ], dtype=float)

    output_shape = (y_max - y_min, x_max - x_min)
    print(f"  Canvas size: {output_shape[1]} x {output_shape[0]} px")

    # Warp img1 onto the canvas
    print("  Warping image 1...")
    H_canvas = T @ H
    warped_img1 = warp_image(img1, H_canvas, output_shape)

    # Composite both images
    print("  Blending images...")
    canvas = np.zeros(output_shape + (3,), dtype=np.float32)
    weight = np.zeros(output_shape, dtype=np.float32)

    mask1 = np.any(warped_img1 > 0, axis=2).astype(np.float32)
    canvas += warped_img1.astype(np.float32) * mask1[:, :, np.newaxis]
    weight += mask1

    # Place img2 (offset by the translation)
    y_off, x_off = -y_min, -x_min
    img2_region = canvas[y_off:y_off + h2, x_off:x_off + w2]
    weight_region = weight[y_off:y_off + h2, x_off:x_off + w2]

    overlap = weight_region > 0
    for c in range(3):
        img2_region[:, :, c] = np.where(
            overlap,
            (img2_region[:, :, c] + img2[:, :, c]) / 2.0,  # average in overlap
            img2[:, :, c],                                   # direct copy elsewhere
        )

    canvas[y_off:y_off + h2, x_off:x_off + w2] = img2_region
    weight[y_off:y_off + h2, x_off:x_off + w2] = np.maximum(weight_region, 1)

    return np.clip(canvas, 0, 255).astype(np.uint8)
