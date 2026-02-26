#!/usr/bin/env python3
"""
run_pipeline.py – Classical Feature Matching & Panoramic Stitching Pipeline

Loads configuration from configs/default.yaml (or a user-specified file),
runs the full five-stage pipeline on every scene defined in the config, and
writes all visualisations to the results directory.

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --config configs/default.yaml
    python run_pipeline.py --scenes mural teddy
    python run_pipeline.py --no-panorama
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

# Ensure the project root is on the Python path when invoked directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detectors.harris import get_harris_corners
from src.suppression.nms import non_maximal_suppression
from src.descriptors.mops import extract_descriptors
from src.matching.nndr import match_features
from src.geometry.ransac import ransac_homography
from src.stitching.panorama import create_panorama
from src.utils.image_io import load_image_pair, to_grayscale, ensure_output_dirs
from src.utils.visualization import (
    save_harris_corners,
    save_nms_comparison,
    save_nndr_histogram,
    save_top_matches,
    save_match_lines,
    save_panorama,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def banner(text: str) -> None:
    width = 60
    print("\n" + "─" * width)
    print(f"  {text}")
    print("─" * width)


# ──────────────────────────────────────────────────────────────────────────────
# Per-scene pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_scene(scene_cfg: dict, cfg: dict, results_dir: str,
              run_panorama: bool) -> dict:
    """Execute the full pipeline for a single scene and return summary metrics."""
    name = scene_cfg["name"]
    banner(f"Scene: {name}")

    # ── 1. Load images ────────────────────────────────────────────────────────
    img1, img2 = load_image_pair(scene_cfg["img1"], scene_cfg["img2"])
    gray1 = to_grayscale(img1)
    gray2 = to_grayscale(img2)
    print(f"  Loaded images  {img1.shape[1]}×{img1.shape[0]}  /  "
          f"{img2.shape[1]}×{img2.shape[0]}")

    # ── 2. Harris corner detection ────────────────────────────────────────────
    print("  Stage 1 – Harris corner detection")
    edge = cfg["harris"]["edge_discard"]
    h1, corners1 = get_harris_corners(gray1, edge_discard=edge)
    h2, corners2 = get_harris_corners(gray2, edge_discard=edge)
    print(f"    Image 1: {corners1.shape[1]} corners")
    print(f"    Image 2: {corners2.shape[1]} corners")
    save_harris_corners(img1, img2, corners1, corners2, name, results_dir)

    # ── 3. Non-maximal suppression ────────────────────────────────────────────
    print("  Stage 2 – Non-maximal suppression")
    win = cfg["nms"]["window_size"]
    nms1 = non_maximal_suppression(h1, corners1, window_size=win)
    nms2 = non_maximal_suppression(h2, corners2, window_size=win)
    print(f"    Image 1: {corners1.shape[1]} → {nms1.shape[1]} corners")
    print(f"    Image 2: {corners2.shape[1]} → {nms2.shape[1]} corners")
    save_nms_comparison(img1, img2, corners1, corners2, nms1, nms2,
                        name, results_dir, win)

    # ── 4. Descriptor extraction ──────────────────────────────────────────────
    print("  Stage 3 – Descriptor extraction (MOPS-style RGB)")
    ps = cfg["descriptor"]["patch_size"]
    ds = cfg["descriptor"]["desc_size"]
    desc1, kp1 = extract_descriptors(img1, nms1, patch_size=ps, desc_size=ds)
    desc2, kp2 = extract_descriptors(img2, nms2, patch_size=ps, desc_size=ds)
    print(f"    Image 1: {desc1.shape[0]} descriptors  ({desc1.shape[1]}-dim)")
    print(f"    Image 2: {desc2.shape[0]} descriptors  ({desc2.shape[1]}-dim)")

    # ── 5. Feature matching (NNDR) ────────────────────────────────────────────
    print("  Stage 4 – Feature matching (NNDR)")
    thr = cfg["matching"]["nndr_threshold"]
    matches, all_ratios = match_features(desc1, desc2, threshold=thr)
    print(f"    {len(matches)} matches accepted  "
          f"(threshold={thr}, NNDR range "
          f"[{np.min(all_ratios):.3f}, {np.max(all_ratios):.3f}])")
    save_nndr_histogram(all_ratios, thr, name, results_dir)
    save_top_matches(desc1, desc2, matches, name, results_dir,
                     n=cfg["visualization"]["top_matches"])
    save_match_lines(img1, img2, kp1, kp2, matches, name, results_dir,
                     n=cfg["visualization"]["draw_matches"])

    metrics = {
        "scene": name,
        "harris_1": corners1.shape[1],
        "harris_2": corners2.shape[1],
        "nms_1": nms1.shape[1],
        "nms_2": nms2.shape[1],
        "descriptors_1": desc1.shape[0],
        "descriptors_2": desc2.shape[0],
        "nndr_matches": len(matches),
        "inliers": None,
        "inlier_rate": None,
    }

    # ── 6. RANSAC homography + panorama ──────────────────────────────────────
    if run_panorama and len(matches) >= 4:
        print("  Stage 5 – RANSAC homography + panorama stitching")
        r_cfg = cfg["ransac"]
        H, inliers = ransac_homography(
            kp1, kp2, matches,
            num_iterations=r_cfg["num_iterations"],
            threshold=r_cfg["inlier_threshold"],
        )

        if H is not None and len(inliers) >= r_cfg["min_inliers"]:
            panorama = create_panorama(img1, img2, H)
            save_panorama(panorama, name, results_dir,
                          len(inliers), len(matches))
            print(f"  Saved panorama → results/{name}/panorama.jpg")
            metrics["inliers"] = len(inliers)
            metrics["inlier_rate"] = len(inliers) / len(matches)
        else:
            n = len(inliers) if inliers else 0
            print(f"  Panorama skipped – insufficient inliers ({n})")
    elif run_panorama:
        print(f"  Panorama skipped – too few matches ({len(matches)})")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Classical feature matching and panoramic stitching pipeline"
    )
    p.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    p.add_argument(
        "--scenes", nargs="*", default=None,
        help="Subset of scene names to process (default: all scenes in config)",
    )
    p.add_argument(
        "--no-panorama", action="store_true",
        help="Skip RANSAC homography estimation and panorama stitching",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)
    cfg = load_config(args.config)

    results_dir = cfg.get("results_dir", "results")
    scenes = cfg.get("scenes", [])

    # Optionally restrict to a subset of scenes
    if args.scenes:
        scenes = [s for s in scenes if s["name"] in args.scenes]
        if not scenes:
            print(f"[ERROR] No matching scenes found for: {args.scenes}")
            sys.exit(1)

    # Validate that image files exist
    for sc in scenes:
        for key in ("img1", "img2"):
            if not os.path.exists(sc[key]):
                print(f"[ERROR] Image not found: {sc[key]}")
                sys.exit(1)

    # Create output directories
    ensure_output_dirs([s["name"] for s in scenes], base=results_dir)

    run_panorama = not args.no_panorama

    banner("Classical Feature Matching & Stitching Pipeline")
    print(f"  Config  : {args.config}")
    print(f"  Scenes  : {[s['name'] for s in scenes]}")
    print(f"  Panorama: {'enabled' if run_panorama else 'disabled'}")
    print(f"  Output  : {results_dir}/")

    t0 = time.time()
    all_metrics = []

    for sc in scenes:
        metrics = run_scene(sc, cfg, results_dir, run_panorama)
        all_metrics.append(metrics)

    # ── Summary table ──────────────────────────────────────────────────────
    banner("Results Summary")
    header = f"{'Scene':<10} {'Harris1':>9} {'Harris2':>9} {'NMS1':>7} {'NMS2':>7} {'Desc1':>7} {'Desc2':>7} {'Matches':>9} {'Inliers':>9} {'Rate':>7}"
    print(header)
    print("─" * len(header))
    for m in all_metrics:
        inl = str(m["inliers"]) if m["inliers"] is not None else "–"
        rate = f"{100*m['inlier_rate']:.1f}%" if m["inlier_rate"] is not None else "–"
        print(f"{m['scene']:<10} {m['harris_1']:>9} {m['harris_2']:>9} "
              f"{m['nms_1']:>7} {m['nms_2']:>7} "
              f"{m['descriptors_1']:>7} {m['descriptors_2']:>7} "
              f"{m['nndr_matches']:>9} {inl:>9} {rate:>7}")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s")
    print(f"Results saved to: {os.path.abspath(results_dir)}/")


if __name__ == "__main__":
    main()
