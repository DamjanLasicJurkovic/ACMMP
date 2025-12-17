#!/usr/bin/env python3
"""
Visualize depth maps from ACMMP results.

Usage: python visualize_depths.py <results_directory> [--percentile PERCENTILE]

This script iterates through subdirectories in the results directory,
reads depths.dmb and depths_geom.dmb files, and generates grayscale PNG visualizations
in a 'visualized' subdirectory. Output files are named <image_id>_depth.png and
<image_id>_depth_geom.png with 1-1 pixel mapping to the original depth maps.

Depth normalization uses robust statistics: ignores PERCENTILE% closest and PERCENTILE%
farthest depth values (outliers) to improve contrast, with outliers clamped to black/white.
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import struct


def read_depth_dmb(file_path):
    """
    Read a depth DMB file based on the readDepthDmb function from ACMMP.cpp

    Returns:
        numpy array of shape (h, w) containing depth values as floats
    """
    with open(file_path, 'rb') as f:
        # Read header: type, h, w, nb (all int32_t)
        type_val = struct.unpack('<i', f.read(4))[0]
        h = struct.unpack('<i', f.read(4))[0]
        w = struct.unpack('<i', f.read(4))[0]
        nb = struct.unpack('<i', f.read(4))[0]

        if type_val != 1:
            raise ValueError(f"Invalid DMB file type: {type_val}")

        # Read depth data
        data_size = h * w * nb
        depth_data = struct.unpack('<' + 'f' * data_size, f.read(4 * data_size))

        # Reshape to (h, w) - assuming nb=1 for depth maps
        if nb == 1:
            depth_map = np.array(depth_data).reshape(h, w)
        else:
            depth_map = np.array(depth_data).reshape(h, w, nb)
            # For depth maps, we probably want the first channel
            depth_map = depth_map[:, :, 0]

        return depth_map


def visualize_depth_map(depth_map, output_path, outlier_percentile=1.0):
    """
    Create a grayscale visualization of the depth map and save as PNG.
    Maintains 1-1 pixel mapping with the original depth map.
    Uses robust normalization by ignoring outliers on each end.

    Args:
        depth_map: numpy array of depth values
        output_path: path to save the PNG file
        outlier_percentile: percentage of outliers to ignore on each end (default: 1.0)
    """
    # Handle invalid depth values (typically 0 or negative)
    valid_mask = depth_map > 0

    if not np.any(valid_mask):
        # If no valid depths, create a black image
        grayscale_depth = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        # Get valid depths for percentile calculation
        valid_depths = depth_map[valid_mask]

        # Calculate percentiles to ignore outliers
        depth_low = np.percentile(valid_depths, outlier_percentile)
        depth_high = np.percentile(valid_depths, 100 - outlier_percentile)

        # Use robust range for normalization
        depth_min = depth_low
        depth_max = depth_high

        # Create normalized depth map
        normalized_depth = np.full_like(depth_map, 0, dtype=np.float32)

        # Normalize valid depths, clamping outliers
        valid_values = depth_map[valid_mask]
        normalized_values = (valid_values - depth_min) / (depth_max - depth_min)

        # Clamp to [0, 1] range
        normalized_values = np.clip(normalized_values, 0.0, 1.0)

        normalized_depth[valid_mask] = normalized_values

        # Convert to 0-255 grayscale
        grayscale_depth = (normalized_depth * 255).astype(np.uint8)

    # Create PIL image and save directly (maintains exact pixel dimensions)
    img = Image.fromarray(grayscale_depth, mode='L')  # 'L' for grayscale
    img.save(output_path)

    print(f"Saved visualization: {output_path}")


def process_directory(results_dir, outlier_percentile=1.0):
    """
    Process all subdirectories in the results directory.
    Creates a 'visualized' subdirectory and saves all PNGs there.

    Args:
        results_dir: path to the results directory
        outlier_percentile: percentage of outliers to ignore on each end
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return

    if not results_path.is_dir():
        print(f"Error: '{results_dir}' is not a directory.")
        return

    # Create visualized directory
    visualized_dir = results_path / "visualized"
    visualized_dir.mkdir(exist_ok=True)
    print(f"Created/using output directory: {visualized_dir}")

    # Find all subdirectories
    subdirs = [d for d in results_path.iterdir() if d.is_dir() and d.name != "visualized"]

    if not subdirs:
        print(f"No subdirectories found in '{results_dir}'.")
        return

    print(f"Found {len(subdirs)} subdirectories to process.")

    processed_count = 0

    for subdir in sorted(subdirs):
        print(f"Processing {subdir.name}...")

        # Look for depth files
        depths_file = subdir / "depths.dmb"
        depths_geom_file = subdir / "depths_geom.dmb"

        # Process depths.dmb
        if depths_file.exists():
            try:
                depth_map = read_depth_dmb(str(depths_file))
                output_file = visualized_dir / f"{subdir.name}_depth.png"
                visualize_depth_map(depth_map, str(output_file), outlier_percentile)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {depths_file}: {e}")
        else:
            print(f"Warning: {depths_file} not found in {subdir.name}")

        # Process depths_geom.dmb
        if depths_geom_file.exists():
            try:
                depth_map = read_depth_dmb(str(depths_geom_file))
                output_file = visualized_dir / f"{subdir.name}_depth_geom.png"
                visualize_depth_map(depth_map, str(output_file), outlier_percentile)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {depths_geom_file}: {e}")
        else:
            print(f"Warning: {depths_geom_file} not found in {subdir.name}")

    print(f"\nProcessing complete. Generated {processed_count} visualizations in {visualized_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize depth maps from ACMMP results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_depths.py <dir>/ACMMP
  python visualize_depths.py <dir>/ACMMP -p 0.5
  python visualize_depths.py <dir>/ACMMP --percentile 2.0
        """
    )

    parser.add_argument(
        "results_directory",
        help="Path to the results directory containing image subdirectories"
    )

    parser.add_argument(
        "-p", "--percentile",
        type=float,
        default=1.0,
        help="Percentage of outliers to ignore on each end (default: 1.0)"
    )

    args = parser.parse_args()

    # Validate percentile range
    if args.percentile <= 0 or args.percentile >= 50:
        print(f"Error: Percentile must be between 0 and 50, got {args.percentile}")
        sys.exit(1)

    print(f"Using outlier percentile: {args.percentile}%")
    process_directory(args.results_directory, args.percentile)


if __name__ == "__main__":
    main()
