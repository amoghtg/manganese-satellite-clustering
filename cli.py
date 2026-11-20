#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for SIH26009 Manganese Spectral Clustering & Validation.

Usage Examples:
    # Run full demo on Balaghat Manganese Belt
    python cli.py --demo

    # Run on custom Sentinel-2 GeoTIFF
    python cli.py --geotiff path/to/scene.tif --clusters 5

    # Run with automated optimal K search (Elbow & Silhouette)
    python cli.py --demo --optimize-k
"""

import os
import sys
import argparse

# Ensure current project directory is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import ManganeseProspectingPipeline


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(
        description="SIH26009: K-Means Clustering of Satellite Mineral-Signature Data to Identify Manganese-Rich Zones",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--demo", action="store_true",
        help="Run end-to-end demonstration using physically calibrated Sausar Group (Balaghat) scene"
    )
    parser.add_argument(
        "--region", type=str, default="balaghat", choices=["balaghat", "sandur"],
        help="Candidate mineralized region to process (default: balaghat)"
    )
    parser.add_argument(
        "--geotiff", type=str, default=None,
        help="Path to input multispectral GeoTIFF (6-band Sentinel-2 or Landsat-8/9)"
    )
    parser.add_argument(
        "--clusters", "-k", type=int, default=5,
        help="Number of K-Means clusters (default: 5)"
    )
    parser.add_argument(
        "--optimize-k", action="store_true",
        help="Automatically evaluate optimal K using Silhouette Score & Elbow inertia analysis"
    )
    parser.add_argument(
        "--ground-truth", type=str, default=os.path.join(BASE_DIR, "data", "ground_truth", "gsi_manganese_deposits.csv"),
        help="Path to Geological Survey confirmed deposit CSV"
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join(BASE_DIR, "data", "outputs"),
        help="Directory to save generated GeoTIFFs, plots, and CSV reports"
    )
    parser.add_argument(
        "--ndvi-cutoff", type=float, default=0.40,
        help="NDVI threshold to mask dense vegetation (default: 0.40)"
    )

    args = parser.parse_args()

    pipeline = ManganeseProspectingPipeline(
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir
    )

    results = pipeline.run(
        input_geotiff=args.geotiff,
        region=args.region,
        n_clusters=args.clusters,
        optimize_k=args.optimize_k,
        ndvi_thresh=args.ndvi_cutoff
    )

    print("\n Pipeline Execution Finished Successfully!")
    print(" Check the outputs in:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
