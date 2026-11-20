"""
End-to-end orchestration pipeline for Satellite Mineral-Signature Feature Extraction,
K-Means Clustering, and Geological Survey Deposit Validation.
"""

import os
import json
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from src.data_loader import SatelliteDataLoader
from src.spectral_indices import extract_spectral_features
from src.clustering import SpectralKMeansClassifier
from src.validation import GeologicalValidator
from src.visualizer import SpectralVisualizer


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class ManganeseProspectingPipeline:
    """Master orchestrator for the SIH26009 Manganese Spectral Prospectivity System."""

    def __init__(
        self,
        ground_truth_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        if ground_truth_path is None:
            self.ground_truth_path = os.path.join(PROJECT_ROOT, "data", "ground_truth", "gsi_manganese_deposits.csv")
        elif not os.path.isabs(ground_truth_path):
            self.ground_truth_path = os.path.join(PROJECT_ROOT, ground_truth_path)
        else:
            self.ground_truth_path = ground_truth_path

        if output_dir is None:
            self.output_dir = os.path.join(PROJECT_ROOT, "data", "outputs")
        elif not os.path.isabs(output_dir):
            self.output_dir = os.path.join(PROJECT_ROOT, output_dir)
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        input_geotiff: Optional[str] = None,
        region: str = "balaghat",
        n_clusters: int = 5,
        optimize_k: bool = False,
        ndvi_thresh: float = 0.40,
        ndwi_thresh: float = 0.10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Executes the full 3-step SIH workflow.
        
        Args:
            input_geotiff: Path to custom multi-band GeoTIFF, or None to generate realistic synthetic scene
            region: Candidate region name ('balaghat' or 'sandur')
            n_clusters: Number of K-Means clusters (ignored if optimize_k is True)
            optimize_k: If True, evaluates Silhouette/Inertia to automatically pick optimal K
            ndvi_thresh: Cutoff to mask dense vegetation
            ndwi_thresh: Cutoff to mask water bodies
            random_state: Reproducibility seed
            
        Returns:
            Dictionary containing processed rasters, stats, validation summary, and output file paths
        """
        print("\n" + "=" * 70)
        print("  SIH26009: K-MEANS CLUSTERING OF SATELLITE MINERAL SIGNATURES")
        print("            FOR MANGANESE DEPOSIT IDENTIFICATION")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # STEP 1: Ingest or Synthesize Satellite Data
        # ---------------------------------------------------------------------
        print(f"\n[Step 1/3] Loading Satellite Imagery (Region: {region})...")
        if input_geotiff and os.path.exists(input_geotiff):
            print(f" -> Loading custom GeoTIFF: {input_geotiff}")
            bands, meta = SatelliteDataLoader.load_geotiff(input_geotiff)
        else:
            synth_path = os.path.join(PROJECT_ROOT, "data", "sample_satellite", f"{region}_sentinel2_synthetic.tif")
            print(f" -> Synthesizing physically realistic Sentinel-2 tile: {synth_path}")
            bands, meta = SatelliteDataLoader.generate_synthetic_scene(
                synth_path, region_name=region, width=500, height=500, seed=random_state
            )
            input_geotiff = synth_path

        print(f" -> Dimensions: {meta['width']}x{meta['height']} | Bands: {meta['count']}")
        print(f" -> Geographic Extent (WGS84): {meta['bounds']}")

        # Feature Extraction
        print(" -> Extracting Diagnostic Mineral Signatures & Crosta FPCA...")
        features = extract_spectral_features(
            bands=bands,
            ndvi_threshold=ndvi_thresh,
            ndwi_threshold=ndwi_thresh,
            apply_masking=True
        )
        print("    * Iron Oxide / Gossan Index (Red / Blue)")
        print("    * Ferrous Iron Index (SWIR2 / NIR)")
        print("    * Clay / Hydrothermal Alteration Index (SWIR1 / SWIR2)")
        print("    * Manganese Oxide Composite Ratio (MOCR)")
        print("    * Normalized Manganese Index (NMI)")
        print(f"    * Crosta FPCA Component (PC {features['fpca_pc_idx'] + 1})")

        # ---------------------------------------------------------------------
        # STEP 2: K-Means Clustering & Manganese Likelihood Scoring
        # ---------------------------------------------------------------------
        print(f"\n[Step 2/3] Performing K-Means Clustering...")
        classifier = SpectralKMeansClassifier(
            n_clusters=n_clusters,
            random_state=random_state
        )

        k_opt_results = None
        if optimize_k:
            print(" -> Running automated optimal K search across K=2..8 (Elbow & Silhouette)...")
            k_opt_results = classifier.evaluate_optimal_k(features, k_range=range(2, 8))
            chosen_k = k_opt_results["optimal_k"]
            print(f" -> Optimal K selected by Silhouette Score: K = {chosen_k}")
            classifier.n_clusters = chosen_k
        else:
            print(f" -> Running with user-specified K = {n_clusters}")

        cluster_map, likelihood_map, stats_df = classifier.fit_predict(features)

        print("\n -> Cluster Litho-Unit Profiles & Manganese Prospectivity:")
        display_cols = ["rank", "cluster_id", "area_km2", "manganese_score", "likelihood_tier"]
        print(stats_df[display_cols].to_string(index=False))

        # Target clusters (Rank 1: High Potential; Rank 2: Moderate Potential)
        target_clusters = stats_df.loc[stats_df["rank"] <= 2, "cluster_id"].tolist()
        tier1_only = stats_df.loc[stats_df["rank"] == 1, "cluster_id"].tolist()
        print(f"\n -> Target Manganese Prospectivity Clusters (Tier 1 & 2): {target_clusters}")
        print(f" -> Tier 1 High-Potential Ore Cluster: {tier1_only}")

        # ---------------------------------------------------------------------
        # STEP 3: Validate Against Confirmed GSI Deposit Data
        # ---------------------------------------------------------------------
        print(f"\n[Step 3/3] Validating Against Confirmed Geological Survey Ground Truth...")
        validator = GeologicalValidator(self.ground_truth_path, meta)
        val_summary = validator.evaluate_clusters(
            cluster_map=cluster_map,
            likelihood_map=likelihood_map,
            target_cluster_ids=target_clusters,
            buffer_radii_km=[0.25, 0.5, 1.0, 2.0]
        )

        metrics = val_summary["metrics_at_1km"]
        print(f" -> Deposits Analyzed in Scene: {val_summary['confirmed_deposits']} confirmed, {val_summary['control_points']} barren controls")
        print(f" -> Exact Pixel Hit Rate: {val_summary['exact_hit_rate_pct']}%")
        print(f" -> Hit Rate at 500m Buffer: {val_summary['buffer_results']['0.5 km']['hit_rate_pct']}%")
        print(f" -> Hit Rate at 1.0 km Buffer: {val_summary['buffer_results']['1.0 km']['hit_rate_pct']}%")
        print(f" -> Classification Metrics (1 km Tolerance):")
        print(f"    * Accuracy:    {metrics['accuracy'] * 100:.1f}%")
        print(f"    * Sensitivity: {metrics['recall_sensitivity'] * 100:.1f}%")
        print(f"    * Precision:   {metrics['precision'] * 100:.1f}%")
        print(f"    * F1-Score:    {metrics['f1_score']:.3f}")
        print(f"    * ROC-AUC:     {metrics['roc_auc']:.3f}")
        print(f"    * Baseline:    {metrics['random_baseline_recall'] * 100:.1f}% (p-value: {metrics['p_value']})")

        # ---------------------------------------------------------------------
        # Export Rasters, Plots, and GeoJSON
        # ---------------------------------------------------------------------
        print("\n[Exporting Deliverables & Scientific Figures]...")
        
        # 1. Classified GeoTIFF
        clust_tif_path = os.path.join(self.output_dir, f"{region}_kmeans_clusters.tif")
        SatelliteDataLoader.save_classified_geotiff(clust_tif_path, cluster_map, meta)
        
        # 2. Likelihood GeoTIFF
        likeli_tif_path = os.path.join(self.output_dir, f"{region}_manganese_likelihood.tif")
        SatelliteDataLoader.save_classified_geotiff(
            likeli_tif_path, (likelihood_map * 100).astype(np.int16), meta
        )

        # 3. CSV Reports
        stats_csv_path = os.path.join(self.output_dir, f"{region}_cluster_statistics.csv")
        stats_df.to_csv(stats_csv_path, index=False)

        pts_csv_path = os.path.join(self.output_dir, f"{region}_deposit_validation_results.csv")
        val_summary["evaluated_points_df"].to_csv(pts_csv_path, index=False)

        # 4. Plots
        idx_plot_path = os.path.join(self.output_dir, f"{region}_spectral_indices.png")
        SpectralVisualizer.plot_spectral_indices(features, idx_plot_path)

        res_plot_path = os.path.join(self.output_dir, f"{region}_cluster_prospectivity.png")
        SpectralVisualizer.plot_cluster_results(
            cluster_map, likelihood_map, stats_df, val_summary["evaluated_points_df"], res_plot_path
        )

        val_plot_path = os.path.join(self.output_dir, f"{region}_validation_metrics.png")
        SpectralVisualizer.plot_validation_report(val_summary, val_plot_path)

        spec_curve_path = os.path.join(self.output_dir, f"{region}_spectral_profiles.png")
        SpectralVisualizer.plot_spectral_curves(stats_df, spec_curve_path)

        print(f" -> Output Figures & Data saved to: {self.output_dir}/")
        print("=" * 70 + "\n")

        return {
            "meta": meta,
            "bands": bands,
            "features": features,
            "cluster_map": cluster_map,
            "likelihood_map": likelihood_map,
            "stats_df": stats_df,
            "target_clusters": target_clusters,
            "validation_summary": val_summary,
            "k_opt_results": k_opt_results,
            "output_files": {
                "clusters_geotiff": clust_tif_path,
                "likelihood_geotiff": likeli_tif_path,
                "cluster_stats_csv": stats_csv_path,
                "validation_points_csv": pts_csv_path,
                "indices_plot": idx_plot_path,
                "clusters_plot": res_plot_path,
                "validation_plot": val_plot_path,
                "spectral_curves_plot": spec_curve_path
            }
        }
