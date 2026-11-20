"""
Automated Unit and Integration Tests for SIH26009 Manganese Spectral Prospecting Pipeline.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.spectral_indices import (
    safe_divide,
    compute_ndvi,
    compute_ndwi,
    compute_iron_oxide_ratio,
    compute_ferrous_iron_ratio,
    compute_clay_alteration_ratio,
    compute_manganese_composite_ratio,
    compute_crosta_fpca,
    extract_spectral_features
)
from src.data_loader import SatelliteDataLoader
from src.clustering import SpectralKMeansClassifier
from src.validation import GeologicalValidator
from src.pipeline import ManganeseProspectingPipeline


class TestSpectralIndices(unittest.TestCase):
    """Tests for spectral feature extraction and mineral ratios."""

    def setUp(self):
        self.H, self.W = 50, 50
        # Mock bands
        self.blue = np.full((self.H, self.W), 0.10, dtype=np.float32)
        self.green = np.full((self.H, self.W), 0.12, dtype=np.float32)
        self.red = np.full((self.H, self.W), 0.20, dtype=np.float32)
        self.nir = np.full((self.H, self.W), 0.25, dtype=np.float32)
        self.swir1 = np.full((self.H, self.W), 0.30, dtype=np.float32)
        self.swir2 = np.full((self.H, self.W), 0.22, dtype=np.float32)

    def test_safe_divide(self):
        num = np.array([1.0, 2.0, 0.0])
        den = np.array([2.0, 0.0, 5.0])
        res = safe_divide(num, den)
        self.assertAlmostEqual(res[0], 0.5)
        self.assertFalse(np.isnan(res[1]))
        self.assertAlmostEqual(res[2], 0.0)

    def test_ndvi_and_ndwi(self):
        ndvi = compute_ndvi(self.nir, self.red)
        # (0.25 - 0.20) / (0.25 + 0.20) = 0.05 / 0.45 ~ 0.111
        self.assertAlmostEqual(ndvi[0, 0], 0.111111, places=4)
        
        ndwi = compute_ndwi(self.green, self.nir)
        # (0.12 - 0.25) / (0.12 + 0.25) = -0.13 / 0.37 ~ -0.351
        self.assertAlmostEqual(ndwi[0, 0], -0.351351, places=4)

    def test_mineral_indices(self):
        ioi = compute_iron_oxide_ratio(self.red, self.blue)
        self.assertAlmostEqual(ioi[0, 0], 2.0, places=4)

        fii = compute_ferrous_iron_ratio(self.swir2, self.nir)
        self.assertAlmostEqual(fii[0, 0], 0.22 / 0.25, places=4)

        cai = compute_clay_alteration_ratio(self.swir1, self.swir2)
        self.assertAlmostEqual(cai[0, 0], 0.30 / 0.22, places=4)

        mocr = compute_manganese_composite_ratio(self.red, self.blue, self.nir, self.swir2)
        expected_mocr = (0.20 * 0.22) / (0.10 * 0.25)  # 0.044 / 0.025 = 1.76
        self.assertAlmostEqual(mocr[0, 0], expected_mocr, places=4)

    def test_crosta_fpca(self):
        # Add slight variation to prevent singular covariance
        b = self.blue + np.random.normal(0, 0.01, (self.H, self.W))
        r = self.red + np.random.normal(0, 0.01, (self.H, self.W))
        s1 = self.swir1 + np.random.normal(0, 0.01, (self.H, self.W))
        s2 = self.swir2 + np.random.normal(0, 0.01, (self.H, self.W))
        
        pc_img, loadings, pc_idx = compute_crosta_fpca(b, r, s1, s2)
        self.assertEqual(pc_img.shape, (self.H, self.W))
        self.assertEqual(loadings.shape, (4, 4))
        self.assertIn(pc_idx, [0, 1, 2, 3])


class TestDataLoader(unittest.TestCase):
    """Tests for synthetic data generation and GeoTIFF I/O."""

    def test_synthetic_scene_generation(self):
        test_tif = os.path.join(PROJECT_ROOT, "data", "sample_satellite", "test_mock.tif")
        bands, meta = SatelliteDataLoader.generate_synthetic_scene(
            test_tif, region_name="balaghat", width=100, height=100, seed=123
        )
        self.assertTrue(os.path.exists(test_tif))
        self.assertEqual(meta["width"], 100)
        self.assertEqual(meta["height"], 100)
        self.assertEqual(meta["count"], 6)

        # Check all 6 bands present and valid
        for name in ["blue", "green", "red", "nir", "swir1", "swir2"]:
            self.assertIn(name, bands)
            self.assertEqual(bands[name].shape, (100, 100))
            self.assertTrue(np.all(bands[name] >= 0.0))
            self.assertTrue(np.all(bands[name] <= 1.0))

        # Test reload
        loaded_bands, loaded_meta = SatelliteDataLoader.load_geotiff(test_tif)
        self.assertEqual(loaded_meta["width"], 100)
        np.testing.assert_allclose(loaded_bands["red"], bands["red"], atol=1e-5)

        # Cleanup
        if os.path.exists(test_tif):
            os.remove(test_tif)
        meta_json = test_tif.replace(".tif", "_meta.json")
        if os.path.exists(meta_json):
            os.remove(meta_json)


class TestClustering(unittest.TestCase):
    """Tests for KMeans clustering and prospectivity scoring."""

    def test_kmeans_and_scoring(self):
        # Create synthetic scene
        test_tif = os.path.join(PROJECT_ROOT, "data", "sample_satellite", "test_cluster.tif")
        bands, meta = SatelliteDataLoader.generate_synthetic_scene(
            test_tif, width=80, height=80, seed=42
        )
        features = extract_spectral_features(bands, apply_masking=True)

        clf = SpectralKMeansClassifier(n_clusters=4, random_state=42)
        cluster_map, likelihood_map, stats_df = clf.fit_predict(features)

        self.assertEqual(cluster_map.shape, (80, 80))
        self.assertEqual(likelihood_map.shape, (80, 80))
        self.assertEqual(len(stats_df), 4)

        # Check ranking
        self.assertEqual(list(stats_df["rank"]), [1, 2, 3, 4])
        # Score must be non-negative and <= 100
        self.assertTrue(np.all(stats_df["manganese_score"] >= 0))
        self.assertTrue(np.all(stats_df["manganese_score"] <= 100))

        # Cleanup
        if os.path.exists(test_tif):
            os.remove(test_tif)
        meta_json = test_tif.replace(".tif", "_meta.json")
        if os.path.exists(meta_json):
            os.remove(meta_json)


class TestValidation(unittest.TestCase):
    """Tests for ground truth validation metrics and buffer analysis."""

    def test_validation_logic(self):
        # Synthetic mock ground truth
        mock_deposits = pd.DataFrame({
            "deposit_id": ["MN_1", "MN_2", "CTRL_1"],
            "name": ["Test Mine 1", "Test Mine 2", "Barren Control"],
            "latitude": [21.80, 21.90, 21.60],
            "longitude": [80.00, 80.20, 79.80],
            "is_deposit": [1, 1, 0]
        })

        meta = {
            "bounds": (79.60, 21.50, 80.50, 22.05),
            "width": 100,
            "height": 100,
            "crs": "EPSG:4326"
        }

        # Cluster map where row 20-50, col 20-50 is target cluster 0
        cluster_map = np.ones((100, 100), dtype=np.int16)
        cluster_map[20:60, 20:60] = 0
        likelihood_map = np.full((100, 100), 0.2, dtype=np.float32)
        likelihood_map[20:60, 20:60] = 0.9

        validator = GeologicalValidator(mock_deposits, meta)
        results = validator.evaluate_clusters(
            cluster_map=cluster_map,
            likelihood_map=likelihood_map,
            target_cluster_ids=[0],
            buffer_radii_km=[0.5, 1.0, 2.0]
        )

        self.assertIn("metrics_at_1km", results)
        self.assertIn("buffer_results", results)
        self.assertEqual(results["confirmed_deposits"], 2)
        self.assertEqual(results["control_points"], 1)


class TestEndToEndPipeline(unittest.TestCase):
    """Full integration test of pipeline runner."""

    def test_full_pipeline_run(self):
        pipeline = ManganeseProspectingPipeline(
            output_dir=os.path.join(PROJECT_ROOT, "data", "outputs", "test_run")
        )
        res = pipeline.run(
            region="balaghat",
            n_clusters=4,
            optimize_k=False,
            random_state=42
        )

        self.assertIsNotNone(res["cluster_map"])
        self.assertIsNotNone(res["likelihood_map"])
        self.assertIn("1.0 km", res["validation_summary"]["buffer_results"])
        
        # Verify generated output files
        for fpath in res["output_files"].values():
            self.assertTrue(os.path.exists(fpath), f"Expected file missing: {fpath}")


if __name__ == "__main__":
    unittest.main()
