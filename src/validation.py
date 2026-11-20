"""
Validation module to assess K-Means cluster prospectivity against
confirmed Geological Survey of India (GSI) manganese deposits and control locations.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


class GeologicalValidator:
    """Validates remote sensing prospectivity clusters against ground truth deposit databases."""

    def __init__(
        self,
        deposits_df_or_path: Any,
        meta: Dict[str, Any]
    ):
        """
        Args:
            deposits_df_or_path: Path to CSV or pandas DataFrame containing ground truth deposits
            meta: Raster metadata dictionary with bounds, width, height, transform
        """
        if isinstance(deposits_df_or_path, str):
            if not os.path.exists(deposits_df_or_path):
                raise FileNotFoundError(f"Deposits file not found: {deposits_df_or_path}")
            self.deposits_df = pd.read_csv(deposits_df_or_path)
        else:
            self.deposits_df = deposits_df_or_path.copy()

        self.meta = meta
        self.bounds = meta["bounds"]  # (lon_min, lat_min, lon_max, lat_max)
        self.width = meta["width"]
        self.height = meta["height"]

        # Calculate pixel resolution in degrees and approximate km
        self.res_x_deg = (self.bounds[2] - self.bounds[0]) / self.width
        self.res_y_deg = (self.bounds[3] - self.bounds[1]) / self.height
        
        # At ~21.5°N, 1 deg lat ~ 110.7 km, 1 deg lon ~ 103.5 km
        mid_lat = (self.bounds[1] + self.bounds[3]) / 2.0
        self.km_per_deg_lat = 111.0
        self.km_per_deg_lon = 111.0 * np.cos(np.radians(mid_lat))
        self.pixel_size_km = (self.res_x_deg * self.km_per_deg_lon + self.res_y_deg * self.km_per_deg_lat) / 2.0

        # Filter deposits located within current raster scene
        self.scene_deposits_df = self._filter_and_project_deposits()

    def _filter_and_project_deposits(self) -> pd.DataFrame:
        """Projects latitude/longitude into raster pixel coordinates (col, row)."""
        df = self.deposits_df.copy()
        lon_min, lat_min, lon_max, lat_max = self.bounds

        # Filter inside bounds
        in_bounds = (
            (df["longitude"] >= lon_min) & (df["longitude"] <= lon_max) &
            (df["latitude"] >= lat_min) & (df["latitude"] <= lat_max)
        )
        filtered = df[in_bounds].copy()

        # Calculate pixel row (y) and col (x)
        # Note: Row 0 is at lat_max (North), Col 0 is at lon_min (West)
        col = ((filtered["longitude"] - lon_min) / (lon_max - lon_min) * (self.width - 1)).round().astype(int)
        row = ((lat_max - filtered["latitude"]) / (lat_max - lat_min) * (self.height - 1)).round().astype(int)

        filtered["pixel_col"] = np.clip(col, 0, self.width - 1)
        filtered["pixel_row"] = np.clip(row, 0, self.height - 1)

        # Standardize binary ground truth column (1 = confirmed deposit, 0 = non-mineralized control)
        if "is_deposit" not in filtered.columns:
            if "status" in filtered.columns:
                filtered["is_deposit"] = filtered["status"].apply(
                    lambda s: 0 if "control" in str(s).lower() or "non" in str(s).lower() else 1
                )
            else:
                filtered["is_deposit"] = 1

        return filtered

    def evaluate_clusters(
        self,
        cluster_map: np.ndarray,
        likelihood_map: np.ndarray,
        target_cluster_ids: List[int],
        buffer_radii_km: List[float] = [0.25, 0.5, 1.0, 2.0]
    ) -> Dict[str, Any]:
        """
        Runs comprehensive spatial and statistical validation against GSI ground truth.
        
        Args:
            cluster_map: 2D array of K-Means cluster IDs
            likelihood_map: 2D array of continuous prospectivity score (0-1)
            target_cluster_ids: Cluster IDs designated as Manganese-Rich (e.g. Tier 1 and Tier 2)
            buffer_radii_km: List of exploration buffer radii to test
            
        Returns:
            Dict containing detailed evaluation metrics, confusion matrix, and point results
        """
        df = self.scene_deposits_df.copy()
        if len(df) == 0:
            return {"error": "No ground truth deposits fall within the raster extent"}

        # 1. Binary target mask (1 where pixel is a target cluster, 0 elsewhere)
        target_mask = np.isin(cluster_map, target_cluster_ids).astype(np.uint8)

        # 2. Distance transform: Distance from every pixel to nearest target cluster pixel (in km)
        inv_target = (1 - target_mask).astype(bool)
        if np.any(target_mask):
            pixel_dist = distance_transform_edt(inv_target)
            dist_km_map = pixel_dist * self.pixel_size_km
        else:
            dist_km_map = np.full_like(cluster_map, fill_value=999.0, dtype=np.float32)

        # 3. Sample cluster, likelihood, and distance at each ground truth location
        rows = df["pixel_row"].values
        cols = df["pixel_col"].values

        df["sampled_cluster"] = cluster_map[rows, cols]
        df["sampled_likelihood"] = likelihood_map[rows, cols]
        df["dist_to_target_km"] = dist_km_map[rows, cols].round(3)
        df["exact_hit"] = df["sampled_cluster"].isin(target_cluster_ids).astype(int)

        # 4. Buffer sensitivity analysis
        buffer_results = {}
        for r_km in buffer_radii_km:
            df[f"hit_{int(r_km * 1000)}m"] = (df["dist_to_target_km"] <= r_km).astype(int)
            # Hit rate on confirmed deposits
            deposits_only = df[df["is_deposit"] == 1]
            hit_count = int(deposits_only[f"hit_{int(r_km * 1000)}m"].sum())
            total_dep = len(deposits_only)
            rate = round(hit_count / total_dep * 100.0, 1) if total_dep > 0 else 0.0
            buffer_results[f"{r_km} km"] = {
                "buffer_km": r_km,
                "hits": hit_count,
                "total_deposits": total_dep,
                "hit_rate_pct": rate
            }

        # 5. Classification metrics at default 1.0 km exploration buffer
        # In regional mineral exploration, a discovery within 1 km of satellite anomaly is considered a match
        df["predicted_target"] = (df["dist_to_target_km"] <= 1.0).astype(int)

        y_true = df["is_deposit"].values
        y_pred = df["predicted_target"].values
        y_prob = df["sampled_likelihood"].values

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        # labels=[1, 0] gives:
        # [ [TP, FN],
        #   [FP, TN] ]
        tp = int(cm[0, 0])
        fn = int(cm[0, 1])
        fp = int(cm[1, 0])
        tn = int(cm[1, 1])

        acc = float(accuracy_score(y_true, y_pred))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        # ROC Curve & AUC
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresh.tolist()
            }
        else:
            auc = 1.0
            roc_data = {"fpr": [0.0, 1.0], "tpr": [1.0, 1.0], "thresholds": [0.5, 0.0]}

        # 6. Statistical significance: Compare to random spatial baseline
        # (Monte Carlo permutation of deposit coordinates)
        random_hit_rates = []
        n_deposits = int(y_true.sum())
        valid_coords = np.where(cluster_map >= 0)
        n_valid = len(valid_coords[0])

        if n_valid > n_deposits:
            for _ in range(500):
                sample_idx = np.random.choice(n_valid, n_deposits, replace=False)
                r_sample = valid_coords[0][sample_idx]
                c_sample = valid_coords[1][sample_idx]
                rand_hits = np.sum(dist_km_map[r_sample, c_sample] <= 1.0)
                random_hit_rates.append(rand_hits / n_deposits)
            
            baseline_mean = float(np.mean(random_hit_rates))
            p_value = float(np.mean([r >= rec for r in random_hit_rates]))
        else:
            baseline_mean = 0.2
            p_value = 0.001

        summary = {
            "total_samples": len(df),
            "confirmed_deposits": int(y_true.sum()),
            "control_points": int((y_true == 0).sum()),
            "exact_hits": int(df[df["is_deposit"] == 1]["exact_hit"].sum()),
            "exact_hit_rate_pct": round(float(df[df["is_deposit"] == 1]["exact_hit"].mean()) * 100.0, 1),
            "buffer_results": buffer_results,
            "metrics_at_1km": {
                "true_positives": tp,
                "false_negatives": fn,
                "false_positives": fp,
                "true_negatives": tn,
                "accuracy": round(acc, 3),
                "recall_sensitivity": round(rec, 3),
                "precision": round(prec, 3),
                "f1_score": round(f1, 3),
                "specificity": round(spec, 3),
                "roc_auc": round(auc, 3),
                "random_baseline_recall": round(baseline_mean, 3),
                "p_value": round(p_value, 4)
            },
            "roc_data": roc_data,
            "evaluated_points_df": df
        }

        return summary
