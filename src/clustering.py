"""
K-Means Clustering engine for satellite mineral-signature segmentation
and automated manganese-likelihood scoring.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


class SpectralKMeansClassifier:
    """Performs K-Means segmentation on multispectral feature spaces and scores manganese prospectivity."""

    DEFAULT_FEATURE_KEYS = [
        "iron_oxide_index",
        "manganese_composite_ratio",
        "ferrous_iron_index",
        "clay_alteration_index",
        "normalized_manganese_index",
        "crosta_oxide_pc",
        "red",
        "blue",
        "nir",
        "swir2"
    ]

    def __init__(
        self,
        n_clusters: int = 5,
        feature_keys: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.n_clusters = n_clusters
        self.feature_keys = feature_keys or self.DEFAULT_FEATURE_KEYS
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Optional[KMeans] = None
        self.cluster_stats_: Optional[pd.DataFrame] = None
        self.ranked_clusters_: Optional[List[int]] = None
        self.cluster_labels_map_: Optional[Dict[int, Dict[str, Any]]] = None

    def _prepare_feature_matrix(
        self, features_dict: Dict[str, np.ndarray], mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Extracts and flattens requested features for masked pixels."""
        first_key = self.feature_keys[0]
        H, W = features_dict[first_key].shape

        if mask is None:
            mask = features_dict.get("valid_mask", np.ones((H, W), dtype=bool))

        valid_indices = np.where(mask.flatten())[0]

        feature_cols = []
        for key in self.feature_keys:
            arr = features_dict[key].flatten()
            feature_cols.append(arr[valid_indices])

        X = np.stack(feature_cols, axis=1)
        # Handle any residual non-finite values
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        return X, valid_indices, (H, W)

    def evaluate_optimal_k(
        self,
        features_dict: Dict[str, np.ndarray],
        k_range: range = range(2, 9),
        sample_size: int = 10000,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates clustering performance across a range of K using Inertia,
        Silhouette Score, and Davies-Bouldin Index.
        """
        X, _, _ = self._prepare_feature_matrix(features_dict, mask=mask)
        
        # Subsample for fast evaluation
        if len(X) > sample_size:
            idx = np.random.RandomState(self.random_state).choice(len(X), sample_size, replace=False)
            X_eval = X[idx]
        else:
            X_eval = X

        X_scaled = self.scaler.fit_transform(X_eval)

        inertias = []
        silhouettes = []
        davies_bouldin = []
        k_values = list(k_range)

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            labels = km.fit_predict(X_scaled)
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X_scaled, labels))
            db = float(davies_bouldin_score(X_scaled, labels))
            silhouettes.append(sil)
            davies_bouldin.append(db)

        # Select best K based on peak silhouette score
        best_k_idx = int(np.argmax(silhouettes))
        optimal_k = k_values[best_k_idx]

        return {
            "k_values": k_values,
            "inertias": inertias,
            "silhouettes": silhouettes,
            "davies_bouldin": davies_bouldin,
            "optimal_k": optimal_k
        }

    def fit_predict(
        self,
        features_dict: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        pixel_size_km: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Fits KMeans on feature set, generates full 2D cluster map,
        and computes manganese prospectivity scores.
        
        Returns:
            cluster_map: 2D array of cluster IDs (-1 for masked non-geological pixels)
            likelihood_map: 2D array of continuous Manganese Prospectivity (0.0 to 1.0)
            stats_df: Summary DataFrame with cluster profiles and likelihood designations
        """
        X, valid_indices, (H, W) = self._prepare_feature_matrix(features_dict, mask=mask)
        X_scaled = self.scaler.fit_transform(X)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = self.model.fit_predict(X_scaled)

        # Reconstruct full 2D cluster raster
        full_labels = np.full(H * W, fill_value=-1, dtype=np.int16)
        full_labels[valid_indices] = labels
        cluster_map = full_labels.reshape((H, W))

        # Compute cluster mean properties
        stats_list = []
        raw_feature_df = pd.DataFrame(X, columns=self.feature_keys)
        raw_feature_df["cluster"] = labels

        pixel_area_km2 = pixel_size_km ** 2

        for c_id in range(self.n_clusters):
            c_data = raw_feature_df[raw_feature_df["cluster"] == c_id]
            count = len(c_data)
            area_km2 = count * pixel_area_km2

            stat_entry = {
                "cluster_id": c_id,
                "pixel_count": count,
                "area_km2": round(area_km2, 2),
                "pct_of_valid_area": round(count / len(labels) * 100.0, 2)
            }
            for feat in self.feature_keys:
                stat_entry[feat] = float(c_data[feat].mean()) if count > 0 else 0.0

            stats_list.append(stat_entry)

        stats_df = pd.DataFrame(stats_list)

        # Calculate Manganese Prospectivity Score (0 - 100)
        # Manganese-bearing ore has:
        # High Iron Oxide Index, High Manganese Composite Ratio, High Ferrous Iron Index,
        # High Crosta Oxide PC, and Low NIR Reflectance.
        def min_max_norm(series: pd.Series) -> pd.Series:
            s_min, s_max = series.min(), series.max()
            if s_max - s_min == 0:
                return pd.Series(0.5, index=series.index)
            return (series - s_min) / (s_max - s_min)

        mocr_n = min_max_norm(stats_df["manganese_composite_ratio"])
        ioi_n = min_max_norm(stats_df["iron_oxide_index"])
        fii_n = min_max_norm(stats_df["ferrous_iron_index"])
        fpca_n = min_max_norm(stats_df["crosta_oxide_pc"])
        nir_inv = 1.0 - min_max_norm(stats_df["nir"])

        # Composite prospectivity weighting
        composite_score = (
            0.35 * mocr_n +
            0.25 * ioi_n +
            0.15 * fii_n +
            0.15 * fpca_n +
            0.10 * nir_inv
        ) * 100.0

        stats_df["manganese_score"] = composite_score.round(1)

        # Rank clusters: Highest score = rank 1
        stats_df = stats_df.sort_values(by="manganese_score", ascending=False).reset_index(drop=True)
        stats_df["rank"] = stats_df.index + 1

        # Assign geological likelihood tier based on rank and relative prospectivity
        def assign_tier(score: float, rank: int, total_k: int) -> str:
            if rank == 1:
                return "Tier 1: High Potential (Manganese Ore / Gossan Zone)"
            elif rank == 2:
                return "Tier 2: Moderate Potential (Ferromanganese / Gondite Horizon)"
            elif rank == 3:
                return "Tier 3: Host Lithology (Quartz-Mica Schist / Carbonate)"
            elif rank < total_k:
                return "Tier 4: Country Rock (Granite-Gneiss / Basement)"
            else:
                return "Tier 5: Unmineralized Regolith / Barren Alluvium"

        stats_df["likelihood_tier"] = [
            assign_tier(row["manganese_score"], row["rank"], len(stats_df)) for _, row in stats_df.iterrows()
        ]

        self.cluster_stats_ = stats_df
        self.ranked_clusters_ = stats_df["cluster_id"].tolist()

        # Build 2D likelihood probability raster
        score_lookup = dict(zip(stats_df["cluster_id"], stats_df["manganese_score"] / 100.0))
        full_likelihood = np.zeros(H * W, dtype=np.float32)
        
        for c_id, scr in score_lookup.items():
            c_indices = valid_indices[labels == c_id]
            full_likelihood[c_indices] = scr

        likelihood_map = full_likelihood.reshape((H, W))

        # Build metadata map
        self.cluster_labels_map_ = {}
        for _, row in stats_df.iterrows():
            cid = int(row["cluster_id"])
            self.cluster_labels_map_[cid] = {
                "rank": int(row["rank"]),
                "score": float(row["manganese_score"]),
                "tier": str(row["likelihood_tier"]),
                "area_km2": float(row["area_km2"]),
                "is_target": bool(row["rank"] <= 2)
            }

        return cluster_map, likelihood_map, stats_df
