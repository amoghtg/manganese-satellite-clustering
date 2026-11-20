"""
Visualization utilities for mineral indices, K-Means cluster maps,
ground truth overlays, and evaluation metrics.
"""

import os
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless generation
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


class SpectralVisualizer:
    """Creates scientific, high-resolution figures for remote sensing mineral exploration."""

    @staticmethod
    def get_rgb_composite(
        bands: Dict[str, np.ndarray],
        rgb_bands: List[str] = ["red", "green", "blue"],
        percentile_stretch: Tuple[float, float] = (2, 98)
    ) -> np.ndarray:
        """Constructs a contrast-stretched RGB composite image."""
        r = bands[rgb_bands[0]]
        g = bands[rgb_bands[1]]
        b = bands[rgb_bands[2]]
        
        rgb = np.stack([r, g, b], axis=-1)
        
        # Percentile contrast stretch
        p_low, p_high = np.percentile(rgb, percentile_stretch)
        rgb_stretched = np.clip((rgb - p_low) / (p_high - p_low + 1e-6), 0.0, 1.0)
        return rgb_stretched

    @classmethod
    def plot_spectral_indices(
        cls,
        features: Dict[str, np.ndarray],
        output_path: str,
        title: str = "Diagnostic Spectral Mineral Indices"
    ) -> str:
        """Plots a 4-panel grid of the diagnostic manganese exploration indices."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

        indices = [
            ("iron_oxide_index", "Iron Oxide / Gossan Index (Red / Blue)", "YlOrRd"),
            ("manganese_composite_ratio", "Manganese Composite Ratio (MOCR)", "inferno"),
            ("ferrous_iron_index", "Ferrous Iron Index (SWIR2 / NIR)", "magma"),
            ("crosta_oxide_pc", "Crosta FPCA Oxide Component", "viridis")
        ]

        for ax, (key, label, cmap) in zip(axes.flat, indices):
            data = features[key]
            valid_mask = features.get("valid_mask", np.ones_like(data, dtype=bool))
            display_data = np.where(valid_mask, data, np.nan)
            
            # Clip 2nd to 98th percentile for robust contrast
            valid_vals = display_data[~np.isnan(display_data)]
            vmin, vmax = np.percentile(valid_vals, [2, 98])
            
            im = ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(label, fontsize=13, fontweight="semibold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @classmethod
    def plot_cluster_results(
        cls,
        cluster_map: np.ndarray,
        likelihood_map: np.ndarray,
        stats_df: pd.DataFrame,
        deposits_df: pd.DataFrame,
        output_path: str,
        title: str = "K-Means Manganese Prospectivity & GSI Deposit Validation"
    ) -> str:
        """Plots cluster segmentation side-by-side with continuous likelihood and GSI deposit overlays."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

        # Discrete colormap for clusters
        n_clusters = len(stats_df)
        palette = ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"]
        if n_clusters > len(palette):
            palette = plt.cm.tab10.colors[:n_clusters]
        else:
            palette = palette[:n_clusters]

        # 1. Cluster Segmentation Map
        ax1 = axes[0]
        # Re-map cluster IDs so rank 1 is highest
        display_map = np.full_like(cluster_map, fill_value=np.nan, dtype=np.float32)
        rank_lookup = dict(zip(stats_df["cluster_id"], stats_df["rank"]))
        for cid, rank in rank_lookup.items():
            display_map[cluster_map == cid] = rank

        cmap_discrete = mcolors.ListedColormap(palette[:n_clusters])
        bounds_ticks = np.arange(0.5, n_clusters + 1.5, 1)
        norm = mcolors.BoundaryNorm(bounds_ticks, cmap_discrete.N)

        im1 = ax1.imshow(display_map, cmap=cmap_discrete, norm=norm)
        ax1.set_title("K-Means Lithological & Mineral Clusters", fontsize=14, fontweight="bold")
        ax1.axis("off")

        # Colorbar with cluster tier labels
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, ticks=range(1, n_clusters + 1))
        tick_labels = [f"Rank {r}: {stats_df.loc[stats_df['rank']==r, 'likelihood_tier'].values[0]}" for r in range(1, n_clusters + 1)]
        cbar1.ax.set_yticklabels(tick_labels, fontsize=9)

        # 2. Continuous Prospectivity Map + GSI Deposits
        ax2 = axes[1]
        im2 = ax2.imshow(likelihood_map * 100.0, cmap="hot_r", vmin=0, vmax=100)
        ax2.set_title("Manganese Likelihood Score (%) & GSI Ground Truth", fontsize=14, fontweight="bold")
        ax2.axis("off")

        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Manganese Prospectivity (%)", fontsize=11, fontweight="semibold")

        # Overlay Ground Truth Points
        # Confirmed deposits (stars)
        dep = deposits_df[deposits_df["is_deposit"] == 1]
        ax2.scatter(
            dep["pixel_col"], dep["pixel_row"],
            c="#00FFCC", marker="*", s=180, edgecolors="black", linewidths=1.5,
            label="GSI Confirmed Mn Deposit", zorder=5
        )
        for _, row in dep.iterrows():
            ax2.annotate(
                row["name"].split(" ")[0],
                (row["pixel_col"] + 4, row["pixel_row"] - 4),
                fontsize=8, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, ec="none")
            )

        # Control non-deposits (circles)
        ctrl = deposits_df[deposits_df["is_deposit"] == 0]
        if len(ctrl) > 0:
            ax2.scatter(
                ctrl["pixel_col"], ctrl["pixel_row"],
                c="#333333", marker="o", s=70, edgecolors="white", linewidths=1.2,
                label="Control Non-Mineralized Point", zorder=5
            )

        ax2.legend(loc="lower right", framealpha=0.9, fontsize=10)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @classmethod
    def plot_validation_report(
        cls,
        val_summary: Dict[str, Any],
        output_path: str
    ) -> str:
        """Plots confusion matrix, buffer sensitivity curve, and ROC curve."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        ax1 = axes[0]
        m = val_summary["metrics_at_1km"]
        cm_data = np.array([
            [m["true_positives"], m["false_negatives"]],
            [m["false_positives"], m["true_negatives"]]
        ])
        sns.heatmap(
            cm_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
            xticklabels=["Predicted Mn Zone", "Predicted Barren"],
            yticklabels=["GSI Confirmed Deposit", "Barren Control Point"],
            annot_kws={"fontsize": 14, "fontweight": "bold"}
        )
        ax1.set_title(f"Confusion Matrix (1km Buffer)\nAccuracy: {m['accuracy']*100:.1f}% | F1: {m['f1_score']:.2f}", fontsize=13, fontweight="bold")

        # 2. Buffer Sensitivity Curve
        ax2 = axes[1]
        b_res = val_summary["buffer_results"]
        radii = [v["buffer_km"] * 1000 for v in b_res.values()]
        rates = [v["hit_rate_pct"] for v in b_res.values()]
        
        ax2.plot(radii, rates, marker="o", linewidth=2.5, color="#d95f02", markersize=8)
        ax2.set_xlabel("Exploration Buffer Radius (Meters)", fontsize=11, fontweight="semibold")
        ax2.set_ylabel("Confirmed Deposit Hit Rate (%)", fontsize=11, fontweight="semibold")
        ax2.set_title("Spatial Detection Rate vs Buffer Radius", fontsize=13, fontweight="bold")
        ax2.set_ylim(-5, 105)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. ROC Curve
        ax3 = axes[2]
        roc = val_summary["roc_data"]
        ax3.plot(roc["fpr"], roc["tpr"], color="#7570b3", lw=2.5, label=f"ROC Curve (AUC = {m['roc_auc']:.2f})")
        ax3.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance (AUC = 0.50)")
        ax3.set_xlabel("False Positive Rate", fontsize=11, fontweight="semibold")
        ax3.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="semibold")
        ax3.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @classmethod
    def plot_spectral_curves(
        cls,
        stats_df: pd.DataFrame,
        output_path: str
    ) -> str:
        """Plots mean spectral response profiles across bands for each cluster."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
        wavelengths = [490, 560, 665, 842, 1610, 2190]  # nm for Sentinel-2
        
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(stats_df)))

        for i, (_, row) in enumerate(stats_df.iterrows()):
            vals = [row.get(b, 0.2) for b in band_names]
            label = f"Rank {row['rank']}: {row['likelihood_tier'][:25]} (Score: {row['manganese_score']}%)"
            ax.plot(wavelengths, vals, marker="s", lw=2.2, color=colors[i], label=label)

        ax.set_xlabel("Wavelength (nm)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Surface Reflectance", fontsize=12, fontweight="bold")
        ax.set_title("Mean Spectral Reflectance Curves by Mineral Cluster", fontsize=14, fontweight="bold")
        ax.set_xticks(wavelengths)
        ax.set_xticklabels(["B02\n490nm", "B03\n560nm", "B04\n665nm", "B08\n842nm", "B11\n1610nm", "B12\n2190nm"])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
