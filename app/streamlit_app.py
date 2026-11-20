"""
Interactive Web Application for SIH26009:
K-Means Clustering of Satellite Mineral-Signature Data to Identify Manganese-Rich Zones.
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src is discoverable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import SatelliteDataLoader
from src.spectral_indices import extract_spectral_features
from src.clustering import SpectralKMeansClassifier
from src.validation import GeologicalValidator
from src.visualizer import SpectralVisualizer


# Page configuration
st.set_page_config(
    page_title="Manganese Satellite Prospector (SIH26009)",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 14px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar controls
st.sidebar.image("https://img.icons8.com/color/96/satellite.png", width=64)
st.sidebar.title("Configuration")

region_option = st.sidebar.selectbox(
    "Candidate Mining Region",
    [
        "Balaghat-Bhandara Belt (Sausar Group, MP/MH)",
        "Sandur-Bellary Belt (Dharwar Craton, Karnataka)",
        "Custom Upload (GeoTIFF)"
    ]
)

custom_file = None
if region_option == "Custom Upload (GeoTIFF)":
    custom_file = st.sidebar.file_uploader("Upload 6-band GeoTIFF", type=["tif", "tiff"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Hyperparameters")

k_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=3, max_value=8, value=5, step=1)
auto_optimize_k = st.sidebar.checkbox("Auto-select optimal K (Silhouette/Elbow)", value=False)
ndvi_cutoff = st.sidebar.slider("NDVI Vegetation Cutoff", min_value=0.20, max_value=0.60, value=0.40, step=0.05)
ndwi_cutoff = st.sidebar.slider("NDWI Water Cutoff", min_value=-0.10, max_value=0.30, value=0.10, step=0.05)
tolerance_buffer = st.sidebar.select_slider(
    "Evaluation Buffer Radius (km)",
    options=[0.25, 0.5, 1.0, 2.0],
    value=1.0
)

run_button = st.sidebar.button("Run Prospecting Pipeline", type="primary", use_container_width=True)

# Main Title Header
st.markdown('<div class="main-title">K-Means Clustering of Satellite Mineral-Signatures</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Identification of Manganese-Rich Zones | Smart India Hackathon (SIH26009)</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_process(region_code: str, k: int, opt_k: bool, ndvi_cut: float, ndwi_cut: float, custom_path: str = None):
    """Cached loader and processor."""
    if custom_path and os.path.exists(custom_path):
        bands, meta = SatelliteDataLoader.load_geotiff(custom_path)
    else:
        synth_path = os.path.join(PROJECT_ROOT, "data", "sample_satellite", f"{region_code}_sentinel2_synthetic.tif")
        bands, meta = SatelliteDataLoader.generate_synthetic_scene(
            synth_path, region_name=region_code, width=500, height=500
        )

    # 1. Feature Extraction
    features = extract_spectral_features(
        bands, ndvi_threshold=ndvi_cut, ndwi_threshold=ndwi_cut, apply_masking=True
    )

    # 2. Clustering
    clf = SpectralKMeansClassifier(n_clusters=k, random_state=42)
    k_opt = None
    if opt_k:
        k_opt = clf.evaluate_optimal_k(features, k_range=range(2, 8))
        clf.n_clusters = k_opt["optimal_k"]

    cluster_map, likelihood_map, stats_df = clf.fit_predict(features)
    target_clusters = stats_df.loc[stats_df["rank"] <= 2, "cluster_id"].tolist()

    # 3. Validation
    gt_path = os.path.join(PROJECT_ROOT, "data", "ground_truth", "gsi_manganese_deposits.csv")
    validator = GeologicalValidator(gt_path, meta)
    val_summary = validator.evaluate_clusters(
        cluster_map=cluster_map,
        likelihood_map=likelihood_map,
        target_cluster_ids=target_clusters,
        buffer_radii_km=[0.25, 0.5, 1.0, 2.0]
    )

    return bands, meta, features, cluster_map, likelihood_map, stats_df, val_summary, k_opt


# Determine region key
if "Balaghat" in region_option:
    reg_code = "balaghat"
elif "Sandur" in region_option:
    reg_code = "sandur"
else:
    reg_code = "custom"

custom_temp_path = None
if custom_file is not None:
    custom_temp_path = os.path.join(PROJECT_ROOT, "data", "sample_satellite", "uploaded_temp.tif")
    with open(custom_temp_path, "wb") as f:
        f.write(custom_file.read())

with st.spinner("Processing satellite spectral bands and running K-Means clustering..."):
    bands, meta, features, cluster_map, likelihood_map, stats_df, val_summary, k_opt = load_and_process(
        reg_code, k_clusters, auto_optimize_k, ndvi_cutoff, ndwi_cutoff, custom_temp_path
    )

# Dashboard Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Satellite Imagery",
    "Spectral Mineral Indices",
    "K-Means Clustering",
    "GSI Ground Truth Validation",
    "Export & Geological Summary"
])

# -----------------------------------------------------------------------------
# TAB 1: SATELLITE IMAGERY
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Candidate Mining Area & Satellite Scene Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Natural Color RGB Composite (B4 - Red, B3 - Green, B2 - Blue)**")
        rgb_nat = SpectralVisualizer.get_rgb_composite(bands, ["red", "green", "blue"])
        fig_nat, ax_nat = plt.subplots(figsize=(7, 7))
        ax_nat.imshow(rgb_nat)
        ax_nat.axis("off")
        st.pyplot(fig_nat, use_container_width=True)
        plt.close(fig_nat)

    with col2:
        st.markdown("**False Color Infrared (B8 - NIR, B4 - Red, B3 - Green)**")
        rgb_cir = SpectralVisualizer.get_rgb_composite(bands, ["nir", "red", "green"])
        fig_cir, ax_cir = plt.subplots(figsize=(7, 7))
        ax_cir.imshow(rgb_cir)
        ax_cir.axis("off")
        st.pyplot(fig_cir, use_container_width=True)
        plt.close(fig_cir)

    st.markdown("---")
    st.markdown("#### Scene Metadata & Exploration Parameters")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Grid Dimensions", f"{meta['width']} x {meta['height']} px")
    with m_col2:
        st.metric("Spatial Reference", meta.get("crs", "EPSG:4326"))
    with m_col3:
        b = meta["bounds"]
        st.metric("Bounding Box", f"{b[0]:.2f}°E to {b[2]:.2f}°E")
    with m_col4:
        st.metric("Sensor Configuration", "6 Multi-Spectral Bands (S2/L8)")

# -----------------------------------------------------------------------------
# TAB 2: SPECTRAL MINERAL INDICES
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Diagnostic Remote Sensing Mineral Signatures")
    st.markdown("""
    Manganese oxides (pyrolusite, psilomelane, braunite) exhibit intense charge-transfer absorption in the UV-blue region 
    coupled with supergene ferric/ferrous iron association. Band ratioing isolates these signatures while mitigating topographic illumination effects.
    """)

    idx_col1, idx_col2 = st.columns(2)
    
    with idx_col1:
        st.markdown("##### Iron Oxide / Gossan Index ($B_4 / B_2$)")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        ioi = np.where(features["valid_mask"], features["iron_oxide_index"], np.nan)
        im1 = ax1.imshow(ioi, cmap="YlOrRd", vmin=np.nanpercentile(ioi, 5), vmax=np.nanpercentile(ioi, 95))
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.markdown("##### Ferrous Iron Index ($B_{12} / B_8$)")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fii = np.where(features["valid_mask"], features["ferrous_iron_index"], np.nan)
        im2 = ax2.imshow(fii, cmap="magma", vmin=np.nanpercentile(fii, 5), vmax=np.nanpercentile(fii, 95))
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with idx_col2:
        st.markdown("##### Manganese Composite Ratio ($[B_4 \\times B_{12}] / [B_2 \\times B_8]$)")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        mocr = np.where(features["valid_mask"], features["manganese_composite_ratio"], np.nan)
        im3 = ax3.imshow(mocr, cmap="inferno", vmin=np.nanpercentile(mocr, 5), vmax=np.nanpercentile(mocr, 95))
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.markdown("##### Crosta Feature-Oriented PCA (Oxide Component)")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        fpca = np.where(features["valid_mask"], features["crosta_oxide_pc"], np.nan)
        im4 = ax4.imshow(fpca, cmap="viridis", vmin=np.nanpercentile(fpca, 5), vmax=np.nanpercentile(fpca, 95))
        ax4.axis("off")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    st.markdown("---")
    st.subheader("Endmember Spectral Signatures")
    fig_spec, ax_spec = plt.subplots(figsize=(10, 4.5))
    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
    wavelengths = [490, 560, 665, 842, 1610, 2190]
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats_df)))

    for i, (_, row) in enumerate(stats_df.iterrows()):
        vals = [row.get(b, 0.2) for b in band_names]
        lbl = f"Rank {row['rank']}: {row['likelihood_tier'][:28]} (Score: {row['manganese_score']}%)"
        ax_spec.plot(wavelengths, vals, marker="s", lw=2.2, color=colors[i], label=lbl)

    ax_spec.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax_spec.set_ylabel("Surface Reflectance", fontsize=11, fontweight="bold")
    ax_spec.set_xticks(wavelengths)
    ax_spec.set_xticklabels(["B02\n490nm", "B03\n560nm", "B04\n665nm", "B08\n842nm", "B11\n1610nm", "B12\n2190nm"])
    ax_spec.grid(True, linestyle="--", alpha=0.5)
    ax_spec.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5)
    st.pyplot(fig_spec, use_container_width=True)
    plt.close(fig_spec)

# -----------------------------------------------------------------------------
# TAB 3: K-MEANS CLUSTERING
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Unsupervised Spectral Clustering & Prospectivity Scoring")

    if k_opt:
        st.info(f"Optimal Cluster Count K = {k_opt['optimal_k']} selected based on peak Silhouette score.")
        fig_k, ax_k = plt.subplots(1, 2, figsize=(10, 3.5))
        ax_k[0].plot(k_opt["k_values"], k_opt["inertias"], marker="o", color="navy")
        ax_k[0].set_title("Elbow Inertia Curve")
        ax_k[0].set_xlabel("K Clusters")
        ax_k[0].grid(True, linestyle="--", alpha=0.5)

        ax_k[1].plot(k_opt["k_values"], k_opt["silhouettes"], marker="s", color="crimson")
        ax_k[1].set_title("Silhouette Score Curve")
        ax_k[1].set_xlabel("K Clusters")
        ax_k[1].grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_k, use_container_width=True)
        plt.close(fig_k)

    col_map1, col_map2 = st.columns(2)
    
    with col_map1:
        st.markdown("**Segmented Litho-Mineral Cluster Raster**")
        fig_cl, ax_cl = plt.subplots(figsize=(7, 6))
        # Discrete colormap
        display_cl = np.full_like(cluster_map, fill_value=np.nan, dtype=np.float32)
        rank_lookup = dict(zip(stats_df["cluster_id"], stats_df["rank"]))
        for cid, r in rank_lookup.items():
            display_cl[cluster_map == cid] = r

        im_cl = ax_cl.imshow(display_cl, cmap="turbo", vmin=1, vmax=len(stats_df))
        ax_cl.axis("off")
        plt.colorbar(im_cl, ax=ax_cl, fraction=0.046, pad=0.04, label="Cluster Rank (1 = Top Anomaly)")
        st.pyplot(fig_cl, use_container_width=True)
        plt.close(fig_cl)

    with col_map2:
        st.markdown("**Continuous Manganese Likelihood Score (%)**")
        fig_lik, ax_lik = plt.subplots(figsize=(7, 6))
        im_lik = ax_lik.imshow(likelihood_map * 100.0, cmap="hot_r", vmin=0, vmax=100)
        ax_lik.axis("off")
        plt.colorbar(im_lik, ax=ax_lik, fraction=0.046, pad=0.04, label="Likelihood Score (%)")
        st.pyplot(fig_lik, use_container_width=True)
        plt.close(fig_lik)

    st.markdown("---")
    st.markdown("#### Litho-Unit Prospectivity Summary Table")
    disp_df = stats_df[[
        "rank", "cluster_id", "likelihood_tier", "manganese_score", "area_km2",
        "pct_of_valid_area", "iron_oxide_index", "manganese_composite_ratio", "ferrous_iron_index"
    ]].rename(columns={
        "rank": "Rank",
        "cluster_id": "Cluster ID",
        "likelihood_tier": "Geological Designation",
        "manganese_score": "Mn Score (%)",
        "area_km2": "Area (sq km)",
        "pct_of_valid_area": "% of Scene",
        "iron_oxide_index": "Mean IOI (B4/B2)",
        "manganese_composite_ratio": "Mean MOCR",
        "ferrous_iron_index": "Mean FII (B12/B8)"
    })
    st.dataframe(disp_df, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: GSI GROUND TRUTH VALIDATION
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Geological Survey of India (GSI) Deposit Validation")

    m = val_summary["metrics_at_1km"]
    b_res = val_summary["buffer_results"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Exact Pixel Hit Rate", f"{val_summary['exact_hit_rate_pct']}%")
    with c2:
        st.metric("500m Buffer Hit Rate", f"{b_res['0.5 km']['hit_rate_pct']}%")
    with c3:
        st.metric("1.0 km Buffer Hit Rate", f"{b_res['1.0 km']['hit_rate_pct']}%")
    with c4:
        st.metric("Model F1-Score", f"{m['f1_score']:.3f}")

    st.markdown("---")
    v_col1, v_col2 = st.columns([1, 1])

    with v_col1:
        st.markdown("**Deposit Validation Map (Confirmed Mines vs Predicted Targets)**")
        fig_vmap, ax_vmap = plt.subplots(figsize=(7, 6.5))
        ax_vmap.imshow(likelihood_map * 100.0, cmap="gist_earth_r", vmin=0, vmax=100)
        
        pts_df = val_summary["evaluated_points_df"]
        dep = pts_df[pts_df["is_deposit"] == 1]
        ctrl = pts_df[pts_df["is_deposit"] == 0]

        ax_vmap.scatter(
            dep["pixel_col"], dep["pixel_row"],
            c="#ff1744", marker="*", s=220, edgecolors="white", linewidths=1.5,
            label="GSI Confirmed Mine", zorder=5
        )
        for _, row in dep.iterrows():
            ax_vmap.annotate(
                row["name"].split(" ")[0],
                (row["pixel_col"] + 5, row["pixel_row"] - 5),
                fontsize=8.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7, ec="none")
            )

        if len(ctrl) > 0:
            ax_vmap.scatter(
                ctrl["pixel_col"], ctrl["pixel_row"],
                c="#4caf50", marker="o", s=80, edgecolors="white", linewidths=1.2,
                label="Barren Control Point", zorder=5
            )

        ax_vmap.legend(loc="lower right", fontsize=9)
        ax_vmap.axis("off")
        st.pyplot(fig_vmap, use_container_width=True)
        plt.close(fig_vmap)

    with v_col2:
        st.markdown("**Confusion Matrix & Buffer Sensitivity**")
        fig_stats, ax_s = plt.subplots(2, 1, figsize=(6, 6))

        # Confusion matrix
        cm_data = np.array([
            [m["true_positives"], m["false_negatives"]],
            [m["false_positives"], m["true_negatives"]]
        ])
        ax_s[0].imshow(cm_data, cmap="Blues", alpha=0.7)
        for (i, j), val in np.ndenumerate(cm_data):
            ax_s[0].text(j, i, f"{val}", ha="center", va="center", fontsize=14, fontweight="bold")
        ax_s[0].set_xticks([0, 1])
        ax_s[0].set_yticks([0, 1])
        ax_s[0].set_xticklabels(["Predicted Target", "Predicted Barren"])
        ax_s[0].set_yticklabels(["GSI Deposit", "Control Point"])
        ax_s[0].set_title(f"Confusion Matrix (1.0 km Buffer) | Accuracy: {m['accuracy']*100:.1f}%", fontsize=11, fontweight="bold")

        # Buffer curve
        radii = [v["buffer_km"] * 1000 for v in b_res.values()]
        rates = [v["hit_rate_pct"] for v in b_res.values()]
        ax_s[1].plot(radii, rates, marker="o", lw=2.2, color="#d95f02")
        ax_s[1].set_xlabel("Buffer Radius (m)", fontsize=10)
        ax_s[1].set_ylabel("Hit Rate (%)", fontsize=10)
        ax_s[1].set_ylim(-5, 105)
        ax_s[1].set_title("Detection Sensitivity Curve", fontsize=11, fontweight="bold")
        ax_s[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig_stats, use_container_width=True)
        plt.close(fig_stats)

    st.markdown("---")
    st.markdown("#### Confirmed Deposit Spatial Evaluation Records")
    cols_to_show = [
        "deposit_id", "name", "formation", "ore_type", "mn_grade_pct",
        "dist_to_target_km", "sampled_likelihood", "hit_500m", "hit_1000m"
    ]
    st.dataframe(pts_df[cols_to_show].rename(columns={
        "dist_to_target_km": "Dist to Target (km)",
        "sampled_likelihood": "Likelihood Score",
        "hit_500m": "Hit (500m)",
        "hit_1000m": "Hit (1km)"
    }), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: EXPORT & GEOLOGICAL SUMMARY
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("GIS Deliverables & Field Prospecting Package")
    st.markdown("""
    Export geospatial datasets compatible with QGIS, ArcGIS, Google Earth, and Field GPS units.
    """)

    out_dir = os.path.join(PROJECT_ROOT, "data", "outputs")
    clust_tif = os.path.join(out_dir, f"{reg_code}_kmeans_clusters.tif")
    likeli_tif = os.path.join(out_dir, f"{reg_code}_manganese_likelihood.tif")
    stats_csv = os.path.join(out_dir, f"{reg_code}_cluster_statistics.csv")
    pts_csv = os.path.join(out_dir, f"{reg_code}_deposit_validation_results.csv")

    ex_c1, ex_c2 = st.columns(2)
    with ex_c1:
        if os.path.exists(clust_tif):
            with open(clust_tif, "rb") as f:
                st.download_button(
                    "📥 Download Classified K-Means GeoTIFF",
                    data=f.read(),
                    file_name=f"{reg_code}_kmeans_clusters.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
        if os.path.exists(stats_csv):
            with open(stats_csv, "rb") as f:
                st.download_button(
                    "📊 Download Cluster Spectral Signatures (CSV)",
                    data=f.read(),
                    file_name=f"{reg_code}_cluster_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with ex_c2:
        if os.path.exists(likeli_tif):
            with open(likeli_tif, "rb") as f:
                st.download_button(
                    "📥 Download Continuous Prospectivity GeoTIFF",
                    data=f.read(),
                    file_name=f"{reg_code}_manganese_likelihood.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
        if os.path.exists(pts_csv):
            with open(pts_csv, "rb") as f:
                st.download_button(
                    "📍 Download GSI Deposit Validation Records (CSV)",
                    data=f.read(),
                    file_name=f"{reg_code}_deposit_validation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.markdown("---")
    st.markdown("#### Geological Executive Summary (SIH26009)")
    st.info(f"""
    **Project Assessment Summary for Region: {region_option}**
    
    1. **Spectral Feature Extraction**: Diagnostically isolated high ferric/ferrous oxide absorption (Iron Oxide Ratio B4/B2) and Manganese Composite Ratio (MOCR). Non-mineralized canopies and hydrologic bodies were successfully masked via NDVI/NDWI.
    2. **K-Means Mineral Segmentation**: Formations were segmented into {k_clusters} discrete spectral litho-units. Cluster Rank 1 and Rank 2 correlate to outcropping gonditic horizons, supergene lateritoid caps, and mine cuts.
    3. **GSI Ground Truth Spatial Validation**: Validated against confirmed deposits in the Sausar Manganese Belt. Achieved an exact pixel hit rate of **{val_summary['exact_hit_rate_pct']}%** and a 1.0 km structural buffer hit rate of **{b_res['1.0 km']['hit_rate_pct']}%** with an overall F1-Score of **{m['f1_score']:.3f}**.
    """)
