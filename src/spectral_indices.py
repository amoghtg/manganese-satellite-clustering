"""
Spectral mineral-signature feature extraction for Manganese prospecting.
Supports Sentinel-2 MSI and Landsat-8/9 OLI spectral band architectures.
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA


# Standard band mapping schemas
BAND_MAPPINGS = {
    "sentinel2": {
        "blue": 1,    # B02 (490 nm)
        "green": 2,   # B03 (560 nm)
        "red": 3,     # B04 (665 nm)
        "nir": 4,     # B08 (842 nm)
        "swir1": 5,   # B11 (1610 nm)
        "swir2": 6,   # B12 (2190 nm)
    },
    "landsat8": {
        "blue": 1,    # Band 2 (482 nm)
        "green": 2,   # Band 3 (561 nm)
        "red": 3,     # Band 4 (655 nm)
        "nir": 4,     # Band 5 (865 nm)
        "swir1": 5,   # Band 6 (1609 nm)
        "swir2": 6,   # Band 7 (2201 nm)
    }
}


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Safe element-wise array division with zero/nan prevention."""
    den = np.where(np.abs(denominator) < eps, eps, denominator)
    result = np.divide(numerator, den)
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return result


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Vegetation Index (NDVI).
    NDVI = (NIR - Red) / (NIR + Red)
    Used to mask out dense canopies obscuring bedrock geology.
    """
    return safe_divide(nir - red, nir + red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Water Index (NDWI - McFeeters).
    NDWI = (Green - NIR) / (Green + NIR)
    Used to mask out lakes, reservoirs, and rivers.
    """
    return safe_divide(green - nir, green + nir)


def compute_iron_oxide_ratio(red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """
    Iron Oxide / Gossan Index.
    Ratio = Red / Blue (Sentinel-2 B4 / B2, Landsat B4 / B2)
    Pyrolusite/manganese deposits strongly associate with iron oxide alteration (ferromanganese gossans).
    """
    return safe_divide(red, blue)


def compute_ferrous_iron_ratio(swir2: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Ferrous Iron Index (FII).
    Ratio = SWIR2 / NIR (Sentinel-2 B12 / B8, Landsat B7 / B5)
    Highlights ferrous mineral signatures in silicate and carbonate host rocks.
    """
    return safe_divide(swir2, nir)


def compute_clay_alteration_ratio(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Clay & Hydrothermal Alteration Index (CAI).
    Ratio = SWIR1 / SWIR2 (Sentinel-2 B11 / B12, Landsat B6 / B7)
    Detects Al-OH alteration halos (kaolinite, illite) common along manganese-bearing shear zones.
    """
    return safe_divide(swir1, swir2)


def compute_manganese_composite_ratio(
    red: np.ndarray, blue: np.ndarray, nir: np.ndarray, swir2: np.ndarray
) -> np.ndarray:
    """
    Manganese Oxide Composite Ratio (MOCR).
    MOCR = (Red * SWIR2) / (Blue * NIR)
    Exploits the steep UV-blue absorption of Mn-oxides paired with lower SWIR2 reflectance
    relative to surrounding quartz-mica schists and phyllites.
    """
    numerator = red * swir2
    denominator = blue * nir
    return safe_divide(numerator, denominator)


def compute_normalized_manganese_index(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Normalized Difference Manganese Index (NMI).
    NMI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
    Discriminates opaque manganese oxide crusts from surrounding country rocks.
    """
    return safe_divide(swir1 - swir2, swir1 + swir2)


def compute_crosta_fpca(
    blue: np.ndarray, red: np.ndarray, swir1: np.ndarray, swir2: np.ndarray, mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Feature-Oriented Principal Component Analysis (Crosta Technique) for Oxide detection.
    Analyzes covariance between Blue, Red, SWIR1, SWIR2 to isolate mineral oxide component.
    
    Returns:
        fpca_image: 2D array of the selected oxide PC
        eigenvectors: PC loadings matrix
        oxide_pc_index: Index (0-3) of the PC exhibiting highest Red vs Blue contrast
    """
    H, W = blue.shape
    stacked = np.stack([blue.flatten(), red.flatten(), swir1.flatten(), swir2.flatten()], axis=1)
    
    if mask is not None:
        valid_idx = np.where(mask.flatten())[0]
        if len(valid_idx) > 100:
            sample_data = stacked[valid_idx]
        else:
            sample_data = stacked
    else:
        sample_data = stacked

    # Fit PCA
    pca = PCA(n_components=4)
    pca.fit(sample_data)
    
    # Identify oxide PC: Look for component with largest difference between Red (col 1) and Blue (col 0)
    loadings = pca.components_
    oxide_contrast = np.abs(loadings[:, 1] - loadings[:, 0])
    oxide_pc_index = int(np.argmax(oxide_contrast))
    
    # If loading for Red is negative, invert so that high values indicate oxide presence
    sign = 1.0 if (loadings[oxide_pc_index, 1] > loadings[oxide_pc_index, 0]) else -1.0
    
    all_pcs = pca.transform(stacked)
    oxide_pc = (sign * all_pcs[:, oxide_pc_index]).reshape((H, W))
    
    return oxide_pc, loadings, oxide_pc_index


def extract_spectral_features(
    bands: Dict[str, np.ndarray],
    ndvi_threshold: float = 0.45,
    ndwi_threshold: float = 0.10,
    apply_masking: bool = True
) -> Dict[str, np.ndarray]:
    """
    Comprehensive spectral feature extractor.
    
    Inputs:
        bands: Dictionary with keys 'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
        ndvi_threshold: Vegetation mask cutoff
        ndwi_threshold: Water body mask cutoff
        apply_masking: If True, masks out dense vegetation and open water
        
    Returns:
        Dictionary of raw bands, computed indices, mask, and feature stack
    """
    blue = bands["blue"].astype(np.float32)
    green = bands["green"].astype(np.float32)
    red = bands["red"].astype(np.float32)
    nir = bands["nir"].astype(np.float32)
    swir1 = bands["swir1"].astype(np.float32)
    swir2 = bands["swir2"].astype(np.float32)
    
    # Environmental Indices
    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    
    # Geological Mask: False where water or dense vegetation
    if apply_masking:
        valid_mask = (ndvi < ndvi_threshold) & (ndwi < ndwi_threshold)
    else:
        valid_mask = np.ones_like(ndvi, dtype=bool)
        
    # Mineral Indices
    ioi = compute_iron_oxide_ratio(red, blue)
    fii = compute_ferrous_iron_ratio(swir2, nir)
    cai = compute_clay_alteration_ratio(swir1, swir2)
    mocr = compute_manganese_composite_ratio(red, blue, nir, swir2)
    nmi = compute_normalized_manganese_index(swir1, swir2)
    
    # Crosta FPCA for oxide enhancement
    fpca_oxide, loadings, pc_idx = compute_crosta_fpca(blue, red, swir1, swir2, mask=valid_mask)
    
    return {
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
        "swir1": swir1,
        "swir2": swir2,
        "ndvi": ndvi,
        "ndwi": ndwi,
        "valid_mask": valid_mask,
        "iron_oxide_index": ioi,
        "ferrous_iron_index": fii,
        "clay_alteration_index": cai,
        "manganese_composite_ratio": mocr,
        "normalized_manganese_index": nmi,
        "crosta_oxide_pc": fpca_oxide,
        "fpca_loadings": loadings,
        "fpca_pc_idx": pc_idx
    }
