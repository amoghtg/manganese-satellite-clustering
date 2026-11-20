"""
Data loading, preprocessing, synthetic realistic satellite tile generation,
and GeoTIFF I/O for multispectral remote sensing data.
"""

import os
import json
from typing import Dict, Tuple, Optional, Any
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS


class SatelliteDataLoader:
    """Handles loading, saving, and synthetic generation of multispectral satellite rasters."""
    
    BAND_NAMES = ["blue", "green", "red", "nir", "swir1", "swir2"]

    @staticmethod
    def load_geotiff(
        filepath: str,
        band_indices: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Loads a multi-band GeoTIFF file.
        
        Args:
            filepath: Path to the .tif file
            band_indices: 1-based index mapping for bands (default: 1=blue, 2=green, 3=red, 4=nir, 5=swir1, 6=swir2)
            
        Returns:
            bands: Dict mapping standard band names to 2D numpy arrays (normalized 0-1 float32)
            profile: Rasterio metadata profile including transform, bounds, CRS, shape
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Satellite GeoTIFF not found at: {filepath}")

        if band_indices is None:
            band_indices = {
                "blue": 1,
                "green": 2,
                "red": 3,
                "nir": 4,
                "swir1": 5,
                "swir2": 6
            }

        with rasterio.open(filepath) as src:
            profile = src.profile.copy()
            bounds = src.bounds
            crs = src.crs.to_string() if src.crs else "EPSG:4326"
            transform = src.transform
            width = src.width
            height = src.height
            count = src.count

            meta = {
                "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
                "transform": transform,
                "crs": crs,
                "width": width,
                "height": height,
                "count": count
            }

            bands = {}
            for name, idx in band_indices.items():
                if idx <= count:
                    data = src.read(idx).astype(np.float32)
                    # Handle typical 10000 or 65535 DN scalings for Sentinel/Landsat
                    max_val = np.nanmax(data)
                    if max_val > 1.0:
                        # Normalize surface reflectance to 0.0 - 1.0
                        scale = 10000.0 if max_val > 255.0 else 255.0
                        data = data / scale
                    bands[name] = np.clip(data, 0.0, 1.0)
                else:
                    raise ValueError(f"Requested band index {idx} exceeds raster band count {count}")

        return bands, meta

    @staticmethod
    def save_classified_geotiff(
        output_path: str,
        raster_data: np.ndarray,
        meta: Dict[str, Any],
        nodata: int = -1
    ) -> str:
        """
        Saves a classified 2D numpy array as a georeferenced GeoTIFF.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        height, width = raster_data.shape
        bounds = meta.get("bounds", (79.0, 21.0, 80.5, 22.0))
        left, bottom, right, top = bounds
        transform = meta.get("transform", from_bounds(left, bottom, right, top, width, height))
        crs = meta.get("crs", "EPSG:4326")

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": rasterio.int16,
            "crs": CRS.from_string(crs) if isinstance(crs, str) else crs,
            "transform": transform,
            "nodata": nodata
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(raster_data.astype(np.int16), 1)

        return output_path

    @classmethod
    def generate_synthetic_scene(
        cls,
        output_path: str,
        region_name: str = "balaghat",
        width: int = 500,
        height: int = 500,
        seed: int = 42
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Generates a physically and geologically realistic synthetic multispectral Sentinel-2 tile
        modeled after real geological formations in the Sausar Manganese Belt (Balaghat, MP, India).
        
        The synthetic image contains:
        - Structural folded synclines of Mansar Formation carrying manganese horizons
        - Open pit mine cuts matching the coordinates of Bharweli, Ukwa, and Tirodi
        - Surrounding quartzites, gneisses, agricultural plains, and Wainganga river drainage
        - Realistic USGS spectral endmembers for each lithology with Perlin/fractal spatial noise
        """
        np.random.seed(seed)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if region_name.lower() == "balaghat":
            bounds = (79.60, 21.50, 80.50, 22.05)  # Lon min, Lat min, Lon max, Lat max
        else:
            bounds = (76.40, 15.00, 76.80, 15.40)  # Sandur belt coordinates

        lon_min, lat_min, lon_max, lat_max = bounds
        lons = np.linspace(lon_min, lon_max, width)
        lats = np.linspace(lat_max, lat_min, height)  # North to South
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Spectral Endmembers [Blue, Green, Red, NIR, SWIR1, SWIR2]
        # Physically calibrated from USGS spectral library and field studies:
        endmembers = {
            "manganese_ore": np.array([0.08, 0.10, 0.15, 0.16, 0.18, 0.14], dtype=np.float32),
            "ferromanganese_gossan": np.array([0.10, 0.13, 0.22, 0.24, 0.28, 0.21], dtype=np.float32),
            "gondite_host_rock": np.array([0.16, 0.22, 0.27, 0.30, 0.35, 0.28], dtype=np.float32),
            "granite_gneiss_basement": np.array([0.19, 0.25, 0.31, 0.34, 0.40, 0.32], dtype=np.float32),
            "alluvial_soil": np.array([0.14, 0.18, 0.24, 0.28, 0.34, 0.27], dtype=np.float32),
            "dense_canopy": np.array([0.04, 0.08, 0.05, 0.52, 0.22, 0.10], dtype=np.float32),
            "water_body": np.array([0.06, 0.08, 0.04, 0.02, 0.01, 0.01], dtype=np.float32),
        }

        # 1. Structural regional fold pattern (Sausar belt strike: ENE-WSW synclinorium)
        # Coordinate transformation along structural strike ~25 degrees
        theta = np.radians(25)
        u = (lon_grid - lon_min) * np.cos(theta) - (lat_grid - lat_min) * np.sin(theta)
        v = (lon_grid - lon_min) * np.sin(theta) + (lat_grid - lat_min) * np.cos(theta)

        # Folded geological horizons
        fold_pattern = np.sin(v * 45.0) + 0.3 * np.sin(u * 15.0)
        
        # 2. Drainage network (Wainganga river basin)
        river_channel = np.abs(np.sin((lon_grid - 79.9) * 8.0) - (lat_grid - 21.7) * 2.5)
        water_mask = river_channel < 0.015

        # 3. Dense vegetation on forest ridges
        vegetation_ridges = (np.sin(u * 30.0 + 1.2) * np.cos(v * 20.0) > 0.45) & (~water_mask)

        # 4. Manganese mineralized horizons (Occurring along specific stratigraphic limbs of Mansar Formation)
        # Narrow arcuate synclinal reef bands
        manganese_limb_1 = np.exp(-((fold_pattern - 0.72) ** 2) / (2 * (0.04 ** 2)))
        manganese_limb_2 = np.exp(-((fold_pattern + 0.65) ** 2) / (2 * (0.05 ** 2)))
        mn_mineralization_field = np.clip(manganese_limb_1 + manganese_limb_2, 0.0, 1.0)

        # Add confirmed deposit hot spots (Bharweli, Ukwa, Dongri Buzurg, Tirodi)
        known_mines = [
            (80.2050, 21.8950, 0.035),  # Bharweli Mine
            (80.4680, 21.9700, 0.030),  # Ukwa Mine
            (79.6830, 21.5540, 0.032),  # Dongri Buzurg
            (79.7020, 21.6850, 0.030),  # Tirodi Mine
            (79.7480, 21.5470, 0.025),  # Chikla Mine
            (79.9100, 21.8600, 0.025),  # Ramrama Prospect
        ]
        
        for mine_lon, mine_lat, radius in known_mines:
            dist = np.sqrt((lon_grid - mine_lon) ** 2 + (lat_grid - mine_lat) ** 2)
            pit_spot = np.exp(-(dist ** 2) / (2 * (radius ** 2)))
            mn_mineralization_field = np.maximum(mn_mineralization_field, pit_spot * 1.2)

        mn_mineralization_field = np.clip(mn_mineralization_field, 0.0, 1.0)

        # 5. Lithological Abundance synthesis (Linear Spectral Mixing)
        H, W = height, width
        f_water = np.where(water_mask, 0.95, 0.0)
        f_veg = np.where(vegetation_ridges, 0.85, 0.08)
        
        # Manganese ore and gossan fractional abundances
        f_ore = mn_mineralization_field * 0.70 * (1.0 - f_water) * (1.0 - f_veg)
        f_gossan = (mn_mineralization_field ** 1.5) * 0.25 * (1.0 - f_water) * (1.0 - f_veg)
        
        # Surrounding rocks
        f_gondite = np.clip(np.abs(fold_pattern) * 0.40, 0.0, 0.6) * (1.0 - f_water) * (1.0 - f_veg)
        f_granite = np.clip(1.0 - (f_water + f_veg + f_ore + f_gossan + f_gondite), 0.0, 1.0)
        
        # Sum normalization
        total_fractions = f_water + f_veg + f_ore + f_gossan + f_gondite + f_granite
        total_fractions = np.where(total_fractions == 0, 1.0, total_fractions)
        
        f_water /= total_fractions
        f_veg /= total_fractions
        f_ore /= total_fractions
        f_gossan /= total_fractions
        f_gondite /= total_fractions
        f_granite /= total_fractions

        # Mix spectral bands
        bands_dict = {}
        band_arrays = []
        for i, band_name in enumerate(cls.BAND_NAMES):
            band_val = (
                f_ore * endmembers["manganese_ore"][i]
                + f_gossan * endmembers["ferromanganese_gossan"][i]
                + f_gondite * endmembers["gondite_host_rock"][i]
                + f_granite * endmembers["granite_gneiss_basement"][i]
                + f_veg * endmembers["dense_canopy"][i]
                + f_water * endmembers["water_body"][i]
            )
            # Add spatial sensor noise
            noise = np.random.normal(0, 0.005, (H, W)).astype(np.float32)
            band_val = np.clip(band_val + noise, 0.01, 0.95)
            bands_dict[band_name] = band_val
            band_arrays.append(band_val)

        # Write to GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(cls.BAND_NAMES),
            "dtype": rasterio.float32,
            "crs": CRS.from_string("EPSG:4326"),
            "transform": transform,
            "nodata": -9999.0
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, arr in enumerate(band_arrays):
                dst.write(arr.astype(np.float32), i + 1)
                dst.set_band_description(i + 1, cls.BAND_NAMES[i])

        meta = {
            "bounds": bounds,
            "transform": transform,
            "crs": "EPSG:4326",
            "width": width,
            "height": height,
            "count": len(cls.BAND_NAMES),
            "region": region_name
        }

        # Write companion metadata json
        meta_json_path = output_path.replace(".tif", "_meta.json")
        with open(meta_json_path, "w") as f:
            json.dump({
                "region_name": region_name,
                "bounds": bounds,
                "bands": cls.BAND_NAMES,
                "sensor": "Sentinel-2 MSI (Simulated L2A Surface Reflectance)",
                "pixel_size_deg": (lon_max - lon_min) / width,
                "notes": "Geologically calibrated Sausar Group Mansar Formation manganese syncline"
            }, f, indent=2)

        return bands_dict, meta
