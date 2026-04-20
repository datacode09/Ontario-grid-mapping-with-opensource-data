"""Local SAR processor — rasterio/numpy fallback when GEE unavailable.

Steps:
  1. Mosaic multi-scene tiles per date
  2. Convert dB: 10*log10(amplitude²)
  3. Apply Lee speckle filter
  4. Compute log-ratio (before/after)
  5. Threshold → binary change mask

Uses ASF RTC downloads (Source 21) as input.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..sar import _SAR_DISCLAIMER
from ..utils.config_loader import load_sar_settings
from ..utils.logger import get_logger
from ..utils.raster_utils import (
    read_full_band, write_cog, apply_lee_filter, linear_to_db
)

log = get_logger(__name__)

try:
    import rasterio
    from rasterio.merge import merge as rasterio_merge
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


class LocalSARProcessor:
    """Local (CPU) SAR processing pipeline for Sentinel-1 RTC scenes."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        cfg = load_sar_settings().get("sar", {})
        self._cfg = cfg
        self.output_dir = output_dir or Path("data/processed/sar_derived")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def mosaic_scenes(
        self,
        scene_paths: list[Path],
        output_path: Optional[Path] = None,
        band: int = 1,
        date_label: str = "",
    ) -> Optional[Path]:
        """Merge multiple Sentinel-1 RTC GeoTIFF tiles into a single mosaic.

        Parameters
        ----------
        scene_paths:   List of single-band GeoTIFF paths to mosaic.
        output_path:   Output path for merged GeoTIFF.
        band:          Band index to extract from each scene.
        date_label:    Date label for output filename.

        Returns
        -------
        Path to mosaicked GeoTIFF, or None on failure.
        """
        if not _RASTERIO_AVAILABLE:
            log.warning("LocalSARProcessor: rasterio unavailable")
            return None

        if not scene_paths:
            log.warning("LocalSARProcessor: no scenes to mosaic")
            return None

        if output_path is None:
            output_path = self.output_dir / f"s1_mosaic_{date_label}.tif"

        try:
            datasets = [rasterio.open(p) for p in scene_paths if Path(p).exists()]
            if not datasets:
                return None

            mosaic, transform = rasterio_merge(datasets, indexes=[band])
            profile = datasets[0].profile.copy()
            profile.update({
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": 1,
            })

            data = mosaic[0]
            for ds in datasets:
                ds.close()

            crs_str = str(profile.get("crs", "EPSG:32617"))
            return write_cog(
                data, output_path, crs=crs_str, transform=transform,
                dtype=str(data.dtype),
                tags={"source": "SAR_contextual_RS", "date": date_label}
            )
        except Exception as exc:
            log.error("Mosaic failed: %s", exc)
            return None

    def process_rtc_to_db(self, rtc_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Convert an RTC amplitude GeoTIFF to dB scale with speckle filtering."""
        if not Path(rtc_path).exists():
            return None

        data, profile = read_full_band(rtc_path, band=1)
        data = data.astype(np.float32)

        # Apply Lee speckle filter
        kernel = self._cfg.get("speckle_filter", {}).get("kernel_size", 5)
        filtered = apply_lee_filter(data, kernel_size=kernel)

        # Convert to dB
        db_data = linear_to_db(filtered)

        if output_path is None:
            output_path = rtc_path.parent / (rtc_path.stem + "_db.tif")

        crs_str = str(profile.get("crs", "EPSG:32617"))
        return write_cog(
            db_data, output_path, crs=crs_str,
            transform=profile.get("transform"), dtype="float32",
            tags={"source": "SAR_contextual_RS", "unit": "dB"}
        )

    def build_reference_composite(
        self,
        scene_db_paths: list[Path],
        output_path: Optional[Path] = None,
        method: str = "median",
    ) -> Optional[Path]:
        """Build a dry-season reference composite from multiple dB scenes.

        Computes median (or mean) across scenes for a stable reference.
        """
        valid_paths = [p for p in scene_db_paths if Path(p).exists()]
        if not valid_paths:
            log.warning("Reference composite: no valid scene paths")
            return None

        arrays = []
        profile = None
        for p in valid_paths:
            try:
                arr, pf = read_full_band(p, band=1)
                arrays.append(arr.astype(np.float32))
                if profile is None:
                    profile = pf
            except Exception as exc:
                log.debug("Could not read %s: %s", p, exc)

        if not arrays:
            return None

        # Align to minimum common shape
        min_rows = min(a.shape[0] for a in arrays)
        min_cols = min(a.shape[1] for a in arrays)
        stack = np.stack([a[:min_rows, :min_cols] for a in arrays], axis=0)

        composite = np.nanmedian(stack, axis=0) if method == "median" else np.nanmean(stack, axis=0)

        if output_path is None:
            output_path = self.output_dir / "reference_composite_db.tif"

        crs_str = str(profile.get("crs", "EPSG:32617")) if profile else "EPSG:32617"
        transform = profile.get("transform") if profile else None

        return write_cog(
            composite, output_path, crs=crs_str, transform=transform,
            dtype="float32",
            tags={
                "source": "SAR_contextual_RS",
                "product_type": "reference_composite",
                "n_scenes": str(len(arrays)),
                "disclaimer": _SAR_DISCLAIMER,
            }
        )
