"""Raster I/O helpers — COG read/write, window reads, reprojection.

Shared between the SAR module and gridfinder_runner.
All SAR outputs are written as Cloud-Optimised GeoTIFFs (COG).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import rasterio
    from rasterio import windows
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.io import MemoryFile
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


def _check_rasterio() -> None:
    if not _RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for raster operations. "
            "Install with: pip install rasterio"
        )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_cog_window(
    path: Path | str,
    bbox: Tuple[float, float, float, float],
    crs: str = "EPSG:4326",
    band: int = 1,
) -> Tuple[np.ndarray, "rasterio.transform.Affine", str]:
    """Read a spatial window from a (Cloud-Optimised) GeoTIFF.

    Parameters
    ----------
    path:   Path to GeoTIFF / COG file.
    bbox:   (west, south, east, north) in ``crs`` coordinates.
    crs:    CRS of the bbox coordinates.
    band:   Band index (1-based).

    Returns
    -------
    (data_array, transform, crs_wkt)
    """
    _check_rasterio()
    with rasterio.open(path) as src:
        src_crs = src.crs
        target_crs = CRS.from_string(crs)

        # Reproject bbox to source CRS if needed
        if src_crs != target_crs:
            from rasterio.warp import transform_bounds
            bbox = transform_bounds(target_crs, src_crs, *bbox)

        win = windows.from_bounds(*bbox, transform=src.transform)
        data = src.read(band, window=win)
        win_transform = src.window_transform(win)
        return data, win_transform, src_crs.to_wkt()


def read_full_band(path: Path | str, band: int = 1) -> Tuple[np.ndarray, dict]:
    """Read a complete band from a GeoTIFF, returning (array, profile)."""
    _check_rasterio()
    with rasterio.open(path) as src:
        data = src.read(band)
        return data, src.profile.copy()


# ---------------------------------------------------------------------------
# Writing COG
# ---------------------------------------------------------------------------

def write_cog(
    data: np.ndarray,
    path: Path | str,
    crs: str,
    transform,
    dtype: Optional[str] = None,
    nodata: Optional[float] = None,
    blocksize: int = 512,
    compression: str = "deflate",
    overview_levels: Optional[list] = None,
    tags: Optional[dict] = None,
) -> Path:
    """Write a numpy array as a Cloud-Optimised GeoTIFF.

    Follows the COG spec: block-tiled, internal overviews, DEFLATE compression.
    """
    _check_rasterio()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = dtype or str(data.dtype)
    overview_levels = overview_levels or [2, 4, 8, 16]
    bands = 1 if data.ndim == 2 else data.shape[0]

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": data.shape[-1] if data.ndim > 1 else data.shape[0],
        "height": data.shape[-2] if data.ndim > 1 else 1,
        "count": bands,
        "crs": CRS.from_string(crs),
        "transform": transform,
        "nodata": nodata,
        "compress": compression,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "interleave": "band",
        "copy_src_overviews": True,
    }

    # Write to in-memory first, then copy as COG with overviews
    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            if data.ndim == 2:
                tmp.write(data, 1)
            else:
                tmp.write(data)
            if tags:
                tmp.update_tags(**tags)

            # Build overviews inside the MemoryFile
            tmp.build_overviews(overview_levels, Resampling.average)
            tmp.update_tags(ns="rio_overview", resampling="average")

        # Copy from memory to disk as COG
        with mem.open() as tmp:
            profile.update({"copy_src_overviews": True})
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(tmp.read())
                dst.build_overviews(overview_levels, Resampling.average)
                if tags:
                    dst.update_tags(**tags)

    return path


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------

def reproject_raster(
    src_path: Path | str,
    dst_path: Path | str,
    dst_crs: str = "EPSG:4326",
    resampling: str = "bilinear",
) -> Path:
    """Reproject a GeoTIFF to a new CRS and save."""
    _check_rasterio()
    resample_method = getattr(Resampling, resampling)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs, transform=transform, width=width, height=height
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resample_method,
                )
    return dst_path


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def db_to_linear(db_array: np.ndarray) -> np.ndarray:
    """Convert dB values to linear scale: linear = 10^(dB/10)."""
    return np.power(10, db_array / 10.0)


def linear_to_db(linear_array: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Convert linear amplitude to dB: dB = 10 * log10(amplitude²)."""
    return 10.0 * np.log10(np.maximum(linear_array ** 2, eps))


def apply_lee_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a simplified Lee speckle filter to a 2-D SAR intensity array."""
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(data.astype(float), size=kernel_size)
    mean_sq = uniform_filter(data.astype(float) ** 2, size=kernel_size)
    variance = mean_sq - mean ** 2
    noise_var = np.mean(variance)

    weight = variance / (variance + noise_var + 1e-10)
    return mean + weight * (data - mean)
