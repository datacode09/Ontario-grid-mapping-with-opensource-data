"""SAR flood detector — water/flood extent from VV backscatter threshold.

Method:
  1. Reference VV backscatter (dry-season median composite, e.g. Aug–Sep)
  2. Event VV scene (post-rain / spring)
  3. Threshold: VV < (ref - 3dB) → water
  4. Mask out permanent water bodies (LIO hydrography, Source 6)
  5. Output: binary water-extent GeoTIFF (COG)

LABELLING: Output tagged source="SAR_contextual_RS", product_type="flood_extent"
"""
from __future__ import annotations

from datetime import date
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
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False


def detect_flood_extent(
    event_vv_path: Path,
    reference_composite_path: Path,
    permanent_water_mask,
    threshold_db: float = -3.0,
    output_path: Optional[Path] = None,
    event_date: Optional[str] = None,
) -> Optional[Path]:
    """Detect flood-inundated areas by comparing event VV against dry-season reference.

    Parameters
    ----------
    event_vv_path:              Path to post-event Sentinel-1 VV GeoTIFF.
    reference_composite_path:   Path to dry-season median VV composite GeoTIFF.
    permanent_water_mask:       GeoDataFrame of permanent water bodies to exclude.
    threshold_db:               VV drop below reference to flag as water (default -3 dB).
    output_path:                Output COG path. Auto-generated if None.
    event_date:                 ISO date string for output filename.

    Returns
    -------
    Path to binary water-extent COG (1=flood, 0=dry), or None on failure.
    Tagged with source="SAR_contextual_RS", product_type="flood_extent".
    """
    if not _RASTERIO_AVAILABLE:
        log.warning("FloodDetector: rasterio not available — SAR module disabled")
        return None

    cfg = load_sar_settings().get("sar", {})

    # Spring melt artefact suppression
    if event_date and _is_spring_melt_period(event_date, cfg):
        log.warning(
            "FloodDetector: event date %s is in spring melt window (months %s). "
            "sar_confidence set to 'uncertain'.",
            event_date,
            cfg.get("artefact_suppression", {}).get("spring_melt_months", [3, 4]),
        )
        artefact_flag = True
    else:
        artefact_flag = False

    try:
        event_arr, event_profile = read_full_band(event_vv_path, band=1)
        ref_arr, ref_profile = read_full_band(reference_composite_path, band=1)
    except Exception as exc:
        log.error("FloodDetector: failed to read input rasters: %s", exc)
        return None

    # Apply Lee speckle filter
    event_arr = apply_lee_filter(event_arr.astype(np.float32))
    ref_arr = apply_lee_filter(ref_arr.astype(np.float32))

    # Align arrays (simple shape match — in production use rasterio.warp.reproject)
    if event_arr.shape != ref_arr.shape:
        log.warning(
            "FloodDetector: shape mismatch (%s vs %s) — using min common shape",
            event_arr.shape, ref_arr.shape
        )
        min_rows = min(event_arr.shape[0], ref_arr.shape[0])
        min_cols = min(event_arr.shape[1], ref_arr.shape[1])
        event_arr = event_arr[:min_rows, :min_cols]
        ref_arr = ref_arr[:min_rows, :min_cols]
        event_profile.update(height=min_rows, width=min_cols)

    # Convert to dB
    event_db = linear_to_db(event_arr)
    ref_db = linear_to_db(ref_arr)

    # Flood detection: event < reference + threshold_db
    flood_mask = (event_db < (ref_db + threshold_db)).astype(np.uint8)

    # Remove permanent water bodies
    if permanent_water_mask is not None and len(permanent_water_mask):
        flood_mask = _remove_permanent_water(
            flood_mask, permanent_water_mask, event_profile
        )

    # Determine output path
    if output_path is None:
        date_str = event_date or date.today().isoformat().replace("-", "")
        output_dir = Path("data/processed/sar_derived")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"water_extent_{date_str}.tif"

    tags = {
        "source":                  "SAR_contextual_RS",
        "product_type":            "flood_extent",
        "is_exact_asset_geometry": "false",
        "event_date":              event_date or "",
        "threshold_db":            str(threshold_db),
        "sar_confidence":          "uncertain" if artefact_flag else "unvalidated",
        "disclaimer":              _SAR_DISCLAIMER,
    }

    import rasterio.transform
    transform = event_profile.get("transform")
    crs_str = str(event_profile.get("crs", "EPSG:32617"))

    result_path = write_cog(
        flood_mask,
        output_path,
        crs=crs_str,
        transform=transform,
        dtype="uint8",
        nodata=255,
        tags=tags,
    )

    flood_pixels = int(flood_mask.sum())
    log.info(
        "FloodDetector: wrote flood mask to %s (%d flood pixels, confidence=%s)",
        result_path, flood_pixels, tags["sar_confidence"]
    )
    return result_path


def _remove_permanent_water(
    mask: np.ndarray,
    water_polygons,
    profile: dict,
) -> np.ndarray:
    """Zero out pixels corresponding to permanent water bodies."""
    try:
        from rasterio.features import rasterize as rs
        from shapely.geometry import mapping

        geoms = [(mapping(g), 1) for g in water_polygons.geometry if g and not g.is_empty]
        if not geoms:
            return mask

        perm_water = rs(
            geoms,
            out_shape=mask.shape,
            transform=profile.get("transform"),
            dtype=np.uint8,
            fill=0,
        )
        mask[perm_water == 1] = 0
    except Exception as exc:
        log.debug("Permanent water removal failed: %s", exc)
    return mask


def _is_spring_melt_period(event_date: str, cfg: dict) -> bool:
    """Check if the event date falls in the spring melt artefact window."""
    suppression = cfg.get("artefact_suppression", {})
    if not suppression.get("flag_spring_melt", True):
        return False
    melt_months = suppression.get("spring_melt_months", [3, 4])
    try:
        d = date.fromisoformat(event_date)
        return d.month in melt_months
    except ValueError:
        return False
