"""SAR change detector — bi-temporal log-ratio backscatter change screening.

Method:
  ratio = VV_t2_dB − VV_t1_dB
  |ratio| > threshold → change flagged

Thresholds (configurable in sar_settings.yaml):
  |ratio| > 3 dB  → moderate change
  |ratio| > 5 dB  → significant change
  |ratio| > 8 dB  → extreme change

LABELLING: Outputs tagged source="SAR_contextual_RS",
           product_type="backscatter_change"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np

from ..sar import _SAR_DISCLAIMER
from ..utils.config_loader import load_sar_settings
from ..utils.logger import get_logger
from ..utils.raster_utils import read_full_band, write_cog, apply_lee_filter, linear_to_db

log = get_logger(__name__)


def compute_backscatter_change(
    vv_before_path: Path,
    vv_after_path: Path,
    corridor_buffer: Optional[gpd.GeoDataFrame] = None,
    thresholds_db: Optional[list] = None,
    output_dir: Optional[Path] = None,
    before_date: Optional[str] = None,
    after_date: Optional[str] = None,
) -> dict:
    """Compute signed log-ratio of VV amplitude between two dates.

    Parameters
    ----------
    vv_before_path:   Path to VV GeoTIFF for reference (before) date.
    vv_after_path:    Path to VV GeoTIFF for event (after) date.
    corridor_buffer:  GeoDataFrame of corridor polygons to clip to.
    thresholds_db:    Change tier thresholds [moderate, significant, extreme].
    output_dir:       Directory for output COG files.
    before_date:      ISO date string for reference scene.
    after_date:       ISO date string for event scene.

    Returns
    -------
    Dict with:
      change_raster_path    (signed log-ratio GeoTIFF COG)
      change_mask_path      (tiered binary mask GeoTIFF COG)
      corridor_change_summary  (per corridor segment stats)
    """
    cfg = load_sar_settings().get("sar", {})
    thresholds_db = thresholds_db or [
        cfg.get("change_thresholds_db", {}).get("moderate", 3.0),
        cfg.get("change_thresholds_db", {}).get("significant", 5.0),
        cfg.get("change_thresholds_db", {}).get("extreme", 8.0),
    ]

    output_dir = output_dir or Path("data/processed/sar_derived")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and filter SAR rasters
    try:
        before_arr, before_profile = read_full_band(vv_before_path, band=1)
        after_arr, after_profile = read_full_band(vv_after_path, band=1)
    except Exception as exc:
        log.error("ChangeDetector: failed to load rasters: %s", exc)
        return _empty_change_result()

    # Align shapes
    rows = min(before_arr.shape[0], after_arr.shape[0])
    cols = min(before_arr.shape[1], after_arr.shape[1])
    before_arr = before_arr[:rows, :cols].astype(np.float32)
    after_arr  = after_arr[:rows, :cols].astype(np.float32)

    # Apply Lee speckle filter
    before_arr = apply_lee_filter(before_arr)
    after_arr  = apply_lee_filter(after_arr)

    # Compute signed log-ratio (dB scale)
    before_db = linear_to_db(before_arr)
    after_db  = linear_to_db(after_arr)
    ratio_db  = after_db - before_db

    # Three-tier threshold mask
    # 0=no change, 1=moderate, 2=significant, 3=extreme
    tier_mask = np.zeros_like(ratio_db, dtype=np.uint8)
    abs_ratio = np.abs(ratio_db)
    tier_mask[abs_ratio >= thresholds_db[0]] = 1  # moderate
    tier_mask[abs_ratio >= thresholds_db[1]] = 2  # significant
    tier_mask[abs_ratio >= thresholds_db[2]] = 3  # extreme

    # Write outputs as COG
    date_tag = f"{before_date or 'before'}_{after_date or 'after'}"
    transform = before_profile.get("transform")
    crs_str = str(before_profile.get("crs", "EPSG:32617"))

    change_raster_path = output_dir / f"change_ratio_{date_tag}.tif"
    change_mask_path   = output_dir / f"change_mask_{date_tag}.tif"

    ratio_tags = {
        "source":                  "SAR_contextual_RS",
        "product_type":            "backscatter_change_ratio",
        "is_exact_asset_geometry": "false",
        "before_date":             before_date or "",
        "after_date":              after_date or "",
        "disclaimer":              _SAR_DISCLAIMER,
    }
    mask_tags = {**ratio_tags, "product_type": "backscatter_change_mask"}

    write_cog(ratio_db, change_raster_path, crs=crs_str, transform=transform,
              dtype="float32", nodata=np.nan, tags=ratio_tags)
    write_cog(tier_mask, change_mask_path, crs=crs_str, transform=transform,
              dtype="uint8", nodata=255, tags=mask_tags)

    # Corridor change summary
    corridor_summary = []
    if corridor_buffer is not None and len(corridor_buffer):
        corridor_summary = _compute_corridor_summary(
            ratio_db, tier_mask, corridor_buffer, before_profile, thresholds_db,
            before_date, after_date
        )

    log.info(
        "ChangeDetector: change mask written to %s (%d moderate, %d significant, %d extreme pixels)",
        change_mask_path,
        int((tier_mask == 1).sum()),
        int((tier_mask == 2).sum()),
        int((tier_mask == 3).sum()),
    )

    return {
        "change_raster_path":      change_raster_path,
        "change_mask_path":        change_mask_path,
        "corridor_change_summary": corridor_summary,
    }


def _compute_corridor_summary(
    ratio_db, tier_mask, corridors, profile, thresholds, before_date, after_date
) -> list[dict]:
    """Compute per-corridor change statistics."""
    try:
        from rasterio.features import rasterize
        from shapely.geometry import mapping
        import rasterio

        records = []
        transform = profile.get("transform")
        shape = ratio_db.shape

        for idx, row in corridors.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Create corridor mask
            try:
                corridor_mask = rasterize(
                    [(mapping(geom), 1)],
                    out_shape=shape,
                    transform=transform,
                    dtype=np.uint8,
                    fill=0,
                )
            except Exception:
                continue

            valid = ratio_db[corridor_mask == 1]
            if len(valid) == 0:
                continue

            abs_vals = np.abs(valid)
            mean_ratio = float(np.mean(valid))
            max_abs = float(np.max(abs_vals))

            # Assign tier
            if max_abs >= thresholds[2]:
                tier = "extreme"
            elif max_abs >= thresholds[1]:
                tier = "significant"
            elif max_abs >= thresholds[0]:
                tier = "moderate"
            else:
                tier = "none"

            records.append({
                "corridor_id":           str(row.get("corridor_id", idx)),
                "voltage_kv":            float(row.get("voltage_kv", 0)),
                "mean_ratio_db":         round(mean_ratio, 3),
                "max_abs_ratio_db":      round(max_abs, 3),
                "change_tier":           tier,
                "flagged":               tier != "none",
                "before_date":           before_date or "",
                "after_date":            after_date or "",
                "source":                "SAR_contextual_RS",
                "is_exact_asset_geometry": False,
                "disclaimer":            _SAR_DISCLAIMER,
            })

        return records

    except ImportError:
        log.debug("rasterio unavailable for corridor summary")
        return []


def _tier_label(max_abs_db: float, thresholds: list) -> str:
    if max_abs_db >= thresholds[2]:
        return "extreme"
    if max_abs_db >= thresholds[1]:
        return "significant"
    if max_abs_db >= thresholds[0]:
        return "moderate"
    return "none"


def _empty_change_result() -> dict:
    return {
        "change_raster_path":      None,
        "change_mask_path":        None,
        "corridor_change_summary": [],
    }
