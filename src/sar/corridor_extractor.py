"""SAR corridor extractor — buffer transmission lines into ROW polygons.

Buffer widths by voltage:
  500 kV → 120 m
  230 kV →  60 m
  115 kV →  40 m
  distribution → 15 m

Clips SAR amplitude mosaic to each corridor buffer and computes
per-corridor backscatter statistics.

LABELLING: All outputs tagged with:
  source="SAR_contextual_RS"
  is_exact_asset_geometry=False
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ..sar import _SAR_DISCLAIMER
from ..utils.config_loader import load_sar_settings
from ..utils.geometry import to_lambert, to_4326
from ..utils.logger import get_logger
from ..utils.raster_utils import read_cog_window

log = get_logger(__name__)


def extract_corridor_footprints(
    transmission_lines: gpd.GeoDataFrame,
    buffer_widths_m: Optional[dict] = None,
    sar_mosaic_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Buffer transmission lines → ROW corridor polygons with optional SAR stats.

    Parameters
    ----------
    transmission_lines:
        GeoDataFrame of transmission line LineStrings with voltage_kv column.
    buffer_widths_m:
        Dict mapping voltage (kV as int/str) to buffer width (metres).
        Default: {500: 120, 230: 60, 115: 40, 0: 15}
    sar_mosaic_path:
        Optional path to SAR VV amplitude raster. If provided, computes
        mean VV backscatter statistics per corridor.

    Returns
    -------
    GeoDataFrame with mandatory metadata columns:
        corridor_id, voltage_kv, buffer_width_m, length_km,
        source, is_exact_asset_geometry, product_type, sar_scene_date,
        mean_vv_db, std_vv_db, p10_vv_db, p90_vv_db, disclaimer_text
        geometry (Polygon)
    """
    cfg = load_sar_settings().get("sar", {})
    if buffer_widths_m is None:
        raw = cfg.get("corridor_buffers_m", {})
        buffer_widths_m = {
            int(k) if str(k).isdigit() else 0: v
            for k, v in raw.items()
        }
        buffer_widths_m.setdefault(0, 15)

    lines_lam = to_lambert(transmission_lines)
    records = []

    for idx, row in lines_lam.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        voltage_kv = float(row.get("voltage_kv", 0) or 0)
        buf_width = _select_buffer_width(voltage_kv, buffer_widths_m)
        corridor_poly = geom.buffer(buf_width)

        rec = {
            "corridor_id":             f"corridor_{idx:06d}",
            "voltage_kv":              voltage_kv,
            "buffer_width_m":          buf_width,
            "length_km":               round(geom.length / 1000, 3),
            "operator":                str(row.get("operator", "")),
            "source":                  "SAR_contextual_RS",
            "is_exact_asset_geometry": False,
            "product_type":            "corridor_footprint",
            "sar_scene_date":          None,
            "mean_vv_db":              None,
            "std_vv_db":               None,
            "p10_vv_db":               None,
            "p90_vv_db":               None,
            "sar_confidence":          None,
            "disclaimer_text":         _SAR_DISCLAIMER,
            "geometry":                corridor_poly,
        }

        records.append(rec)

    if not records:
        return _empty_corridor_gdf()

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:3347")
    gdf = to_4326(gdf)

    # Attach SAR statistics if raster provided
    if sar_mosaic_path and Path(sar_mosaic_path).exists():
        gdf = _attach_sar_statistics(gdf, sar_mosaic_path)

    log.info("CorridorExtractor: extracted %d corridor polygons", len(gdf))
    return gdf


def _select_buffer_width(voltage_kv: float, buffer_widths_m: dict) -> float:
    """Select buffer width for a voltage level."""
    voltage_int = int(round(voltage_kv))
    # Match to nearest configured threshold
    if voltage_int >= 450:
        return buffer_widths_m.get(500, 120)
    elif voltage_int >= 200:
        return buffer_widths_m.get(230, 60)
    elif voltage_int >= 100:
        return buffer_widths_m.get(115, 40)
    elif voltage_int > 0:
        return buffer_widths_m.get(int(voltage_kv), buffer_widths_m.get(0, 15))
    return buffer_widths_m.get(0, 15)


def _attach_sar_statistics(
    corridors: gpd.GeoDataFrame, sar_path: Path
) -> gpd.GeoDataFrame:
    """Compute per-corridor VV backscatter statistics from a SAR raster."""
    try:
        import rasterio
        from rasterio.mask import mask as raster_mask
        from ..utils.raster_utils import linear_to_db, apply_lee_filter

        with rasterio.open(sar_path) as src:
            for idx, row in corridors.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                try:
                    from shapely.geometry import mapping
                    masked, _ = raster_mask(
                        src, [mapping(geom)], crop=True, nodata=np.nan
                    )
                    data = masked[0]
                    valid = data[np.isfinite(data) & (data > 0)]
                    if len(valid):
                        db_vals = linear_to_db(valid)
                        corridors.loc[idx, "mean_vv_db"] = round(np.mean(db_vals), 2)
                        corridors.loc[idx, "std_vv_db"] = round(np.std(db_vals), 2)
                        corridors.loc[idx, "p10_vv_db"] = round(np.percentile(db_vals, 10), 2)
                        corridors.loc[idx, "p90_vv_db"] = round(np.percentile(db_vals, 90), 2)
                        corridors.loc[idx, "sar_confidence"] = "unvalidated"
                except Exception as exc:
                    log.debug("SAR stats failed for corridor %s: %s", idx, exc)

    except ImportError:
        log.warning("rasterio unavailable — skipping SAR corridor statistics")

    return corridors


def _empty_corridor_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        columns=[
            "corridor_id", "voltage_kv", "buffer_width_m", "length_km",
            "source", "is_exact_asset_geometry", "product_type",
            "sar_scene_date", "mean_vv_db", "disclaimer_text", "geometry"
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
