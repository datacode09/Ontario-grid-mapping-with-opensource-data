"""SAR resilience-risk exposure scorer — composite index per grid asset.

Inputs (per substation):
  - SAR water extent mask (flood_detector)
  - SAR change mask summary (change_detector)
  - Substation elevation (Copernicus DEM, Source 25)
  - Official flood zones (ECCC, Source 26)
  - Watershed catchment (OIHN, Source 27)
  - SAIDI/SAIFI history (OEB RRR, Source 2)

Output per substation:
  asset_id, asset_type, sar_flood_exposure, sar_change_tier,
  elevation_m, in_official_floodzone, saidi_percentile,
  resilience_risk_score (0–1), risk_label (LOW/MEDIUM/HIGH/CRITICAL)

CRITICAL REQUIREMENT: sar_confidence is NEVER merged with grid topology
confidence. They remain independent attributes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ..sar import _SAR_DISCLAIMER
from ..utils.config_loader import load_sar_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_WEIGHTS = {
    "sar_flood_exposure":  0.30,
    "official_flood_zone": 0.25,
    "elevation_inverse":   0.15,
    "sar_change_tier":     0.15,
    "saidi_percentile":    0.15,
}

_CHANGE_TIER_VALUES = {
    "none": 0.0,
    "moderate": 0.33,
    "significant": 0.67,
    "extreme": 1.0,
}


def score_substation_resilience(
    substations: gpd.GeoDataFrame,
    flood_mask_path: Optional[Path],
    change_summary: list[dict],
    dem_path: Optional[Path],
    eccc_flood_zones: Optional[gpd.GeoDataFrame],
    oeb_saidi: Optional[pd.DataFrame],
    weights: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """Compute composite resilience-risk score for each substation.

    Parameters
    ----------
    substations:       GeoDataFrame of transmission/zone substations.
    flood_mask_path:   Path to SAR water-extent binary COG.
    change_summary:    Per-corridor change summary list from change_detector.
    dem_path:          Path to DEM GeoTIFF for elevation extraction.
    eccc_flood_zones:  ECCC official flood zone polygons.
    oeb_saidi:         OEB RRR reliability DataFrame with saidi_minutes column.
    weights:           Custom risk score weights (sum to 1.0).

    Returns
    -------
    GeoDataFrame with resilience risk scoring columns added.
    All SAR-derived columns are independent of topology confidence.
    """
    cfg = load_sar_settings().get("sar", {})
    weights = weights or cfg.get("resilience_score_weights", _DEFAULT_WEIGHTS)
    result = substations.copy()

    # --- Component 1: SAR flood exposure ---
    result["sar_flood_exposure"] = 0
    if flood_mask_path and Path(flood_mask_path).exists():
        result = _add_flood_exposure(result, flood_mask_path)

    # --- Component 2: Official ECCC flood zone overlap ---
    result["in_official_floodzone"] = False
    if eccc_flood_zones is not None and len(eccc_flood_zones):
        result = _add_eccc_overlap(result, eccc_flood_zones)

    # --- Component 3: Elevation (inverse — low elevation = higher risk) ---
    result["elevation_m"] = np.nan
    if dem_path and Path(dem_path).exists():
        result = _add_elevation(result, dem_path)

    # --- Component 4: SAR change tier from corridor summary ---
    result["sar_change_tier"] = "none"
    if change_summary:
        result = _add_change_tier(result, change_summary)

    # --- Component 5: SAIDI percentile (reliability history) ---
    result["saidi_percentile"] = 0.50  # Default to median if no data
    if oeb_saidi is not None and len(oeb_saidi):
        result = _add_saidi_percentile(result, oeb_saidi)

    # --- Composite score ---
    result = _compute_composite_score(result, weights)

    # --- SAR confidence (independent of topology confidence) ---
    result["sar_confidence"] = result["sar_flood_exposure"].apply(
        lambda x: "unvalidated" if x else None
    )

    # Mark all SAR fields
    result["source"] = "SAR_contextual_RS"
    result["is_exact_asset_geometry"] = False
    result["disclaimer_text"] = _SAR_DISCLAIMER

    log.info(
        "ExposureScorer: scored %d substations "
        "(CRITICAL=%d, HIGH=%d, MEDIUM=%d, LOW=%d)",
        len(result),
        (result["risk_label"] == "CRITICAL").sum(),
        (result["risk_label"] == "HIGH").sum(),
        (result["risk_label"] == "MEDIUM").sum(),
        (result["risk_label"] == "LOW").sum(),
    )
    return result


# ------------------------------------------------------------------
# Component calculators
# ------------------------------------------------------------------

def _add_flood_exposure(gdf: gpd.GeoDataFrame, flood_path: Path) -> gpd.GeoDataFrame:
    """Add sar_flood_exposure (0 or 1) from SAR water extent raster."""
    try:
        import rasterio
        from rasterio.sample import sample_gen

        with rasterio.open(flood_path) as src:
            coords = [(row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
            values = list(src.sample(coords, indexes=[1]))
            gdf = gdf.copy()
            gdf["sar_flood_exposure"] = [int(v[0]) if v and len(v) else 0 for v in values]
    except Exception as exc:
        log.debug("Flood exposure extraction failed: %s", exc)
    return gdf


def _add_eccc_overlap(
    gdf: gpd.GeoDataFrame, flood_zones: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Set in_official_floodzone=True for substations within ECCC polygons."""
    try:
        zones_4326 = flood_zones.to_crs("EPSG:4326")
        joined = gpd.sjoin(
            gdf[["geometry"]].to_crs("EPSG:4326"),
            zones_4326[["geometry"]],
            how="left",
            predicate="within",
        )
        gdf = gdf.copy()
        gdf["in_official_floodzone"] = ~joined["index_right"].isna()
    except Exception as exc:
        log.debug("ECCC flood zone join failed: %s", exc)
    return gdf


def _add_elevation(gdf: gpd.GeoDataFrame, dem_path: Path) -> gpd.GeoDataFrame:
    """Extract elevation from DEM for each substation."""
    try:
        import rasterio
        from ..utils.raster_utils import reproject_raster

        with rasterio.open(dem_path) as src:
            coords = [(row.geometry.x, row.geometry.y) for _, row in gdf.iterrows()]
            values = list(src.sample(coords, indexes=[1]))
            gdf = gdf.copy()
            gdf["elevation_m"] = [
                float(v[0]) if v and len(v) and np.isfinite(v[0]) else np.nan
                for v in values
            ]
    except Exception as exc:
        log.debug("Elevation extraction failed: %s", exc)
    return gdf


def _add_change_tier(gdf: gpd.GeoDataFrame, corridor_summary: list[dict]) -> gpd.GeoDataFrame:
    """Assign worst change tier from corridors feeding each substation."""
    # Map corridor_id → change_tier
    tier_map = {s["corridor_id"]: s["change_tier"] for s in corridor_summary}
    if not tier_map:
        return gdf

    # Very simplified: assign the worst tier found in any corridor
    worst_tier = max(tier_map.values(), key=lambda t: _CHANGE_TIER_VALUES.get(t, 0))
    gdf = gdf.copy()
    gdf["sar_change_tier"] = worst_tier
    return gdf


def _add_saidi_percentile(
    gdf: gpd.GeoDataFrame, saidi_df: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Assign SAIDI percentile for each substation's LDC."""
    if "saidi_minutes" not in saidi_df.columns or "ldc_name" not in saidi_df.columns:
        return gdf

    saidi_series = saidi_df.set_index("ldc_name")["saidi_minutes"].dropna()
    if len(saidi_series) == 0:
        return gdf

    max_saidi = saidi_series.max()
    gdf = gdf.copy()

    def _percentile(row):
        ldc = row.get("operator", row.get("ldc_name", ""))
        if ldc in saidi_series.index:
            return float(saidi_series[ldc] / max_saidi) if max_saidi else 0.5
        return 0.5

    gdf["saidi_percentile"] = gdf.apply(_percentile, axis=1)
    return gdf


def _compute_composite_score(
    gdf: gpd.GeoDataFrame, weights: dict
) -> gpd.GeoDataFrame:
    """Compute composite resilience risk score [0, 1]."""
    gdf = gdf.copy()

    # Normalise elevation inverse
    elev = pd.to_numeric(gdf["elevation_m"], errors="coerce").fillna(50)
    max_elev = elev.max() or 1.0
    elev_inv = 1.0 - (elev / max_elev).clip(0, 1)

    tier_numeric = gdf["sar_change_tier"].map(_CHANGE_TIER_VALUES).fillna(0)

    w = weights
    score = (
        float(w.get("sar_flood_exposure", 0.30))  * gdf["sar_flood_exposure"].fillna(0).astype(float) +
        float(w.get("official_flood_zone", 0.25)) * gdf["in_official_floodzone"].fillna(False).astype(float) +
        float(w.get("elevation_inverse", 0.15))   * elev_inv +
        float(w.get("sar_change_tier", 0.15))     * tier_numeric +
        float(w.get("saidi_percentile", 0.15))    * gdf["saidi_percentile"].fillna(0.5)
    )

    gdf["resilience_risk_score"] = score.clip(0, 1).round(3)
    gdf["risk_label"] = score.apply(_risk_label)
    return gdf


def _risk_label(score: float) -> str:
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.25:
        return "MEDIUM"
    return "LOW"
