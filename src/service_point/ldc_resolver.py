"""LDC resolver — identify the licensed electricity distributor for any Ontario address."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
from shapely.geometry import Point

from ..utils.logger import get_logger

log = get_logger(__name__)


class LDCResolver:
    """Resolve the Local Distribution Company (LDC) serving any Ontario point.

    Uses OEB distributor service area boundaries as the authoritative source.
    Ontario has ~60 licensed distributors — never assume Hydro One.
    """

    def __init__(self, ldc_boundaries: gpd.GeoDataFrame) -> None:
        if ldc_boundaries is None or len(ldc_boundaries) == 0:
            raise ValueError("LDCResolver requires OEB LDC boundary GeoDataFrame")
        self._boundaries = ldc_boundaries.to_crs("EPSG:4326")
        # Build spatial index for fast point-in-polygon queries
        self._sindex = self._boundaries.sindex
        log.info("LDCResolver: initialized with %d LDC boundaries", len(self._boundaries))

    def resolve_point(self, lon: float, lat: float) -> Optional[dict]:
        """Return the LDC serving a point (lon, lat) in WGS84.

        Returns
        -------
        Dict with: ldc_name, ldc_id, source, confidence
        None if point is outside all known service areas.
        """
        pt = Point(lon, lat)
        candidates = list(self._sindex.query(pt))

        for idx in candidates:
            row = self._boundaries.iloc[idx]
            if row.geometry.contains(pt):
                return {
                    "ldc_name":   str(row.get("ldc_name", "Unknown")),
                    "ldc_id":     str(row.get("ldc_id", "")),
                    "source":     "OEB_authoritative",
                    "confidence": 1.0,
                }

        log.debug("LDCResolver: no LDC found for (%.4f, %.4f)", lat, lon)
        return None

    def resolve_gdf(self, points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add ldc_name and ldc_id columns to a point GeoDataFrame.

        Uses spatial join for batch efficiency.
        """
        if len(points_gdf) == 0:
            points_gdf = points_gdf.copy()
            points_gdf["ldc_name"] = ""
            points_gdf["ldc_id"] = ""
            return points_gdf

        # Ensure WGS84
        pts = points_gdf.to_crs("EPSG:4326") if points_gdf.crs else points_gdf.set_crs("EPSG:4326")

        joined = gpd.sjoin(
            pts,
            self._boundaries[["ldc_name", "ldc_id", "geometry"]],
            how="left",
            predicate="within",
        )

        # Handle multiple matches (e.g. overlapping boundaries) — take first
        joined = joined[~joined.index.duplicated(keep="first")]

        result = points_gdf.copy()
        result["ldc_name"] = joined["ldc_name"].values
        result["ldc_id"] = joined.get("ldc_id", "").values
        result["ldc_source"] = "OEB_authoritative"

        unresolved = result["ldc_name"].isna().sum()
        if unresolved:
            log.warning("LDCResolver: %d points outside known LDC boundaries", unresolved)

        return result

    def list_ldcs(self) -> list[str]:
        """Return sorted list of all LDC names in the boundary dataset."""
        return sorted(self._boundaries["ldc_name"].dropna().unique().tolist())
