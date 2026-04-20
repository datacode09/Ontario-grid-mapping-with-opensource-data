"""CRS normalization — ensure all layers are in EPSG:4326 for output."""
from __future__ import annotations

from typing import Union

import geopandas as gpd
import pandas as pd

from ..utils.geometry import reproject_gdf
from ..utils.logger import get_logger

log = get_logger(__name__)


class CRSNormalizer:
    """Normalize all spatial layers to a consistent output CRS (EPSG:4326)."""

    DEFAULT_OUTPUT_CRS = "EPSG:4326"

    def normalize(
        self,
        gdf: gpd.GeoDataFrame,
        target_crs: str = DEFAULT_OUTPUT_CRS,
        source_label: str = "unknown",
    ) -> gpd.GeoDataFrame:
        """Reproject gdf to target_crs if needed; assign WGS84 if CRS missing."""
        if gdf is None or len(gdf) == 0:
            return gdf

        if gdf.crs is None:
            log.warning("%s: no CRS detected — assuming EPSG:4326", source_label)
            gdf = gdf.set_crs("EPSG:4326")

        if gdf.crs.to_epsg() != _epsg_int(target_crs):
            log.debug(
                "%s: reprojecting %s → %s", source_label, gdf.crs.to_string(), target_crs
            )
            gdf = reproject_gdf(gdf, target_crs)

        return gdf

    def normalize_all(
        self,
        layers: dict[str, gpd.GeoDataFrame],
        target_crs: str = DEFAULT_OUTPUT_CRS,
    ) -> dict[str, gpd.GeoDataFrame]:
        """Normalize a dict of named GeoDataFrames."""
        return {
            name: self.normalize(gdf, target_crs, source_label=name)
            for name, gdf in layers.items()
        }

    def validate_geometry(self, gdf: gpd.GeoDataFrame, layer: str = "") -> gpd.GeoDataFrame:
        """Drop null or invalid geometries and log statistics."""
        original_len = len(gdf)
        gdf = gdf[~gdf.geometry.isna()].copy()
        gdf = gdf[gdf.geometry.is_valid].copy()
        dropped = original_len - len(gdf)
        if dropped:
            log.warning("%s: dropped %d invalid/null geometries", layer, dropped)
        return gdf


def _epsg_int(crs_str: str) -> int:
    """Extract integer EPSG code from strings like 'EPSG:4326'."""
    return int(crs_str.split(":")[-1])
