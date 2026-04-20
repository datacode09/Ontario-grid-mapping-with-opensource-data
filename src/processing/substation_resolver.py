"""Substation resolver — deduplicate substations across OSM, IESO, NRCan."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..utils.geometry import to_lambert
from ..utils.logger import get_logger

log = get_logger(__name__)

# Minimum separation (metres) to treat two records as distinct substations
_DEDUP_RADIUS_M = 200


class SubstationResolver:
    """Resolve duplicate substation records across multiple open data sources.

    Merges records within a spatial tolerance, preferring higher-confidence
    sources (IESO > OSM confirmed > OSM inferred > NRCan).
    """

    def resolve(
        self,
        osm_gdf: Optional[gpd.GeoDataFrame] = None,
        ieso_gdf: Optional[gpd.GeoDataFrame] = None,
        nrcan_gdf: Optional[gpd.GeoDataFrame] = None,
        dedup_radius_m: float = _DEDUP_RADIUS_M,
    ) -> gpd.GeoDataFrame:
        """Merge and deduplicate substation point layers.

        Priority: IESO → OSM confirmed → OSM inferred → NRCan
        """
        layers = []
        if ieso_gdf is not None and len(ieso_gdf):
            layers.append(ieso_gdf.copy())
        if osm_gdf is not None and len(osm_gdf):
            layers.append(osm_gdf.copy())
        if nrcan_gdf is not None and len(nrcan_gdf):
            layers.append(nrcan_gdf.copy())

        if not layers:
            log.warning("SubstationResolver: no input layers")
            return self._empty_gdf()

        import pandas as pd
        combined = gpd.GeoDataFrame(
            pd.concat(layers, ignore_index=True), crs="EPSG:4326"
        )
        combined = self._ensure_point_geometry(combined)
        combined = self._add_priority(combined)
        deduplicated = self._spatial_dedup(combined, dedup_radius_m)

        log.info(
            "SubstationResolver: %d records → %d unique substations",
            len(combined),
            len(deduplicated),
        )
        return deduplicated

    def _ensure_point_geometry(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Convert polygon substations to centroid points."""
        mask = gdf.geometry.geom_type != "Point"
        if mask.any():
            gdf = gdf.copy()
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].centroid
        return gdf

    def _add_priority(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assign numeric priority for dedup conflict resolution."""
        source_priority = {
            "IESO_TX_Registry": 1,
            "IESO_Generator_Registry": 1,
            "OSM": 2,
            "NRCan_CanVec": 3,
        }
        gdf = gdf.copy()
        gdf["_priority"] = gdf.get("source", pd.Series(dtype=object)).map(
            lambda s: next(
                (v for k, v in source_priority.items() if k in str(s)), 4
            )
        )
        return gdf.sort_values("_priority")

    def _spatial_dedup(
        self, gdf: gpd.GeoDataFrame, radius_m: float
    ) -> gpd.GeoDataFrame:
        """Remove duplicate substations within radius_m using spatial index."""
        lambert = to_lambert(gdf)
        kept_indices = []
        used = set()
        sindex = lambert.sindex

        for idx, row in lambert.iterrows():
            if idx in used:
                continue
            candidates = list(sindex.query(row.geometry.buffer(radius_m)))
            for cand_idx in candidates:
                real_idx = lambert.index[cand_idx]
                if real_idx != idx:
                    used.add(real_idx)
            kept_indices.append(idx)

        result = gdf.loc[kept_indices].copy()
        result = result.drop(columns=["_priority"], errors="ignore")
        return result

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["facility_name", "voltage_kv", "operator", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
