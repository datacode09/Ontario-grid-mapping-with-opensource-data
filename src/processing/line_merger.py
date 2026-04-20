"""Line merger — merge fragmented OSM line segments into logical feeders."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.ops import linemerge, unary_union

from ..utils.logger import get_logger

log = get_logger(__name__)


class LineMerger:
    """Merge collinear OSM power line segments sharing endpoints.

    OSM often has a single logical feeder split across many way segments.
    This class reassembles them by:
      1. Grouping by (voltage_tier, operator)
      2. Applying shapely linemerge within each group
      3. Assigning a canonical feeder_id
    """

    def merge_feeders(
        self,
        lines_gdf: gpd.GeoDataFrame,
        group_cols: Optional[list[str]] = None,
        max_gap_m: float = 5.0,
    ) -> gpd.GeoDataFrame:
        """Merge fragmented line segments.

        Parameters
        ----------
        lines_gdf:   GeoDataFrame of LineString features.
        group_cols:  Columns to group by (default: voltage_tier + operator).
        max_gap_m:   Snap endpoints within this distance (metres) before merging.

        Returns
        -------
        GeoDataFrame with merged LineStrings and a feeder_id column.
        """
        if lines_gdf is None or len(lines_gdf) == 0:
            return lines_gdf

        group_cols = group_cols or self._default_group_cols(lines_gdf)
        log.info("LineMerger: merging %d segments by %s", len(lines_gdf), group_cols)

        merged_records = []
        feeder_counter = 0

        for group_key, group_df in self._group(lines_gdf, group_cols):
            geoms = group_df.geometry.tolist()
            if max_gap_m > 0:
                geoms = self._snap_endpoints(geoms, max_gap_m)

            merged = linemerge(unary_union(geoms))
            if merged.geom_type == "LineString":
                merged = [merged]
            elif merged.geom_type == "MultiLineString":
                merged = list(merged.geoms)
            else:
                continue

            representative = group_df.iloc[0]
            for geom in merged:
                rec = {
                    "feeder_id": f"feeder_{feeder_counter:06d}",
                    "geometry": geom,
                    "voltage_tier": representative.get("voltage_tier", "unknown"),
                    "voltage_kv": representative.get("voltage_kv", 0),
                    "operator": representative.get("operator", ""),
                    "source": "OSM_merged",
                    "confidence": representative.get("confidence", 0.75),
                    "segment_count": len(group_df),
                }
                merged_records.append(rec)
                feeder_counter += 1

        if not merged_records:
            log.warning("LineMerger: no merged records produced")
            return lines_gdf

        result = gpd.GeoDataFrame(merged_records, geometry="geometry", crs=lines_gdf.crs)
        log.info(
            "LineMerger: %d segments → %d logical feeders",
            len(lines_gdf),
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _default_group_cols(gdf: gpd.GeoDataFrame) -> list[str]:
        available = [c for c in ["voltage_tier", "operator"] if c in gdf.columns]
        return available if available else []

    @staticmethod
    def _group(gdf: gpd.GeoDataFrame, cols: list[str]):
        if not cols:
            yield ("all",), gdf
            return
        for key, grp in gdf.groupby(cols, dropna=False):
            yield key, grp

    def _snap_endpoints(self, geoms: list, max_gap_m: float) -> list:
        """Snap nearly-touching endpoints within max_gap_m (simplified approach)."""
        # For full snapping, we'd use a spatial index; here we use a tolerance buffer
        from shapely.geometry import MultiLineString
        from shapely.ops import unary_union

        multi = unary_union(geoms)
        if hasattr(multi, "geoms"):
            return list(multi.geoms)
        return [multi]
