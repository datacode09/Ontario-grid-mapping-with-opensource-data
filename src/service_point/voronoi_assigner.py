"""Voronoi assigner — assign buildings to nearest transformer via Voronoi regions."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Point
from shapely.ops import unary_union

from ..utils.geometry import to_lambert, to_4326, voronoi_gdf
from ..utils.logger import get_logger

log = get_logger(__name__)


class VoronoiAssigner:
    """Assign each building to a serving secondary transformer using Voronoi cells.

    Voronoi regions are computed in EPSG:3347 (Statistics Canada Lambert)
    for metrically accurate boundaries, then reprojected to WGS84 for output.

    Confidence: 0.20 (Voronoi proxy only, no transformer located) when no
    transformer is within the TransformerFinder radius.
    """

    def __init__(self, transformers_gdf: gpd.GeoDataFrame) -> None:
        if len(transformers_gdf) == 0:
            raise ValueError("VoronoiAssigner requires at least one transformer")
        self._transformers = transformers_gdf

    def assign(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        clip_boundary: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """Assign each building to its Voronoi-nearest transformer.

        Parameters
        ----------
        buildings_gdf:   Building centroid GeoDataFrame.
        clip_boundary:   Optional polygon to clip Voronoi regions.

        Returns
        -------
        GeoDataFrame with added columns: assigned_tx_id, assignment_method,
          assignment_confidence.
        """
        log.info(
            "VoronoiAssigner: assigning %d buildings to %d transformers",
            len(buildings_gdf),
            len(self._transformers),
        )

        # Project to Lambert for geometric accuracy
        tx_lambert = to_lambert(self._transformers)
        tx_lambert["geometry"] = tx_lambert.geometry.centroid

        bldg_lambert = to_lambert(buildings_gdf)
        bldg_lambert["geometry"] = bldg_lambert.geometry.centroid

        # Build Voronoi regions
        clip_poly = None
        if clip_boundary is not None and len(clip_boundary):
            clip_poly = to_lambert(clip_boundary).geometry.unary_union

        voronoi_regions = voronoi_gdf(tx_lambert, clip_to=clip_poly)

        # Spatial join buildings → Voronoi regions
        joined = gpd.sjoin(
            bldg_lambert,
            voronoi_regions[["tx_id", "voronoi_poly"]].rename(
                columns={"voronoi_poly": "geometry"}
            ).set_geometry("geometry"),
            how="left",
            predicate="within",
        )

        # Handle unmatched buildings (outside all Voronoi cells — use nearest)
        result = buildings_gdf.copy()
        matched_idx = ~joined["tx_id"].isna()
        result.loc[matched_idx.values, "assigned_tx_id"] = joined.loc[matched_idx, "tx_id"].values
        result.loc[~matched_idx.values, "assigned_tx_id"] = self._fallback_nearest(
            bldg_lambert[~matched_idx.values], tx_lambert
        )

        result["assignment_method"] = result["assigned_tx_id"].apply(
            lambda x: "voronoi" if pd.notna(x) else "unassigned"
        )
        result["assignment_confidence"] = result["assigned_tx_id"].apply(
            lambda x: 0.20 if pd.isna(x) else 0.45
        )

        assigned = result["assigned_tx_id"].notna().sum()
        log.info("VoronoiAssigner: assigned %d/%d buildings", assigned, len(result))
        return result

    def _fallback_nearest(
        self, unmatched_bldg: gpd.GeoDataFrame, tx_lambert: gpd.GeoDataFrame
    ) -> list:
        """Assign unmatched buildings to the nearest transformer by centroid distance."""
        assignments = []
        for _, bldg_row in unmatched_bldg.iterrows():
            bldg_pt = bldg_row.geometry
            distances = tx_lambert.geometry.distance(bldg_pt)
            nearest_idx = distances.idxmin()
            tx_row = tx_lambert.loc[nearest_idx]
            assignments.append(str(tx_row.get("tx_id", nearest_idx)))
        return assignments
