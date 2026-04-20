"""Transformer finder — find the probable serving secondary transformer for an address."""
from __future__ import annotations

from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.neighbors import BallTree

from ..utils.geometry import to_lambert
from ..utils.logger import get_logger

log = get_logger(__name__)

_MAX_SEARCH_RADIUS_M = 200  # Default maximum search radius


class TransformerFinder:
    """Find the nearest secondary transformer to any service point.

    Uses a BallTree spatial index for efficient KNN queries.
    Returns confidence based on data source (OSM > ML inferred > Voronoi).
    """

    def __init__(
        self,
        transformers_gdf: gpd.GeoDataFrame,
        max_radius_m: float = _MAX_SEARCH_RADIUS_M,
    ) -> None:
        self._transformers = transformers_gdf.to_crs("EPSG:4326")
        self._max_radius_m = max_radius_m
        self._ball_tree = None
        self._build_index()

    def _build_index(self) -> None:
        """Build a BallTree index on transformer locations."""
        if len(self._transformers) == 0:
            log.warning("TransformerFinder: no transformers to index")
            return

        centroids = to_lambert(self._transformers)
        coords = np.array([(g.x, g.y) for g in centroids.geometry if g])
        if coords.shape[0] > 0:
            self._ball_tree = BallTree(np.radians(
                np.column_stack([
                    self._transformers.geometry.y.values,
                    self._transformers.geometry.x.values,
                ])
            ))
        log.info("TransformerFinder: indexed %d transformers", len(self._transformers))

    def find_nearest(
        self,
        point: Point,
        radius_m: Optional[float] = None,
    ) -> Optional[dict]:
        """Find the nearest secondary transformer to a WGS84 Point.

        Parameters
        ----------
        point:    WGS84 Point (lon, lat).
        radius_m: Search radius override (metres).

        Returns
        -------
        Dict with: tx_id, distance_m, source, confidence, geometry
        or None if no transformer found within radius.
        """
        if self._ball_tree is None:
            return None

        radius_m = radius_m or self._max_radius_m
        radius_rad = radius_m / 6_371_000  # Earth radius in metres

        query_pt = np.radians([[point.y, point.x]])
        indices, distances = self._ball_tree.query_radius(
            query_pt, r=radius_rad, return_distance=True
        )

        if not len(indices[0]):
            return None

        best_idx = indices[0][np.argmin(distances[0])]
        best_dist_rad = distances[0][np.argmin(distances[0])]
        best_dist_m = best_dist_rad * 6_371_000

        tx_row = self._transformers.iloc[best_idx]
        return {
            "tx_id":       str(tx_row.get("tx_id", f"tx_{best_idx}")),
            "distance_m":  round(best_dist_m, 1),
            "source":      str(tx_row.get("source", "")),
            "confidence":  float(tx_row.get("confidence", 0.45)),
            "geometry":    tx_row.geometry,
            "operator":    str(tx_row.get("operator", "")),
        }

    def find_nearest_batch(
        self,
        points_gdf: gpd.GeoDataFrame,
        radius_m: Optional[float] = None,
    ) -> gpd.GeoDataFrame:
        """Find nearest transformer for each point in a GeoDataFrame.

        Adds columns: tx_id, distance_m, tx_source, tx_confidence.
        """
        if self._ball_tree is None:
            points_gdf = points_gdf.copy()
            points_gdf["tx_id"] = None
            points_gdf["distance_m"] = None
            return points_gdf

        radius_m = radius_m or self._max_radius_m
        radius_rad = radius_m / 6_371_000

        pts_4326 = points_gdf.to_crs("EPSG:4326")
        query = np.radians(
            np.column_stack([pts_4326.geometry.y.values, pts_4326.geometry.x.values])
        )
        indices, distances = self._ball_tree.query_radius(
            query, r=radius_rad, return_distance=True
        )

        result = points_gdf.copy()
        tx_ids, dists, sources, confs = [], [], [], []

        for idx_arr, dist_arr in zip(indices, distances):
            if len(idx_arr) == 0:
                tx_ids.append(None)
                dists.append(None)
                sources.append(None)
                confs.append(0.20)  # Voronoi fallback
            else:
                best = idx_arr[np.argmin(dist_arr)]
                tx_row = self._transformers.iloc[best]
                tx_ids.append(str(tx_row.get("tx_id", f"tx_{best}")))
                dists.append(round(dist_arr.min() * 6_371_000, 1))
                sources.append(str(tx_row.get("source", "")))
                confs.append(float(tx_row.get("confidence", 0.45)))

        result["tx_id"] = tx_ids
        result["distance_m"] = dists
        result["tx_source"] = sources
        result["tx_confidence"] = confs
        return result
