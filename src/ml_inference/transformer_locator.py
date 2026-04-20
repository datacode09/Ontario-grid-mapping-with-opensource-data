"""Secondary transformer locator — infer pad-mount/pole-top transformer locations.

Places secondary transformers at regular intervals along reconstructed feeders,
informed by building density from Microsoft footprints.
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from ..utils.geometry import to_lambert
from ..utils.logger import get_logger

log = get_logger(__name__)

# Ontario typical service radiues: one transformer per ~8–12 homes
_RESIDENTIAL_HOMES_PER_TX = 10
_MAX_SERVICE_RADIUS_M = 75  # metres — secondary transformer to home


class TransformerLocator:
    """Locate secondary distribution transformers along feeder lines.

    Method:
      1. Cluster buildings into groups of ~N homes (configurable)
      2. Place a transformer at each cluster centroid
      3. Snap transformer to nearest feeder line point
    """

    def __init__(self, homes_per_tx: int = _RESIDENTIAL_HOMES_PER_TX) -> None:
        self._homes_per_tx = homes_per_tx

    def locate(
        self,
        feeders_gdf: gpd.GeoDataFrame,
        buildings_gdf: gpd.GeoDataFrame,
        max_service_radius_m: float = _MAX_SERVICE_RADIUS_M,
    ) -> gpd.GeoDataFrame:
        """Infer secondary transformer locations.

        Parameters
        ----------
        feeders_gdf:       Primary feeder LineStrings.
        buildings_gdf:     Building footprint centroids.
        max_service_radius_m: Maximum distance from building to transformer.

        Returns
        -------
        GeoDataFrame of transformer point locations.
        """
        if len(buildings_gdf) == 0 or len(feeders_gdf) == 0:
            return self._empty_tx_gdf()

        # Project to Lambert for distance operations
        buildings_lam = to_lambert(buildings_gdf.copy())
        buildings_lam["geometry"] = buildings_lam.geometry.centroid

        feeders_lam = to_lambert(feeders_gdf)
        feeder_union = feeders_lam.geometry.unary_union

        # Cluster buildings using K-Means
        coords = np.array([(g.x, g.y) for g in buildings_lam.geometry if g])
        n_clusters = max(1, len(buildings_lam) // self._homes_per_tx)

        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            labels = kmeans.fit_predict(coords)
            cluster_centers = kmeans.cluster_centers_
        except ImportError:
            log.warning("scikit-learn unavailable — using grid-based transformer placement")
            return self._grid_place_transformers(feeders_lam, buildings_lam)

        records = []
        for i, center in enumerate(cluster_centers):
            pt = Point(center[0], center[1])
            # Snap to nearest point on feeder
            if feeder_union and not feeder_union.is_empty:
                snapped = feeder_union.interpolate(feeder_union.project(pt))
                dist = pt.distance(snapped)
                tx_pt = snapped if dist <= max_service_radius_m * 3 else pt
            else:
                tx_pt = pt

            n_buildings = int((labels == i).sum())
            records.append({
                "tx_id":          f"inferred_tx_{i:06d}",
                "n_buildings_served": n_buildings,
                "confidence":     0.45,  # ML suburban confidence
                "source":         "GridMapping_ML_transformer",
                "node_type":      "secondary_transformer",
                "inferred":       True,
                "geometry":       tx_pt,
            })

        if not records:
            return self._empty_tx_gdf()

        # Reproject to WGS84
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:3347")
        gdf = gdf.to_crs("EPSG:4326")
        log.info(
            "TransformerLocator: placed %d inferred transformers for %d buildings",
            len(gdf), len(buildings_gdf)
        )
        return gdf

    def _grid_place_transformers(
        self, feeders: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Fallback: place transformers at regular intervals along feeders."""
        spacing_m = 100.0
        records = []
        tx_id = 0

        for _, row in feeders.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            n_pts = max(1, int(geom.length / spacing_m))
            for i in range(n_pts):
                pt = geom.interpolate(i * spacing_m)
                records.append({
                    "tx_id": f"grid_tx_{tx_id:06d}",
                    "n_buildings_served": self._homes_per_tx,
                    "confidence": 0.30,
                    "source": "GridMapping_ML_transformer_fallback",
                    "node_type": "secondary_transformer",
                    "inferred": True,
                    "geometry": pt,
                })
                tx_id += 1

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:3347")
        return gdf.to_crs("EPSG:4326")

    def _empty_tx_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["tx_id", "n_buildings_served", "confidence", "source",
                     "node_type", "inferred", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
