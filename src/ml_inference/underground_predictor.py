"""Underground cable predictor — identify underground distribution zones.

Erin Mills (post-1975 Mississauga) and similar post-war planned developments
have extensive underground residential distribution. This module predicts
underground zones using land-use classification and OSM tags.
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd

from ..utils.geometry import buffer_degrees, to_lambert
from ..utils.logger import get_logger

log = get_logger(__name__)

# Ontario municipalities with known significant underground distribution
_KNOWN_UNDERGROUND_AREAS = [
    # (name, centroid_lon, centroid_lat, radius_m, year_built_after)
    ("Erin Mills", -79.77, 43.55, 5000, 1975),
    ("Kanata", -75.91, 45.35, 3000, 1970),
    ("Mississauga City Centre", -79.64, 43.59, 2000, 1980),
    ("North York suburban", -79.42, 43.77, 4000, 1970),
    ("Brampton Springdale", -79.79, 43.74, 3000, 1990),
]


class UndergroundPredictor:
    """Predict underground cable distribution zones.

    Uses a rule-based approach:
      1. Known underground planned communities (hardcoded)
      2. OSM power=cable features
      3. Land-use classification (residential + year > threshold)
    """

    def predict_underground_zones(
        self,
        area_of_interest: gpd.GeoDataFrame,
        osm_cables: Optional[gpd.GeoDataFrame] = None,
        land_use: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """Return polygons flagged as likely underground distribution.

        Parameters
        ----------
        area_of_interest:   Boundary polygon of the study area.
        osm_cables:         OSM power=cable features (explicit underground).
        land_use:           Land-use GeoDataFrame with 'land_use' column.

        Returns
        -------
        GeoDataFrame of underground zone polygons with confidence scores.
        """
        records = []

        # 1. Known planned underground communities
        for name, lon, lat, radius_m, year in _KNOWN_UNDERGROUND_AREAS:
            from shapely.geometry import Point
            from pyproj import Transformer
            # Check if this area overlaps the study AOI
            pt = Point(lon, lat)
            aoi_union = area_of_interest.geometry.unary_union
            if aoi_union.distance(pt) < 0.1:  # within ~10 km
                # Build buffer polygon
                lambert_t = Transformer.from_crs("EPSG:4326", "EPSG:3347", always_xy=True)
                x_lam, y_lam = lambert_t.transform(lon, lat)
                from shapely.geometry import Point as ShapelyPoint
                pt_lam = ShapelyPoint(x_lam, y_lam)
                buf_lam = pt_lam.buffer(radius_m)

                wgs84_t = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
                buf_wgs84 = self._transform_polygon(buf_lam, wgs84_t)

                records.append({
                    "zone_name":        name,
                    "zone_type":        "underground",
                    "evidence":         "known_planned_community",
                    "year_built_after": year,
                    "confidence":       0.80,
                    "geometry":         buf_wgs84,
                    "source":           "UndergroundPredictor_hardcoded",
                })

        # 2. OSM explicit cable features — build convex hull polygons
        if osm_cables is not None and len(osm_cables):
            cable_union = osm_cables.geometry.unary_union
            if not cable_union.is_empty:
                hull = cable_union.convex_hull.buffer(0.002)
                records.append({
                    "zone_name":        "OSM_cable_zone",
                    "zone_type":        "underground",
                    "evidence":         "osm_power_cable",
                    "year_built_after": None,
                    "confidence":       0.90,
                    "geometry":         hull,
                    "source":           "OSM",
                })

        if not records:
            return self._empty_gdf()

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        log.info("UndergroundPredictor: identified %d underground zones", len(gdf))
        return gdf

    def is_underground(self, point, underground_zones: gpd.GeoDataFrame) -> bool:
        """Return True if a point falls within a known underground zone."""
        if len(underground_zones) == 0:
            return False
        from shapely.geometry import Point
        pt = Point(point) if not isinstance(point, Point) else point
        return any(zone.contains(pt) for zone in underground_zones.geometry)

    @staticmethod
    def _transform_polygon(geom, transformer):
        """Transform a Shapely polygon using a pyproj Transformer."""
        from shapely.geometry import Polygon
        coords = list(geom.exterior.coords)
        transformed = [transformer.transform(x, y) for x, y in coords]
        return Polygon(transformed)

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["zone_name", "zone_type", "evidence", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
