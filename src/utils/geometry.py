"""Geometry helper utilities — CRS conversions, bounding boxes, spatial ops."""
from __future__ import annotations

from typing import Tuple

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import MultiLineString, Point, Polygon, box

# Standard CRS constants
WGS84 = "EPSG:4326"
LAMBERT = "EPSG:3347"   # Statistics Canada Lambert — area/distance
UTM17N = "EPSG:32617"   # UTM Zone 17N — SAR / Sentinel-1 over southern Ontario


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def bbox_to_polygon(
    west: float, south: float, east: float, north: float, crs: str = WGS84
) -> Polygon:
    """Return a Shapely Polygon from bbox coordinates."""
    return box(west, south, east, north)


def polygon_to_bbox(poly: Polygon) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) from a polygon."""
    b = poly.bounds
    return b[0], b[1], b[2], b[3]


# ---------------------------------------------------------------------------
# CRS reprojection
# ---------------------------------------------------------------------------

def reproject_gdf(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to target_crs; handle missing CRS gracefully."""
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    if gdf.crs.to_string() == CRS(target_crs).to_string():
        return gdf
    return gdf.to_crs(target_crs)


def to_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to WGS84 (EPSG:4326)."""
    return reproject_gdf(gdf, WGS84)


def to_lambert(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to Statistics Canada Lambert (EPSG:3347) for area/distance."""
    return reproject_gdf(gdf, LAMBERT)


def to_utm17n(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to UTM Zone 17N (EPSG:32617) for SAR raster alignment."""
    return reproject_gdf(gdf, UTM17N)


def transform_point(lon: float, lat: float, from_crs: str, to_crs: str) -> Tuple[float, float]:
    """Transform a single point between CRS."""
    t = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return t.transform(lon, lat)


# ---------------------------------------------------------------------------
# Spatial analysis helpers
# ---------------------------------------------------------------------------

def buffer_degrees(gdf: gpd.GeoDataFrame, meters: float) -> gpd.GeoDataFrame:
    """Buffer geometry by a distance in metres (reproject → buffer → back)."""
    projected = to_lambert(gdf)
    buffered = projected.copy()
    buffered["geometry"] = projected.geometry.buffer(meters)
    return to_4326(buffered)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two WGS84 points."""
    R = 6_371_000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def snap_to_line(point: Point, line: MultiLineString | any, tolerance_m: float = 5.0) -> Point:
    """Project a Point onto the nearest position on a line geometry."""
    return line.interpolate(line.project(point))


def voronoi_gdf(
    points_gdf: gpd.GeoDataFrame,
    clip_to: Polygon | None = None,
) -> gpd.GeoDataFrame:
    """Build Voronoi regions from point GeoDataFrame.

    Returns a GeoDataFrame of Voronoi polygon regions aligned with input point index.
    """
    from shapely.ops import unary_union, voronoi_diagram

    if len(points_gdf) < 2:
        raise ValueError("Need at least 2 points for Voronoi diagram.")

    multipoint = points_gdf.unary_union
    envelope = clip_to or multipoint.convex_hull.buffer(0.01)
    regions = voronoi_diagram(multipoint, envelope=envelope)

    # Align Voronoi polygons to input points by nearest centroid
    polys = list(regions.geoms)
    assignments = {}
    for idx, pt in enumerate(points_gdf.geometry):
        nearest = min(polys, key=lambda p: p.centroid.distance(pt))
        assignments[idx] = nearest

    result = points_gdf.copy()
    result["voronoi_poly"] = [assignments.get(i) for i in range(len(points_gdf))]
    return gpd.GeoDataFrame(result, geometry="voronoi_poly", crs=points_gdf.crs)
