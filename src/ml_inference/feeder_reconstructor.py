"""Feeder reconstructor — rebuild distribution feeder topology from detected poles.

Implements the graph inference step from Stanford GridMapping (Source 17):
  1. Snap detected poles to nearest road segment
  2. Build minimum spanning tree connecting poles along roads
  3. Assign feeder IDs to contiguous pole chains
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points

from ..utils.geometry import to_lambert
from ..utils.logger import get_logger

log = get_logger(__name__)


class FeederReconstructor:
    """Reconstruct distribution feeder geometry from detected pole locations."""

    def __init__(self, snap_radius_m: float = 10.0) -> None:
        self._snap_radius_m = snap_radius_m

    def reconstruct(
        self,
        poles_gdf: gpd.GeoDataFrame,
        roads_gdf: gpd.GeoDataFrame,
        max_pole_spacing_m: float = 80.0,
    ) -> gpd.GeoDataFrame:
        """Build feeder LineStrings from pole point cloud + road network.

        Parameters
        ----------
        poles_gdf:          Detected/OSM pole points.
        roads_gdf:          Road network as LineStrings.
        max_pole_spacing_m: Maximum distance between connected poles.

        Returns
        -------
        GeoDataFrame of reconstructed feeder LineStrings.
        """
        if len(poles_gdf) < 2:
            log.warning("FeederReconstructor: need at least 2 poles")
            return self._empty_feeder_gdf()

        # Work in projected CRS for distance calculations
        poles_lambert = to_lambert(poles_gdf)
        roads_lambert = to_lambert(roads_gdf) if len(roads_gdf) else None

        # Snap poles to roads
        if roads_lambert is not None and len(roads_lambert):
            poles_lambert = self._snap_poles_to_roads(poles_lambert, roads_lambert)

        # Build pole proximity graph
        pole_graph = self._build_pole_graph(poles_lambert, max_pole_spacing_m)

        # Extract feeder chains as LineStrings
        feeder_lines = self._extract_feeder_lines(pole_graph, poles_lambert)

        if not feeder_lines:
            return self._empty_feeder_gdf()

        # Reproject back to WGS84
        result_gdf = gpd.GeoDataFrame(
            feeder_lines, geometry="geometry", crs="EPSG:3347"
        ).to_crs("EPSG:4326")

        result_gdf["source"] = "GridMapping_ML_feeder"
        result_gdf["confidence"] = poles_gdf["confidence"].mean() if "confidence" in poles_gdf.columns else 0.55
        result_gdf["edge_type"] = "primary_feeder"
        result_gdf["inferred"] = True

        log.info("FeederReconstructor: reconstructed %d feeder segments", len(result_gdf))
        return result_gdf

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snap_poles_to_roads(
        self, poles: gpd.GeoDataFrame, roads: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Snap each pole to its nearest point on the road network."""
        roads_union = roads.geometry.unary_union
        snapped_geoms = []
        for geom in poles.geometry:
            nearest_pt, _ = nearest_points(geom, roads_union)
            dist = geom.distance(nearest_pt)
            snapped_geoms.append(nearest_pt if dist < self._snap_radius_m else geom)

        poles = poles.copy()
        poles["geometry"] = snapped_geoms
        return poles

    def _build_pole_graph(
        self, poles: gpd.GeoDataFrame, max_spacing: float
    ) -> nx.Graph:
        """Build undirected graph connecting nearby poles."""
        G = nx.Graph()
        coords = np.array([(g.x, g.y) for g in poles.geometry])
        n = len(coords)

        for i in range(n):
            G.add_node(i, pos=coords[i])

        # Connect poles within max_spacing
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=max_spacing)
        for i, j in pairs:
            dist = np.linalg.norm(coords[i] - coords[j])
            G.add_edge(i, j, weight=dist)

        return G

    def _extract_feeder_lines(
        self, G: nx.Graph, poles: gpd.GeoDataFrame
    ) -> list[dict]:
        """Extract connected components as LineStrings."""
        records = []
        coords = np.array([(g.x, g.y) for g in poles.geometry])

        for comp_id, component in enumerate(nx.connected_components(G)):
            nodes = sorted(component)
            if len(nodes) < 2:
                continue

            # Build ordered path through component using DFS
            subgraph = G.subgraph(nodes)
            try:
                path = list(nx.dfs_preorder_nodes(subgraph, source=nodes[0]))
            except Exception:
                path = nodes

            pts = [Point(coords[i]) for i in path]
            if len(pts) >= 2:
                line = LineString(pts)
                records.append({
                    "feeder_id": f"inferred_{comp_id:05d}",
                    "pole_count": len(path),
                    "geometry": line,
                })

        return records

    def _empty_feeder_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["feeder_id", "pole_count", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
