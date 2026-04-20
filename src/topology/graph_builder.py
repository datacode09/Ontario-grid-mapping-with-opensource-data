"""Build directed NetworkX DiGraph from processed grid layers.

Node types:
  generator | tx_substation | zone_substation | dx_substation |
  dx_feeder_node | pole | secondary_transformer | building | service_point

Edge types:
  tx_line | primary_feeder | secondary_lateral | service_drop (inferred)
"""
from __future__ import annotations

import uuid
from typing import Optional

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point

from ..utils.logger import get_logger

log = get_logger(__name__)

# Confidence constants
CONF_AUTHORITATIVE = 1.00
CONF_OSM_CONFIRMED = 0.90
CONF_OSM_INFERRED  = 0.75
CONF_ML_URBAN      = 0.65
CONF_ML_SUBURBAN   = 0.45
CONF_GRIDFINDER    = 0.30
CONF_VORONOI       = 0.20


class GraphBuilder:
    """Construct a directed electricity network graph from GeoDataFrame layers."""

    def __init__(self) -> None:
        self.G: nx.DiGraph = nx.DiGraph()
        self._node_counter = 0

    # ------------------------------------------------------------------
    # Node addition
    # ------------------------------------------------------------------

    def add_generators(self, gdf: gpd.GeoDataFrame) -> list[str]:
        """Add generation facility nodes. Returns list of node IDs."""
        ids = []
        for _, row in gdf.iterrows():
            nid = self._node_id("gen", row)
            self.G.add_node(nid, **{
                "node_type":      "generator",
                "name":           str(row.get("generator_name", "Unknown Generator")),
                "fuel_type":      str(row.get("fuel_type", "")),
                "capacity_mw":    float(row.get("capacity_mw", 0) or 0),
                "lat":            row.geometry.y if row.geometry else None,
                "lon":            row.geometry.x if row.geometry else None,
                "geometry":       row.geometry,
                "operator":       str(row.get("operator", "")),
                "source":         str(row.get("source", "")),
                "confidence":     float(row.get("confidence", CONF_AUTHORITATIVE)),
                "sar_confidence": None,
            })
            ids.append(nid)
        log.info("GraphBuilder: added %d generator nodes", len(ids))
        return ids

    def add_tx_substations(self, gdf: gpd.GeoDataFrame) -> list[str]:
        return self._add_substation_layer(gdf, node_type="tx_substation")

    def add_zone_substations(self, gdf: gpd.GeoDataFrame) -> list[str]:
        return self._add_substation_layer(gdf, node_type="zone_substation")

    def add_dx_substations(self, gdf: gpd.GeoDataFrame) -> list[str]:
        return self._add_substation_layer(gdf, node_type="dx_substation")

    def _add_substation_layer(
        self, gdf: gpd.GeoDataFrame, node_type: str
    ) -> list[str]:
        ids = []
        for _, row in gdf.iterrows():
            nid = self._node_id(node_type, row)
            self.G.add_node(nid, **{
                "node_type":          node_type,
                "name":               str(row.get("facility_name", row.get("name", "Unknown"))),
                "voltage_kv":         float(row.get("voltage_kv", 0) or 0),
                "voltage_tier":       str(row.get("voltage_tier", "unknown")),
                "operator":           str(row.get("operator", "")),
                "lat":                row.geometry.y if row.geometry else None,
                "lon":                row.geometry.x if row.geometry else None,
                "geometry":           row.geometry,
                "source":             str(row.get("source", "")),
                "confidence":         float(row.get("confidence", CONF_AUTHORITATIVE)),
                "sar_flood_exposure": None,
                "sar_change_tier":    None,
                "resilience_risk_score": None,
                "risk_label":         None,
                "sar_confidence":     None,
            })
            ids.append(nid)
        log.info("GraphBuilder: added %d %s nodes", len(ids), node_type)
        return ids

    def add_feeder_nodes(self, gdf: gpd.GeoDataFrame) -> list[str]:
        ids = []
        for _, row in gdf.iterrows():
            nid = self._node_id("feeder_node", row)
            self.G.add_node(nid, **{
                "node_type":   "dx_feeder_node",
                "feeder_id":   str(row.get("feeder_id", "")),
                "voltage_kv":  float(row.get("voltage_kv", 0) or 0),
                "operator":    str(row.get("operator", "")),
                "geometry":    row.geometry,
                "source":      str(row.get("source", "")),
                "confidence":  float(row.get("confidence", CONF_OSM_INFERRED)),
                "sar_confidence": None,
            })
            ids.append(nid)
        log.info("GraphBuilder: added %d feeder nodes", len(ids))
        return ids

    def add_poles(self, gdf: gpd.GeoDataFrame, inferred: bool = False) -> list[str]:
        conf = CONF_ML_URBAN if inferred else CONF_OSM_CONFIRMED
        ids = []
        for _, row in gdf.iterrows():
            nid = self._node_id("pole", row)
            self.G.add_node(nid, **{
                "node_type":   "pole",
                "osm_id":      str(row.get("osm_id", "")),
                "operator":    str(row.get("operator", "")),
                "geometry":    row.geometry,
                "source":      "ML_inferred" if inferred else "OSM",
                "confidence":  float(row.get("confidence", conf)),
                "inferred":    inferred,
                "sar_confidence": None,
            })
            ids.append(nid)
        return ids

    def add_secondary_transformers(
        self, gdf: gpd.GeoDataFrame, inferred: bool = False
    ) -> list[str]:
        conf = CONF_ML_SUBURBAN if inferred else CONF_OSM_CONFIRMED
        ids = []
        for _, row in gdf.iterrows():
            nid = self._node_id("sec_tx", row)
            self.G.add_node(nid, **{
                "node_type":   "secondary_transformer",
                "operator":    str(row.get("operator", "")),
                "capacity_kva": float(row.get("capacity_kva", 25) or 25),
                "geometry":    row.geometry,
                "source":      "ML_inferred" if inferred else "OSM",
                "confidence":  float(row.get("confidence", conf)),
                "inferred":    inferred,
                "sar_confidence": None,
            })
            ids.append(nid)
        return ids

    def add_buildings(
        self, gdf: gpd.GeoDataFrame, confidence: float = CONF_VORONOI
    ) -> list[str]:
        ids = []
        for _, row in gdf.iterrows():
            nid = self._node_id("building", row)
            self.G.add_node(nid, **{
                "node_type":   "building",
                "building_id": str(row.get("building_id", "")),
                "address":     str(row.get("address", "")),
                "ldc_name":    str(row.get("ldc_name", "")),
                "geometry":    row.geometry.centroid if row.geometry else None,
                "source":      str(row.get("source", "Microsoft_BuildingFootprints")),
                "confidence":  confidence,
                "sar_confidence": None,
            })
            ids.append(nid)
        return ids

    # ------------------------------------------------------------------
    # Edge addition
    # ------------------------------------------------------------------

    def add_tx_lines(self, lines_gdf: gpd.GeoDataFrame) -> int:
        return self._add_line_edges(lines_gdf, edge_type="tx_line")

    def add_primary_feeders(self, lines_gdf: gpd.GeoDataFrame) -> int:
        return self._add_line_edges(lines_gdf, edge_type="primary_feeder")

    def add_service_drops(self, from_node: str, to_node: str) -> None:
        self.G.add_edge(from_node, to_node, edge_type="service_drop", confidence=CONF_VORONOI)

    def _add_line_edges(self, gdf: gpd.GeoDataFrame, edge_type: str) -> int:
        """Add edges for line features (approximate — uses spatial proximity)."""
        count = 0
        nodes_with_geom = [
            (nid, data)
            for nid, data in self.G.nodes(data=True)
            if data.get("geometry") is not None
        ]

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            if hasattr(geom, "coords"):
                start_pt = Point(list(geom.coords)[0])
                end_pt = Point(list(geom.coords)[-1])
            else:
                continue

            from_node = self._nearest_node(start_pt, nodes_with_geom)
            to_node = self._nearest_node(end_pt, nodes_with_geom)

            if from_node and to_node and from_node != to_node:
                self.G.add_edge(from_node, to_node, **{
                    "edge_type":  edge_type,
                    "voltage_kv": float(row.get("voltage_kv", 0) or 0),
                    "voltage_tier": str(row.get("voltage_tier", "unknown")),
                    "operator":   str(row.get("operator", "")),
                    "geometry":   geom,
                    "source":     str(row.get("source", "")),
                    "confidence": float(row.get("confidence", CONF_OSM_INFERRED)),
                    "length_km":  self._line_length_km(geom),
                })
                count += 1

        log.info("GraphBuilder: added %d %s edges", count, edge_type)
        return count

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _node_id(self, prefix: str, row) -> str:
        osm_id = row.get("osm_id") if hasattr(row, "get") else None
        if osm_id and str(osm_id) not in ("", "nan", "None"):
            return f"{prefix}_{osm_id}"
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    @staticmethod
    def _nearest_node(pt: Point, candidates: list) -> Optional[str]:
        if not candidates:
            return None
        best, best_d = None, float("inf")
        for nid, data in candidates:
            geom = data.get("geometry")
            if geom is not None:
                d = pt.distance(geom if isinstance(geom, Point) else geom.centroid)
                if d < best_d:
                    best, best_d = nid, d
        return best

    @staticmethod
    def _line_length_km(geom) -> float:
        from pyproj import Geod
        geod = Geod(ellps="WGS84")
        length = abs(geod.geometry_length(geom))
        return round(length / 1000.0, 3)
