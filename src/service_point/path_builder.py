"""Path builder — assemble full grid path from building to transmission root."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import networkx as nx
import pandas as pd

from ..topology.path_tracer import PathTracer
from ..utils.logger import get_logger

log = get_logger(__name__)


class PathBuilder:
    """Build complete grid path for a given address or building.

    Integrates: Geocoder → LDCResolver → TransformerFinder → PathTracer
    into a single lookup pipeline.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        ldc_resolver,
        transformer_finder,
        geocoder,
    ) -> None:
        self.G = G
        self.ldc_resolver = ldc_resolver
        self.transformer_finder = transformer_finder
        self.geocoder = geocoder
        self._path_tracer = PathTracer(G)

    def lookup_address(self, address: str, radius_m: float = 200.0) -> dict:
        """Full pipeline lookup for a civic address.

        Parameters
        ----------
        address:  Civic address string (e.g. "7086 Tamar Mews, Mississauga, ON").
        radius_m: Transformer search radius in metres.

        Returns
        -------
        Dict with:
          address, ldc_name, secondary_transformer, dx_substation,
          zone_substation, tx_substation, generator,
          path_nodes, confidence_min, confidence_label, sar_exposure
        """
        # Step 1: Geocode
        point = self.geocoder.geocode(address)
        if point is None:
            return {"address": address, "error": "Geocoding failed", "status": "not_found"}

        # Step 2: Resolve LDC
        ldc_info = self.ldc_resolver.resolve_point(point.x, point.y) or {}

        # Step 3: Find nearest transformer
        tx_info = self.transformer_finder.find_nearest(point, radius_m=radius_m)

        # Step 4: Trace grid path
        path_result = {}
        if tx_info:
            # Find the building node closest to this transformer
            building_node = self._find_building_node_near_point(point)
            if building_node:
                try:
                    path_result = self._path_tracer.trace_from_building(building_node)
                except Exception as exc:
                    log.debug("Path trace error: %s", exc)

        # Step 5: Extract SAR exposure if available
        sar_exposure = self._get_sar_exposure(
            path_result.get("zone_substation") or path_result.get("tx_substation")
        )

        return {
            "address":                address,
            "geocoded_lat":           point.y,
            "geocoded_lon":           point.x,
            "ldc_name":               ldc_info.get("ldc_name"),
            "ldc_source":             ldc_info.get("source"),
            "secondary_transformer":  (
                tx_info.get("tx_id") if tx_info
                else path_result.get("secondary_transformer")
            ),
            "tx_distance_m":          tx_info.get("distance_m") if tx_info else None,
            "dx_substation":          path_result.get("dx_substation"),
            "zone_substation":        path_result.get("zone_substation"),
            "tx_substation":          path_result.get("tx_substation"),
            "generator":              path_result.get("generator"),
            "path_nodes":             path_result.get("path_nodes", []),
            "path_length":            path_result.get("path_length", 0),
            "confidence_min":         path_result.get("confidence_min", 0.20),
            "confidence_label":       self._confidence_label(path_result.get("confidence_min", 0.20)),
            "sar_flood_exposure":     sar_exposure.get("sar_flood_exposure"),
            "sar_risk_label":         sar_exposure.get("risk_label"),
            "status":                 "success",
        }

    def _find_building_node_near_point(self, point) -> Optional[str]:
        """Find the nearest building node in the graph to a point."""
        from shapely.geometry import Point
        best_dist = float("inf")
        best_node = None

        for nid, data in self.G.nodes(data=True):
            if data.get("node_type") != "building":
                continue
            geom = data.get("geometry")
            if geom is None:
                continue
            g_pt = geom if isinstance(geom, Point) else geom.centroid
            dist = point.distance(g_pt)
            if dist < best_dist:
                best_dist, best_node = dist, nid

        return best_node

    def _get_sar_exposure(self, substation_node: Optional[str]) -> dict:
        """Extract SAR resilience attributes from a substation node."""
        if not substation_node or substation_node not in self.G:
            return {}
        data = self.G.nodes[substation_node]
        return {
            "sar_flood_exposure": data.get("sar_flood_exposure"),
            "risk_label":         data.get("risk_label"),
            "resilience_risk_score": data.get("resilience_risk_score"),
        }

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.90:
            return "HIGH (authoritative)"
        elif score >= 0.65:
            return "MEDIUM (OSM / ML urban)"
        elif score >= 0.30:
            return "LOW (ML inferred)"
        return "VERY LOW (Voronoi proxy)"
