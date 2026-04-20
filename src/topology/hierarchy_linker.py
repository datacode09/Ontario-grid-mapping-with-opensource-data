"""Hierarchy linker — connect grid layers into a voltage-hierarchy chain.

Hierarchy: generator → tx_substation → zone_substation → dx_substation →
           dx_feeder_node → secondary_transformer → building → service_point
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
from shapely.geometry import Point

from ..utils.logger import get_logger

log = get_logger(__name__)

# Maximum search radius (degrees) when linking adjacent hierarchy levels
_MAX_LINK_RADIUS_DEG = 0.05  # ~5 km at Ontario latitudes


class HierarchyLinker:
    """Link nodes across voltage hierarchy levels using spatial proximity."""

    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G

    def link_all(self) -> nx.DiGraph:
        """Run all hierarchy linkage passes in order."""
        log.info("HierarchyLinker: linking all hierarchy levels")
        self.link_generators_to_tx_substations()
        self.link_tx_to_zone_substations()
        self.link_zone_to_dx_substations()
        self.link_dx_substations_to_feeders()
        self.link_feeders_to_secondary_transformers()
        self.link_transformers_to_buildings()
        return self.G

    def link_generators_to_tx_substations(self) -> int:
        return self._link_levels("generator", "tx_substation", "tx_line")

    def link_tx_to_zone_substations(self) -> int:
        return self._link_levels("tx_substation", "zone_substation", "tx_line")

    def link_zone_to_dx_substations(self) -> int:
        return self._link_levels("zone_substation", "dx_substation", "primary_feeder")

    def link_dx_substations_to_feeders(self) -> int:
        return self._link_levels("dx_substation", "dx_feeder_node", "primary_feeder")

    def link_feeders_to_secondary_transformers(self) -> int:
        return self._link_levels("dx_feeder_node", "secondary_transformer", "secondary_lateral")

    def link_transformers_to_buildings(self) -> int:
        return self._link_levels("secondary_transformer", "building", "service_drop")

    # ------------------------------------------------------------------
    # Core linking logic
    # ------------------------------------------------------------------

    def _link_levels(
        self,
        from_type: str,
        to_type: str,
        edge_type: str,
        max_radius_deg: float = _MAX_LINK_RADIUS_DEG,
    ) -> int:
        """Link each from_type node to its nearest to_type node if not yet connected."""
        from_nodes = self._nodes_of_type(from_type)
        to_nodes = self._nodes_of_type(to_type)

        if not from_nodes or not to_nodes:
            log.debug(
                "HierarchyLinker: no nodes for %s → %s link", from_type, to_type
            )
            return 0

        added = 0
        for fid in from_nodes:
            # Skip if already has an outgoing edge to a to_type node
            already_linked = any(
                self.G.nodes[nbr].get("node_type") == to_type
                for nbr in self.G.successors(fid)
            )
            if already_linked:
                continue

            f_geom = self.G.nodes[fid].get("geometry")
            if f_geom is None:
                continue

            f_pt = f_geom if isinstance(f_geom, Point) else f_geom.centroid
            nearest = self._nearest_of_type(f_pt, to_nodes, max_radius_deg)

            if nearest:
                self.G.add_edge(fid, nearest, **{
                    "edge_type": edge_type,
                    "auto_linked": True,
                    "confidence": min(
                        self.G.nodes[fid].get("confidence", 0.75),
                        self.G.nodes[nearest].get("confidence", 0.75),
                    ),
                })
                added += 1

        log.info(
            "HierarchyLinker: %s → %s: added %d links", from_type, to_type, added
        )
        return added

    def _nodes_of_type(self, node_type: str) -> list[str]:
        return [
            nid
            for nid, data in self.G.nodes(data=True)
            if data.get("node_type") == node_type
        ]

    def _nearest_of_type(
        self,
        pt: Point,
        candidates: list[str],
        max_radius: float,
    ) -> Optional[str]:
        best, best_d = None, float("inf")
        for cid in candidates:
            geom = self.G.nodes[cid].get("geometry")
            if geom is None:
                continue
            c_pt = geom if isinstance(geom, Point) else geom.centroid
            d = pt.distance(c_pt)
            if d < best_d and d <= max_radius:
                best, best_d = cid, d
        return best
