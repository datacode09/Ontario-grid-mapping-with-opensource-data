"""Path tracer — trace the full grid path from a building to a transmission root."""
from __future__ import annotations

from typing import Optional

import networkx as nx

from ..utils.logger import get_logger

log = get_logger(__name__)


class PathTracer:
    """Trace grid paths within a directed NetworkX DiGraph.

    Paths flow from transmission roots → generation nodes down to buildings.
    For a given building node, we find the reverse path up to the nearest
    transmission root.
    """

    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G
        self._root_nodes = None

    @property
    def transmission_roots(self) -> list[str]:
        """Nodes with no predecessors — these are the transmission entry points."""
        if self._root_nodes is None:
            self._root_nodes = [
                n for n in self.G.nodes if self.G.in_degree(n) == 0
            ]
        return self._root_nodes

    def trace_from_building(self, building_node: str) -> Optional[dict]:
        """Find the full electricity supply chain for a building node.

        Returns
        -------
        Dict with keys:
          building_node, path_nodes, path_edges,
          secondary_transformer, dx_substation, zone_substation,
          tx_substation, generator, confidence_min
        Raises ValueError if building_node not in graph.
        """
        if building_node not in self.G:
            raise ValueError(f"Node '{building_node}' not in graph")

        # Reverse graph for ancestor traversal
        rev = self.G.reverse(copy=False)

        # BFS upward to find roots
        path_nodes = []
        current = building_node
        visited = {building_node}
        result = {"building_node": building_node}

        # Walk up the hierarchy
        type_assignments = {
            "secondary_transformer": None,
            "dx_substation": None,
            "zone_substation": None,
            "tx_substation": None,
            "generator": None,
        }

        try:
            # Use NetworkX shortest path to any transmission root
            for root in self.transmission_roots:
                try:
                    path = nx.shortest_path(self.G, root, building_node)
                    if path:
                        path_nodes = path
                        break
                except nx.NetworkXNoPath:
                    continue
        except Exception:
            path_nodes = [building_node]

        if not path_nodes:
            path_nodes = [building_node]

        # Extract hierarchy from path
        for node in path_nodes:
            node_type = self.G.nodes[node].get("node_type")
            if node_type in type_assignments and type_assignments[node_type] is None:
                type_assignments[node_type] = node

        # Collect path edges
        path_edges = [
            (path_nodes[i], path_nodes[i + 1])
            for i in range(len(path_nodes) - 1)
        ]

        # Minimum confidence along path
        confidences = [
            self.G.nodes[n].get("confidence", 1.0) for n in path_nodes
        ]
        conf_min = min(confidences) if confidences else 0.0

        result.update({
            "path_nodes":            path_nodes,
            "path_edges":            path_edges,
            "path_length":           len(path_nodes),
            "secondary_transformer": type_assignments["secondary_transformer"],
            "dx_substation":         type_assignments["dx_substation"],
            "zone_substation":       type_assignments["zone_substation"],
            "tx_substation":         type_assignments["tx_substation"],
            "generator":             type_assignments["generator"],
            "confidence_min":        conf_min,
        })

        return result

    def batch_trace(
        self, building_nodes: list[str], max_workers: int = 4
    ) -> list[dict]:
        """Trace paths for a list of building nodes."""
        results = []
        for node in building_nodes:
            try:
                results.append(self.trace_from_building(node))
            except Exception as exc:
                log.warning("Path trace failed for %s: %s", node, exc)
                results.append({"building_node": node, "error": str(exc)})
        return results

    def find_downstream_buildings(self, node_id: str) -> list[str]:
        """Return all building nodes reachable downstream from a given node."""
        reachable = nx.descendants(self.G, node_id)
        return [
            n for n in reachable
            if self.G.nodes[n].get("node_type") == "building"
        ]

    def substation_load_estimate(self, substation_node: str) -> dict:
        """Estimate building count and load served by a substation."""
        buildings = self.find_downstream_buildings(substation_node)
        return {
            "substation": substation_node,
            "downstream_buildings": len(buildings),
            "estimated_peak_kw": len(buildings) * 5.0,  # ~5 kW average residential
        }
