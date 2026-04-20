"""Graph exporter — serialize the NetworkX graph to GeoJSON and GeoPackage."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point, mapping

from ..utils.logger import get_logger

log = get_logger(__name__)


class GraphExporter:
    """Export a NetworkX DiGraph to geospatial formats."""

    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G

    # ------------------------------------------------------------------
    # GeoJSON export
    # ------------------------------------------------------------------

    def to_geojson(
        self,
        output_path: Path | str,
        include_edges: bool = True,
    ) -> dict:
        """Export graph as GeoJSON FeatureCollection.

        Creates two files: <name>_nodes.geojson and <name>_edges.geojson
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nodes_path = output_path.parent / (output_path.stem + "_nodes.geojson")
        edges_path = output_path.parent / (output_path.stem + "_edges.geojson")

        node_gdf = self.nodes_to_gdf()
        if len(node_gdf):
            node_gdf.to_file(nodes_path, driver="GeoJSON")
            log.info("GraphExporter: wrote %d nodes to %s", len(node_gdf), nodes_path)

        if include_edges:
            edge_gdf = self.edges_to_gdf()
            if len(edge_gdf):
                edge_gdf.to_file(edges_path, driver="GeoJSON")
                log.info("GraphExporter: wrote %d edges to %s", len(edge_gdf), edges_path)

        return {"nodes": str(nodes_path), "edges": str(edges_path)}

    def to_gpkg(self, output_path: Path | str) -> Path:
        """Export to GeoPackage with separate layers for nodes and edges."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        node_gdf = self.nodes_to_gdf()
        if len(node_gdf):
            node_gdf.to_file(output_path, driver="GPKG", layer="nodes")

        edge_gdf = self.edges_to_gdf()
        if len(edge_gdf):
            edge_gdf.to_file(output_path, driver="GPKG", layer="edges")

        log.info("GraphExporter: wrote graph to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # GeoDataFrame conversion
    # ------------------------------------------------------------------

    def nodes_to_gdf(self) -> gpd.GeoDataFrame:
        """Convert graph nodes to a GeoDataFrame."""
        records = []
        for nid, data in self.G.nodes(data=True):
            rec = {"node_id": nid}
            geom = data.pop("geometry", None) if "geometry" in data else None
            rec.update({k: v for k, v in data.items() if not isinstance(v, dict)})
            rec["geometry"] = geom
            records.append(rec)

        if not records:
            return gpd.GeoDataFrame(columns=["node_id", "geometry"], geometry="geometry", crs="EPSG:4326")

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        return gdf

    def edges_to_gdf(self) -> gpd.GeoDataFrame:
        """Convert graph edges to a GeoDataFrame (LineStrings between nodes)."""
        records = []
        for u, v, data in self.G.edges(data=True):
            rec = {"from_node": u, "to_node": v}
            geom = data.get("geometry")

            if geom is None:
                # Construct line from node geometries
                u_geom = self.G.nodes[u].get("geometry")
                v_geom = self.G.nodes[v].get("geometry")
                if u_geom and v_geom:
                    u_pt = u_geom if isinstance(u_geom, Point) else u_geom.centroid
                    v_pt = v_geom if isinstance(v_geom, Point) else v_geom.centroid
                    geom = LineString([u_pt, v_pt])

            rec.update({k: v for k, v in data.items()
                        if k != "geometry" and not isinstance(v, dict)})
            rec["geometry"] = geom
            records.append(rec)

        if not records:
            return gpd.GeoDataFrame(columns=["from_node", "to_node", "geometry"],
                                    geometry="geometry", crs="EPSG:4326")

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        return gdf

    # ------------------------------------------------------------------
    # Building→Transformer lookup table
    # ------------------------------------------------------------------

    def building_transformer_lookup(self, path_results: list[dict]) -> pd.DataFrame:
        """Build a building→transformer Parquet lookup table.

        Parameters
        ----------
        path_results: Output from PathTracer.batch_trace()
        """
        records = []
        for result in path_results:
            if "error" in result:
                continue
            records.append({
                "building_node":       result.get("building_node"),
                "secondary_transformer": result.get("secondary_transformer"),
                "dx_substation":       result.get("dx_substation"),
                "zone_substation":     result.get("zone_substation"),
                "tx_substation":       result.get("tx_substation"),
                "path_length":         result.get("path_length"),
                "confidence_min":      result.get("confidence_min"),
            })
        return pd.DataFrame(records)

    def save_lookup_parquet(self, df: pd.DataFrame, output_path: Path | str) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        log.info("GraphExporter: wrote %d building lookup records to %s", len(df), output_path)
        return output_path
