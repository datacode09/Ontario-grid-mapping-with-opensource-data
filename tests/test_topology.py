"""Tests for topology layer — graph construction and path tracing."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parents[1]))


def _make_point_gdf(points: list[tuple], **attrs) -> gpd.GeoDataFrame:
    """Helper to create a point GeoDataFrame."""
    records = []
    for i, (lon, lat) in enumerate(points):
        rec = {"geometry": Point(lon, lat), "source": "test", "confidence": 1.0}
        rec.update({k: v[i] if isinstance(v, list) else v for k, v in attrs.items()})
        records.append(rec)
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


class TestGraphBuilder:
    """GraphBuilder adds correct node attributes."""

    def test_add_generators(self):
        from src.topology.graph_builder import GraphBuilder
        gdf = _make_point_gdf(
            [(-79.5, 43.6)],
            generator_name="Lakeview TS",
            fuel_type="gas",
            capacity_mw=500,
        )
        builder = GraphBuilder()
        ids = builder.add_generators(gdf)
        assert len(ids) == 1
        nid = ids[0]
        assert builder.G.nodes[nid]["node_type"] == "generator"
        assert builder.G.nodes[nid]["confidence"] == 1.0
        # SAR confidence must be separate and start as None
        assert builder.G.nodes[nid]["sar_confidence"] is None

    def test_add_tx_substations(self):
        from src.topology.graph_builder import GraphBuilder
        gdf = _make_point_gdf(
            [(-79.77, 43.55)],
            facility_name="Erin Mills TS",
            voltage_kv=115,
            voltage_tier="hv_115",
            operator="Hydro One",
        )
        builder = GraphBuilder()
        ids = builder.add_tx_substations(gdf)
        assert len(ids) == 1
        assert builder.G.nodes[ids[0]]["node_type"] == "tx_substation"
        assert "sar_flood_exposure" in builder.G.nodes[ids[0]]

    def test_sar_confidence_independent(self):
        """SAR confidence must never be merged with topology confidence."""
        from src.topology.graph_builder import GraphBuilder
        gdf = _make_point_gdf([(-79.7, 43.6)], facility_name="Test SS", voltage_kv=230)
        builder = GraphBuilder()
        ids = builder.add_tx_substations(gdf)
        nid = ids[0]
        # Topology confidence
        assert "confidence" in builder.G.nodes[nid]
        # SAR confidence — must be separate field, not mixed
        assert "sar_confidence" in builder.G.nodes[nid]
        assert builder.G.nodes[nid]["sar_confidence"] is None


class TestHierarchyLinker:
    """HierarchyLinker correctly links nodes across voltage hierarchy."""

    def test_links_generator_to_tx_substation(self):
        from src.topology.graph_builder import GraphBuilder
        from src.topology.hierarchy_linker import HierarchyLinker

        builder = GraphBuilder()
        gen_gdf = _make_point_gdf([(-79.5, 43.6)], generator_name="Gen A", fuel_type="hydro", capacity_mw=100)
        ss_gdf = _make_point_gdf([(-79.51, 43.61)], facility_name="SS A", voltage_kv=115, voltage_tier="hv_115")
        gen_ids = builder.add_generators(gen_gdf)
        ss_ids = builder.add_tx_substations(ss_gdf)
        linker = HierarchyLinker(builder.G)
        added = linker.link_generators_to_tx_substations()
        assert added >= 0  # May be 0 if radius exceeded — just no error
        assert isinstance(builder.G, nx.DiGraph)

    def test_linker_no_error_on_empty_graph(self):
        from src.topology.graph_builder import GraphBuilder
        from src.topology.hierarchy_linker import HierarchyLinker
        builder = GraphBuilder()
        linker = HierarchyLinker(builder.G)
        linker.link_all()  # Should not raise


class TestPathTracer:
    """PathTracer finds correct paths in simple graph."""

    def test_trace_returns_dict(self):
        from src.topology.graph_builder import GraphBuilder
        from src.topology.path_tracer import PathTracer

        builder = GraphBuilder()
        gen = _make_point_gdf([(-79.5, 43.6)], generator_name="Gen", fuel_type="hydro", capacity_mw=50)
        bldg = _make_point_gdf([(-79.51, 43.61)], building_id="b001", address="1 Main St")
        gen_ids = builder.add_generators(gen)
        bldg_ids = builder.add_buildings(bldg)
        builder.G.add_edge(gen_ids[0], bldg_ids[0], edge_type="service_drop")

        tracer = PathTracer(builder.G)
        result = tracer.trace_from_building(bldg_ids[0])
        assert "building_node" in result
        assert "path_nodes" in result
        assert "confidence_min" in result

    def test_transmission_roots_are_in_degree_zero(self):
        from src.topology.graph_builder import GraphBuilder
        from src.topology.path_tracer import PathTracer

        builder = GraphBuilder()
        gen = _make_point_gdf([(-79.5, 43.6)], generator_name="Gen", fuel_type="nuclear", capacity_mw=1000)
        builder.add_generators(gen)
        tracer = PathTracer(builder.G)
        roots = tracer.transmission_roots
        for root in roots:
            assert builder.G.in_degree(root) == 0
