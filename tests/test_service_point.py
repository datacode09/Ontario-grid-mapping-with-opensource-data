"""Tests for service point layer — LDC resolver, voronoi assigner."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).parents[1]))


def _mock_ldc_boundaries() -> gpd.GeoDataFrame:
    """Return mock LDC service area polygons."""
    records = [
        {"ldc_name": "Toronto Hydro",    "ldc_id": "1",
         "geometry": Polygon([(-79.7, 43.6), (-79.3, 43.6), (-79.3, 43.9), (-79.7, 43.9)])},
        {"ldc_name": "Alectra",          "ldc_id": "2",
         "geometry": Polygon([(-79.9, 43.4), (-79.5, 43.4), (-79.5, 43.6), (-79.9, 43.6)])},
    ]
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    gdf["source"] = "OEB_authoritative"
    gdf["confidence"] = 1.0
    return gdf


def _mock_transformers() -> gpd.GeoDataFrame:
    """Return mock secondary transformer points."""
    records = [
        {"tx_id": "tx_001", "confidence": 0.90, "source": "OSM",
         "geometry": Point(-79.77, 43.55)},
        {"tx_id": "tx_002", "confidence": 0.45, "source": "ML",
         "geometry": Point(-79.75, 43.56)},
        {"tx_id": "tx_003", "confidence": 0.45, "source": "ML",
         "geometry": Point(-79.73, 43.54)},
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


class TestLDCResolver:
    """LDC resolver correctly assigns service areas."""

    def test_resolve_point_in_toronto_hydro(self):
        from src.service_point.ldc_resolver import LDCResolver
        ldc = LDCResolver(_mock_ldc_boundaries())
        result = ldc.resolve_point(-79.5, 43.75)
        assert result is not None
        assert result["ldc_name"] == "Toronto Hydro"
        assert result["confidence"] == 1.0

    def test_resolve_point_in_alectra(self):
        from src.service_point.ldc_resolver import LDCResolver
        ldc = LDCResolver(_mock_ldc_boundaries())
        result = ldc.resolve_point(-79.7, 43.5)
        assert result is not None
        assert result["ldc_name"] == "Alectra"

    def test_resolve_point_outside_returns_none(self):
        from src.service_point.ldc_resolver import LDCResolver
        ldc = LDCResolver(_mock_ldc_boundaries())
        # Point far outside all boundaries
        result = ldc.resolve_point(-80.5, 44.0)
        assert result is None

    def test_list_ldcs(self):
        from src.service_point.ldc_resolver import LDCResolver
        ldc = LDCResolver(_mock_ldc_boundaries())
        names = ldc.list_ldcs()
        assert "Toronto Hydro" in names
        assert "Alectra" in names

    def test_requires_valid_boundaries(self):
        from src.service_point.ldc_resolver import LDCResolver
        with pytest.raises(ValueError):
            LDCResolver(gpd.GeoDataFrame())


class TestTransformerFinder:
    """TransformerFinder returns correct nearest transformer."""

    def test_find_nearest_returns_tx_id(self):
        from src.service_point.transformer_finder import TransformerFinder
        finder = TransformerFinder(_mock_transformers(), max_radius_m=5000)
        pt = Point(-79.77, 43.55)
        result = finder.find_nearest(pt)
        assert result is not None
        assert result["tx_id"] == "tx_001"
        assert result["distance_m"] < 1000

    def test_returns_none_outside_radius(self):
        from src.service_point.transformer_finder import TransformerFinder
        finder = TransformerFinder(_mock_transformers(), max_radius_m=1)  # 1 metre
        # Point far away
        result = finder.find_nearest(Point(-80.5, 44.0))
        assert result is None


class TestVoronoiAssigner:
    """VoronoiAssigner assigns buildings to correct transformers."""

    def test_assigns_each_building_to_transformer(self):
        from src.service_point.voronoi_assigner import VoronoiAssigner
        txs = _mock_transformers()
        buildings = gpd.GeoDataFrame(
            [{"building_id": f"b{i}", "geometry": Point(-79.75 + i * 0.01, 43.55),
              "source": "test", "confidence": 1.0}
             for i in range(5)],
            geometry="geometry", crs="EPSG:4326"
        )
        assigner = VoronoiAssigner(txs)
        result = assigner.assign(buildings)
        assert "assigned_tx_id" in result.columns
        assert "assignment_method" in result.columns

    def test_raises_on_empty_transformers(self):
        from src.service_point.voronoi_assigner import VoronoiAssigner
        with pytest.raises(ValueError):
            VoronoiAssigner(gpd.GeoDataFrame())
