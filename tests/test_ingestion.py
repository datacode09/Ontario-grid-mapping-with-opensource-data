"""Tests for ingestion layer — fetchers return correct GeoDataFrame schemas."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parents[1]))


class TestOEBFetcher:
    """OEB fetcher returns correct schema on failure/empty."""

    def test_fallback_service_areas_schema(self):
        from src.ingestion.oeb_fetcher import OEBFetcher
        fetcher = OEBFetcher(cache_dir=Path("/tmp/test_oeb_cache"))
        gdf = fetcher._fallback_service_areas()
        assert "ldc_name" in gdf.columns
        assert "geometry" in gdf.columns
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_synthetic_rrr_schema(self):
        from src.ingestion.oeb_fetcher import OEBFetcher
        fetcher = OEBFetcher(cache_dir=Path("/tmp/test_oeb_cache"))
        df = fetcher._synthetic_rrr_schema(2023)
        assert "ldc_name" in df.columns
        assert "saidi_minutes" in df.columns
        assert len(df) == 0  # Empty synthetic schema


class TestIESOFetcher:
    """IESO fetcher returns correct schema."""

    def test_empty_tx_gdf_schema(self):
        from src.ingestion.ieso_fetcher import IESOFetcher
        fetcher = IESOFetcher(cache_dir=Path("/tmp/test_ieso_cache"))
        gdf = fetcher._empty_tx_gdf()
        assert "facility_name" in gdf.columns
        assert "voltage_kv" in gdf.columns
        assert gdf.crs.to_epsg() == 4326

    def test_tie_line_nodes(self):
        from src.ingestion.ieso_fetcher import IESOFetcher
        fetcher = IESOFetcher(cache_dir=Path("/tmp/test_ieso_cache"))
        gdf = fetcher.get_tie_line_nodes()
        assert len(gdf) == 6  # 6 known tie lines
        assert "name" in gdf.columns
        assert gdf.crs.to_epsg() == 4326


class TestOSMFetcher:
    """OSM fetcher returns correct layer structure."""

    def test_empty_result_structure(self):
        from src.ingestion.osm_fetcher import OSMFetcher
        fetcher = OSMFetcher(cache_dir=Path("/tmp/test_osm_cache"))
        result = fetcher._empty_result()
        expected_keys = {"lines", "substations", "transformers", "poles", "towers", "generators", "switches"}
        assert set(result.keys()) == expected_keys
        for gdf in result.values():
            assert hasattr(gdf, "geometry")

    def test_overpass_query_structure(self):
        from src.ingestion.osm_fetcher import OSMFetcher
        fetcher = OSMFetcher(cache_dir=Path("/tmp/test_osm_cache"))
        query = fetcher._build_overpass_query((-79.9, 43.45, -79.5, 43.75))
        assert "power" in query
        assert "substation" in query
        assert "43.45,-79.9,43.75,-79.5" in query


class TestStatCanFetcher:
    """StatCan fetcher returns correct schema."""

    def test_empty_db_gdf_schema(self):
        from src.ingestion.statcan_fetcher import StatCanFetcher
        fetcher = StatCanFetcher(cache_dir=Path("/tmp/test_statcan_cache"))
        gdf = fetcher._empty_db_gdf()
        assert "dbuid" in gdf.columns
        assert "dwell_2021" in gdf.columns
        assert gdf.crs.to_epsg() == 4326
