"""Tests for SAR resilience exposure scorer."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).parents[1]))


def _mock_substations() -> gpd.GeoDataFrame:
    """Return a small mock substations GeoDataFrame."""
    records = [
        {"facility_name": "SS A", "voltage_kv": 115, "operator": "Hydro One",
         "geometry": Point(-79.77, 43.55), "source": "IESO", "confidence": 1.0},
        {"facility_name": "SS B", "voltage_kv": 230, "operator": "Hydro One",
         "geometry": Point(-79.73, 43.54), "source": "IESO", "confidence": 1.0},
    ]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


class TestExposureScorer:
    """score_substation_resilience returns correct schema and labels."""

    def test_returns_gdf_with_risk_columns(self):
        from src.sar.exposure_scorer import score_substation_resilience
        substations = _mock_substations()
        result = score_substation_resilience(
            substations=substations,
            flood_mask_path=None,
            change_summary=[],
            dem_path=None,
            eccc_flood_zones=None,
            oeb_saidi=None,
        )
        assert "resilience_risk_score" in result.columns
        assert "risk_label" in result.columns
        assert "sar_flood_exposure" in result.columns
        assert "sar_change_tier" in result.columns
        # SAR confidence must be separate from topology confidence
        assert "sar_confidence" in result.columns
        assert "confidence" not in result.columns or "sar_confidence" in result.columns

    def test_risk_score_range(self):
        from src.sar.exposure_scorer import score_substation_resilience
        substations = _mock_substations()
        result = score_substation_resilience(
            substations=substations,
            flood_mask_path=None,
            change_summary=[],
            dem_path=None,
            eccc_flood_zones=None,
            oeb_saidi=None,
        )
        scores = result["resilience_risk_score"].dropna()
        assert all(0 <= s <= 1 for s in scores)

    def test_risk_labels_valid(self):
        from src.sar.exposure_scorer import score_substation_resilience
        substations = _mock_substations()
        result = score_substation_resilience(
            substations=substations,
            flood_mask_path=None,
            change_summary=[],
            dem_path=None,
            eccc_flood_zones=None,
            oeb_saidi=None,
        )
        valid_labels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for label in result["risk_label"].dropna():
            assert label in valid_labels

    def test_sar_fields_tagged_correctly(self):
        """All SAR products must be tagged source=SAR_contextual_RS."""
        from src.sar.exposure_scorer import score_substation_resilience
        substations = _mock_substations()
        result = score_substation_resilience(
            substations=substations,
            flood_mask_path=None,
            change_summary=[],
            dem_path=None,
            eccc_flood_zones=None,
            oeb_saidi=None,
        )
        assert all(result["source"] == "SAR_contextual_RS")
        assert all(result["is_exact_asset_geometry"] == False)

    def test_change_tier_with_summary(self):
        """Change tier from corridor summary is correctly assigned."""
        from src.sar.exposure_scorer import score_substation_resilience
        substations = _mock_substations()
        change_summary = [
            {"corridor_id": "c001", "change_tier": "significant", "voltage_kv": 115}
        ]
        result = score_substation_resilience(
            substations=substations,
            flood_mask_path=None,
            change_summary=change_summary,
            dem_path=None,
            eccc_flood_zones=None,
            oeb_saidi=None,
        )
        # Worst tier should be 'significant'
        assert all(result["sar_change_tier"] == "significant")

    def test_risk_label_function(self):
        from src.sar.exposure_scorer import _risk_label
        assert _risk_label(0.0) == "LOW"
        assert _risk_label(0.25) == "MEDIUM"
        assert _risk_label(0.50) == "HIGH"
        assert _risk_label(0.75) == "CRITICAL"
        assert _risk_label(1.0) == "CRITICAL"


class TestSARConfidenceSeparation:
    """Verify SAR confidence and topology confidence are never merged."""

    def test_confidence_fields_are_separate(self):
        """topology.confidence and SAR sar_confidence must be independent."""
        from src.topology.graph_builder import GraphBuilder
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            [{"facility_name": "Test SS", "voltage_kv": 115, "confidence": 1.0,
              "geometry": Point(-79.7, 43.6), "source": "IESO"}],
            geometry="geometry", crs="EPSG:4326"
        )
        builder = GraphBuilder()
        ids = builder.add_tx_substations(gdf)
        node = builder.G.nodes[ids[0]]

        # Both must exist and be independent
        assert "confidence" in node
        assert "sar_confidence" in node
        # topology confidence comes from the data source (1.0 for IESO)
        assert node["confidence"] == 1.0
        # sar_confidence starts as None until SAR module assigns it
        assert node["sar_confidence"] is None
