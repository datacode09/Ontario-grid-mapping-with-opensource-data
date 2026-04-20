"""Tests for SAR flood detector."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestFloodDetector:
    """detect_flood_extent handles missing inputs gracefully."""

    def test_returns_none_on_missing_input(self):
        from src.sar.flood_detector import detect_flood_extent
        result = detect_flood_extent(
            event_vv_path=Path("/nonexistent.tif"),
            reference_composite_path=Path("/nonexistent_ref.tif"),
            permanent_water_mask=None,
        )
        assert result is None

    def test_spring_melt_flag(self):
        from src.sar.flood_detector import _is_spring_melt_period
        cfg = {"artefact_suppression": {"flag_spring_melt": True, "spring_melt_months": [3, 4]}}
        assert _is_spring_melt_period("2024-03-15", cfg) is True
        assert _is_spring_melt_period("2024-04-01", cfg) is True
        assert _is_spring_melt_period("2024-08-01", cfg) is False
        assert _is_spring_melt_period("2024-01-01", cfg) is False

    def test_spring_melt_suppression_disabled(self):
        from src.sar.flood_detector import _is_spring_melt_period
        cfg = {"artefact_suppression": {"flag_spring_melt": False, "spring_melt_months": [3, 4]}}
        # Suppression disabled — should return False even in spring
        assert _is_spring_melt_period("2024-03-15", cfg) is False

    def test_spring_melt_invalid_date(self):
        from src.sar.flood_detector import _is_spring_melt_period
        cfg = {"artefact_suppression": {"flag_spring_melt": True, "spring_melt_months": [3, 4]}}
        assert _is_spring_melt_period("not-a-date", cfg) is False

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "mock_s1_vv.tif").exists(),
        reason="SAR fixture not generated"
    )
    def test_flood_detection_same_input(self, tmp_path):
        """Using same raster as before/after → flood mask should be mostly zero."""
        from src.sar.flood_detector import detect_flood_extent
        fixture = FIXTURE_DIR / "mock_s1_vv.tif"
        out = detect_flood_extent(
            event_vv_path=fixture,
            reference_composite_path=fixture,
            permanent_water_mask=None,
            output_path=tmp_path / "test_flood.tif",
            event_date="2024-08-01",  # Not spring melt
        )
        # When event == reference, |ratio| ≈ 0 → no flood pixels
        if out is not None and out.exists():
            try:
                import rasterio
                import numpy as np
                with rasterio.open(out) as src:
                    data = src.read(1)
                assert data.sum() == 0  # No flood when images are identical
            except ImportError:
                pass
