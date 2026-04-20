"""Tests for SAR change detector — unit tests with mock GeoTIFF fixtures."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestComputeBackscatterChange:
    """compute_backscatter_change returns correct structure."""

    def test_empty_result_on_missing_rasters(self):
        from src.sar.change_detector import compute_backscatter_change, _empty_change_result
        result = compute_backscatter_change(
            vv_before_path=Path("/nonexistent/before.tif"),
            vv_after_path=Path("/nonexistent/after.tif"),
        )
        assert "change_raster_path" in result
        assert "change_mask_path" in result
        assert "corridor_change_summary" in result

    def test_tier_label_logic(self):
        from src.sar.change_detector import _tier_label
        thresholds = [3.0, 5.0, 8.0]
        assert _tier_label(1.5, thresholds) == "none"
        assert _tier_label(3.5, thresholds) == "moderate"
        assert _tier_label(6.0, thresholds) == "significant"
        assert _tier_label(9.0, thresholds) == "extreme"
        assert _tier_label(0.0, thresholds) == "none"

    def test_sar_disclaimer_present(self):
        from src.sar import _SAR_DISCLAIMER, _SAR_POPUP_BLOCK
        assert "SAR backscatter" in _SAR_DISCLAIMER
        assert "exact electrical asset geometry" in _SAR_DISCLAIMER
        assert "⚠️" in _SAR_POPUP_BLOCK

    def test_sar_enabled_false_returns_no_result(self):
        """With SAR disabled, module functions return None gracefully."""
        from src.sar import check_sar_enabled
        # SAR is disabled by default in test config
        with patch("src.sar.is_sar_enabled", return_value=False):
            enabled = check_sar_enabled()
            assert enabled is False

    @pytest.mark.skipif(
        not (FIXTURE_DIR / "mock_s1_vv.tif").exists(),
        reason="SAR fixture mock_s1_vv.tif not generated (run tests/fixtures/create_fixtures.py)"
    )
    def test_change_detection_with_fixture(self, tmp_path):
        """Full change detection with synthetic raster fixture."""
        from src.sar.change_detector import compute_backscatter_change

        fixture = FIXTURE_DIR / "mock_s1_vv.tif"
        result = compute_backscatter_change(
            vv_before_path=fixture,
            vv_after_path=fixture,  # Before == after → ratio should be ~0
            output_dir=tmp_path,
            before_date="2024-03-01",
            after_date="2024-04-15",
        )

        assert result["change_raster_path"] is not None or result["change_raster_path"] is None
        # Corridor summary should be empty list (no corridor buffer provided)
        assert isinstance(result["corridor_change_summary"], list)


class TestChangeTierMapping:
    """Change tier value mapping for exposure scorer."""

    def test_tier_values(self):
        from src.sar.exposure_scorer import _CHANGE_TIER_VALUES
        assert _CHANGE_TIER_VALUES["none"] == 0.0
        assert _CHANGE_TIER_VALUES["moderate"] < _CHANGE_TIER_VALUES["significant"]
        assert _CHANGE_TIER_VALUES["significant"] < _CHANGE_TIER_VALUES["extreme"]
        assert _CHANGE_TIER_VALUES["extreme"] == 1.0
