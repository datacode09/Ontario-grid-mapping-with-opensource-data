"""Voltage classification — assign tier and normalised voltage to each feature."""
from __future__ import annotations

import re
from typing import Optional

import geopandas as gpd
import pandas as pd

from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# Ontario voltage tiers
VOLTAGE_TIERS = {
    "hv_500":      (450_000, 550_000),   # 500 kV transmission
    "hv_230":      (200_000, 250_000),   # 230 kV transmission
    "hv_115":      (100_000, 120_000),   # 115 kV transmission
    "mv_27k":      (25_000,  30_000),    # 27.6 kV zone substation secondary
    "mv_25k":      (22_000,  26_000),    # 25 kV primary distribution
    "mv_14k":      (12_000,  16_000),    # 14.4 / 13.8 kV primary distribution
    "mv_8k":       (7_000,   9_000),     # 8.0 kV primary distribution
    "mv_4k":       (3_500,   5_000),     # 4.16 kV legacy (Toronto/Ottawa)
    "mv_2k":       (2_000,   3_000),     # 2.4 kV legacy
    "lv":          (100,     500),       # 120/240 V secondary
}


class VoltageClassifier:
    """Parse and classify voltage tags on OSM/IESO features."""

    def classify(self, gdf: gpd.GeoDataFrame, voltage_col: str = "voltage") -> gpd.GeoDataFrame:
        """Add columns: voltage_v (int), voltage_kv (float), voltage_tier (str).

        Handles OSM multi-value voltages like "115000;230000" (takes max).
        """
        gdf = gdf.copy()
        gdf["voltage_v"] = gdf.get(voltage_col, pd.Series(dtype=object)).apply(
            self._parse_voltage_v
        )
        gdf["voltage_kv"] = (gdf["voltage_v"] / 1000.0).round(1)
        gdf["voltage_tier"] = gdf["voltage_v"].apply(self._assign_tier)
        return gdf

    def classify_from_context(
        self, gdf: gpd.GeoDataFrame, ldc_gdf: Optional[gpd.GeoDataFrame] = None
    ) -> gpd.GeoDataFrame:
        """Infer voltage tier from spatial context when voltage tag is absent."""
        gdf = gdf.copy()
        # If voltage_tier not yet assigned, use geometry type as heuristic
        unclassified = gdf["voltage_tier"] == "unknown" if "voltage_tier" in gdf.columns else slice(None)

        if "power" in gdf.columns:
            gdf.loc[gdf["power"] == "tower", "voltage_tier"] = gdf.loc[
                gdf["power"] == "tower", "voltage_tier"
            ].replace("unknown", "hv_115")

            gdf.loc[gdf["power"] == "pole", "voltage_tier"] = gdf.loc[
                gdf["power"] == "pole", "voltage_tier"
            ].replace("unknown", "mv_25k")

        return gdf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_voltage_v(raw) -> int:
        """Parse OSM voltage string to integer volts (max of multi-values)."""
        if pd.isna(raw) or raw == "":
            return 0
        raw = str(raw).strip()
        # Multi-value: "115000;230000" → take max
        parts = re.split(r"[;|,]", raw)
        values = []
        for part in parts:
            part = part.strip()
            # Handle "115 kV" style
            m = re.match(r"([\d.]+)\s*(kv|kilo)?", part, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if m.group(2) and m.group(2).lower().startswith("k"):
                    val *= 1000
                elif val < 1000:
                    val *= 1000  # Assume kV if small number
                values.append(int(val))
        return max(values) if values else 0

    @staticmethod
    def _assign_tier(voltage_v: int) -> str:
        """Assign a tier label based on voltage in volts."""
        for tier, (lo, hi) in VOLTAGE_TIERS.items():
            if lo <= voltage_v <= hi:
                return tier
        if voltage_v > 0:
            log.debug("Unclassified voltage: %d V", voltage_v)
        return "unknown"

    @staticmethod
    def voltage_label(tier: str) -> str:
        """Human-readable voltage label for display."""
        labels = {
            "hv_500": "500 kV",
            "hv_230": "230 kV",
            "hv_115": "115 kV",
            "mv_27k": "27.6 kV",
            "mv_25k": "25 kV",
            "mv_14k": "13.8–14.4 kV",
            "mv_8k": "8.0 kV",
            "mv_4k": "4.16 kV",
            "mv_2k": "2.4 kV",
            "lv": "120/240 V",
            "unknown": "Unknown",
        }
        return labels.get(tier, tier)
