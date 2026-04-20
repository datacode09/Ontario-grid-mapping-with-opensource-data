"""Coverage reporter — generate data coverage and SAR audit reports."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)


class CoverageReporter:
    """Generate data coverage and SAR audit reports."""

    def __init__(self, region_name: str = "Mississauga") -> None:
        self._region = region_name

    def generate_coverage_report(
        self,
        layers: dict[str, gpd.GeoDataFrame],
        sar_manifest: Optional[dict] = None,
        output_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Generate a coverage report CSV for all data layers.

        Parameters
        ----------
        layers:        Dict of layer_name → GeoDataFrame.
        sar_manifest:  SAR export manifest dict (from SARExporter.export_all).
        output_path:   CSV output path.

        Returns
        -------
        DataFrame with per-layer coverage statistics.
        """
        records = []

        for layer_name, gdf in layers.items():
            if gdf is None:
                records.append(self._empty_record(layer_name, "no_data"))
                continue

            total = len(gdf)
            valid_geom = int(gdf.geometry.is_valid.sum()) if total else 0
            with_confidence = int((gdf.get("confidence", pd.Series(dtype=float)) > 0).sum())
            sources = gdf.get("source", pd.Series(dtype=str)).unique().tolist() if total else []
            avg_confidence = float(gdf.get("confidence", pd.Series(dtype=float)).mean()) if total else 0.0

            records.append({
                "region":          self._region,
                "layer_name":      layer_name,
                "total_features":  total,
                "valid_geometries": valid_geom,
                "with_confidence":  with_confidence,
                "avg_confidence":   round(avg_confidence, 3),
                "sources":          "; ".join(str(s) for s in sources if s),
                "crs":              str(gdf.crs) if gdf.crs else "unknown",
                "status":           "ok" if total > 0 else "empty",
            })

        # Add SAR products to report
        if sar_manifest:
            for product, path in sar_manifest.items():
                if product in ("output_dir", "disclaimer"):
                    continue
                records.append({
                    "region":          self._region,
                    "layer_name":      f"SAR_{product}",
                    "total_features":  1,
                    "valid_geometries": 1,
                    "with_confidence":  0,
                    "avg_confidence":   0.0,
                    "sources":          "SAR_contextual_RS",
                    "crs":             "EPSG:32617 (SAR/COG)",
                    "status":          "sar_product",
                })

        df = pd.DataFrame(records)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            log.info("CoverageReporter: wrote report to %s", output_path)

        return df

    @staticmethod
    def _empty_record(layer_name: str, status: str) -> dict:
        return {
            "region": "", "layer_name": layer_name,
            "total_features": 0, "valid_geometries": 0,
            "with_confidence": 0, "avg_confidence": 0.0,
            "sources": "", "crs": "unknown", "status": status,
        }

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print a human-readable summary of coverage results."""
        log.info("=" * 60)
        log.info("Coverage Report — %s", self._region)
        log.info("=" * 60)
        for _, row in df.iterrows():
            log.info(
                "  %-35s %5d features | confidence: %.2f | %s",
                row["layer_name"],
                row["total_features"],
                row["avg_confidence"],
                row["status"],
            )
        log.info("=" * 60)
