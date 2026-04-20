"""SAR exporter — serialize all SAR products for map layer consumption.

Outputs:
  - corridor_footprints.geojson
  - change_mask_<date>.tif (COG)
  - water_extent_<date>.tif (COG)
  - resilience_risk_index.geojson

All outputs tagged:
  source="SAR_contextual_RS"
  is_exact_asset_geometry=false

NEVER mix SAR contextual confidence with grid topology confidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from ..sar import _SAR_DISCLAIMER
from ..utils.logger import get_logger

log = get_logger(__name__)


class SARExporter:
    """Export SAR products as GeoJSON and COG GeoTIFF for visualization."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("data/processed/sar_derived")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_corridor_footprints(
        self,
        corridors_gdf: gpd.GeoDataFrame,
        filename: str = "corridor_footprints.geojson",
    ) -> Path:
        """Export corridor polygon layer as GeoJSON."""
        self._ensure_sar_tags(corridors_gdf)
        out_path = self.output_dir / filename
        corridors_gdf.to_file(out_path, driver="GeoJSON")
        log.info("SARExporter: wrote %d corridors to %s", len(corridors_gdf), out_path)
        return out_path

    def export_resilience_risk_index(
        self,
        scored_gdf: gpd.GeoDataFrame,
        filename: str = "resilience_risk_index.geojson",
    ) -> Path:
        """Export per-asset resilience risk index as GeoJSON point layer."""
        self._ensure_sar_tags(scored_gdf)

        # Select relevant columns for output
        keep_cols = [
            "asset_id" if "asset_id" in scored_gdf.columns else (
                scored_gdf.columns[0] if len(scored_gdf.columns) else "id"
            ),
            "sar_flood_exposure", "sar_change_tier", "elevation_m",
            "in_official_floodzone", "saidi_percentile",
            "resilience_risk_score", "risk_label",
            "source", "is_exact_asset_geometry", "sar_confidence",
            "disclaimer_text", "geometry",
        ]
        available = [c for c in keep_cols if c in scored_gdf.columns]
        out_gdf = scored_gdf[available].copy()

        # Ensure point geometry for risk index
        if not all(out_gdf.geometry.geom_type == "Point"):
            out_gdf["geometry"] = out_gdf.geometry.centroid

        out_path = self.output_dir / filename
        out_gdf.to_file(out_path, driver="GeoJSON")
        log.info(
            "SARExporter: wrote %d risk records to %s",
            len(out_gdf), out_path
        )
        return out_path

    def export_change_mask_to_cog(
        self,
        change_mask_path: Path,
        output_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Validate and copy a change mask to the export directory as COG."""
        if not Path(change_mask_path).exists():
            log.warning("SARExporter: change mask not found at %s", change_mask_path)
            return None

        if output_filename is None:
            output_filename = Path(change_mask_path).name

        out_path = self.output_dir / output_filename
        if out_path != Path(change_mask_path):
            import shutil
            shutil.copy2(change_mask_path, out_path)

        log.info("SARExporter: change mask at %s", out_path)
        return out_path

    def export_water_extent_to_cog(
        self,
        water_extent_path: Path,
        output_filename: Optional[str] = None,
    ) -> Optional[Path]:
        """Validate and copy a water extent mask to the export directory."""
        if not Path(water_extent_path).exists():
            log.warning("SARExporter: water extent not found at %s", water_extent_path)
            return None

        if output_filename is None:
            output_filename = Path(water_extent_path).name

        out_path = self.output_dir / output_filename
        if out_path != Path(water_extent_path):
            import shutil
            shutil.copy2(water_extent_path, out_path)

        log.info("SARExporter: water extent at %s", out_path)
        return out_path

    def export_all(
        self,
        corridors_gdf: Optional[gpd.GeoDataFrame] = None,
        scored_gdf: Optional[gpd.GeoDataFrame] = None,
        change_mask_path: Optional[Path] = None,
        water_extent_path: Optional[Path] = None,
        before_date: str = "",
        after_date: str = "",
    ) -> dict:
        """Export all SAR products and return a manifest dict."""
        manifest = {"output_dir": str(self.output_dir)}

        if corridors_gdf is not None and len(corridors_gdf):
            manifest["corridor_footprints"] = str(
                self.export_corridor_footprints(corridors_gdf)
            )

        if scored_gdf is not None and len(scored_gdf):
            manifest["resilience_risk_index"] = str(
                self.export_resilience_risk_index(scored_gdf)
            )

        if change_mask_path:
            date_tag = f"{before_date}_{after_date}" if before_date else ""
            fname = f"change_mask_{date_tag}.tif" if date_tag else None
            out = self.export_change_mask_to_cog(change_mask_path, fname)
            if out:
                manifest["change_mask"] = str(out)

        if water_extent_path:
            date_tag = after_date.replace("-", "") if after_date else ""
            fname = f"water_extent_{date_tag}.tif" if date_tag else None
            out = self.export_water_extent_to_cog(water_extent_path, fname)
            if out:
                manifest["water_extent"] = str(out)

        manifest["disclaimer"] = _SAR_DISCLAIMER
        log.info("SARExporter: export complete — %d products", len(manifest) - 2)
        return manifest

    @staticmethod
    def _ensure_sar_tags(gdf: gpd.GeoDataFrame) -> None:
        """Ensure mandatory SAR metadata columns are present."""
        if "source" not in gdf.columns:
            gdf["source"] = "SAR_contextual_RS"
        if "is_exact_asset_geometry" not in gdf.columns:
            gdf["is_exact_asset_geometry"] = False
        if "disclaimer_text" not in gdf.columns:
            gdf["disclaimer_text"] = _SAR_DISCLAIMER
