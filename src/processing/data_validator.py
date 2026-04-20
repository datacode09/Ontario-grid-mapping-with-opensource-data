"""Data validator — integrity checks before topology construction."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd

from ..utils.logger import get_logger

log = get_logger(__name__)


class ValidationResult:
    """Container for validation outcomes."""

    def __init__(self, layer: str) -> None:
        self.layer = layer
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.record_count: int = 0
        self.valid_geometry_count: int = 0

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        return (
            f"ValidationResult(layer={self.layer!r}, "
            f"records={self.record_count}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )


class DataValidator:
    """Validate GeoDataFrames before graph construction.

    Checks:
    - Required columns present
    - No null geometries
    - All geometries valid (is_valid)
    - CRS is set
    - confidence values in [0, 1]
    - No duplicate OSM IDs (if present)
    """

    # Minimum required columns per layer type
    _REQUIRED_COLS = {
        "substations": ["source", "confidence"],
        "lines":       ["source", "confidence"],
        "generators":  ["source", "confidence"],
        "buildings":   ["source"],
    }

    def validate_layer(
        self,
        gdf: gpd.GeoDataFrame,
        layer_type: str = "generic",
        strict: bool = False,
    ) -> ValidationResult:
        """Run all validation checks on a GeoDataFrame."""
        result = ValidationResult(layer=layer_type)
        result.record_count = len(gdf)

        if gdf is None or len(gdf) == 0:
            result.warnings.append("Empty GeoDataFrame")
            return result

        # CRS check
        if gdf.crs is None:
            result.errors.append("No CRS defined")
        elif gdf.crs.to_epsg() != 4326:
            result.warnings.append(f"CRS is {gdf.crs.to_epsg()}, expected 4326")

        # Geometry nulls
        null_geom = gdf.geometry.isna().sum()
        if null_geom:
            msg = f"{null_geom} null geometries"
            (result.errors if strict else result.warnings).append(msg)

        # Geometry validity
        invalid_geom = (~gdf.geometry.is_valid).sum()
        if invalid_geom:
            result.warnings.append(f"{invalid_geom} invalid geometries (use buffer(0) to fix)")
        result.valid_geometry_count = int((~gdf.geometry.isna() & gdf.geometry.is_valid).sum())

        # Required columns
        required = self._REQUIRED_COLS.get(layer_type, [])
        for col in required:
            if col not in gdf.columns:
                result.errors.append(f"Missing required column: '{col}'")

        # Confidence range
        if "confidence" in gdf.columns:
            out_of_range = (
                (gdf["confidence"] < 0) | (gdf["confidence"] > 1)
            ).sum()
            if out_of_range:
                result.warnings.append(
                    f"{out_of_range} records with confidence outside [0,1]"
                )

        # Duplicate OSM IDs
        if "osm_id" in gdf.columns:
            dupes = gdf["osm_id"].duplicated().sum()
            if dupes:
                result.warnings.append(f"{dupes} duplicate osm_id values")

        # Log summary
        if result.errors:
            log.error("Validation FAILED [%s]: %s", layer_type, "; ".join(result.errors))
        elif result.warnings:
            log.warning("Validation warnings [%s]: %s", layer_type, "; ".join(result.warnings))
        else:
            log.info(
                "Validation OK [%s]: %d records, %d valid geometries",
                layer_type,
                result.record_count,
                result.valid_geometry_count,
            )

        return result

    def validate_all(
        self, layers: dict[str, gpd.GeoDataFrame], strict: bool = False
    ) -> dict[str, ValidationResult]:
        return {
            name: self.validate_layer(gdf, layer_type=name, strict=strict)
            for name, gdf in layers.items()
        }

    def fix_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply buffer(0) to repair invalid geometries in-place."""
        mask = ~gdf.geometry.is_valid
        if mask.any():
            gdf = gdf.copy()
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].buffer(0)
        return gdf
