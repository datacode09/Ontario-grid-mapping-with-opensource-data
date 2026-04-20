"""Operator tagger — assign LDC/utility operator to each grid asset."""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd

from ..utils.config_loader import load_settings
from ..utils.geometry import to_lambert
from ..utils.logger import get_logger

log = get_logger(__name__)

# Known Ontario LDC names (OEB-licensed distributors)
_KNOWN_LDCS = {
    "hydro one": "Hydro One",
    "toronto hydro": "Toronto Hydro",
    "alectra": "Alectra",
    "alectra utilities": "Alectra",
    "powerstream": "Alectra",  # PowerStream merged into Alectra
    "hydro ottawa": "Hydro Ottawa",
    "kingston hydro": "Kingston Hydro",
    "union gas": None,         # Gas only — exclude
    "enbridge gas": None,      # Gas only — exclude
    "oakville hydro": "Alectra",  # Merged into Alectra
    "burlington hydro": "Alectra",
    "horizon utilities": "Alectra",
    "essx power": "Hydro One",
}


class OperatorTagger:
    """Tag each grid asset with the responsible LDC operator.

    Uses OEB service-area boundary spatial join as the authoritative source.
    Falls back to OSM operator= tag or geometry-based heuristics.
    """

    def __init__(self, ldc_boundaries: Optional[gpd.GeoDataFrame] = None) -> None:
        self._ldc_boundaries = ldc_boundaries
        cfg = load_settings()
        self._default_rural = cfg["operators"]["default_rural"]

    def tag(
        self,
        gdf: gpd.GeoDataFrame,
        ldc_boundaries: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        """Assign operator column to each feature in gdf.

        Priority:
          1. Spatial join with OEB LDC boundaries (if provided)
          2. OSM operator= tag (cleaned)
          3. Default rural operator (Hydro One)
        """
        boundaries = ldc_boundaries or self._ldc_boundaries
        gdf = gdf.copy()
        gdf["operator"] = gdf.get("operator", pd.Series(dtype=object))

        if boundaries is not None and len(boundaries):
            gdf = self._spatial_join_operator(gdf, boundaries)
        else:
            log.debug("OperatorTagger: no LDC boundaries — using OSM tags + default")

        # Clean up OSM operator tags where spatial join didn't resolve
        mask = gdf["operator"].isna() | (gdf["operator"] == "")
        if "operator" in gdf.columns:
            gdf.loc[mask, "operator"] = gdf.loc[mask, "operator"].apply(
                self._normalise_operator
            )

        # Fill remaining unknowns with rural default
        still_empty = gdf["operator"].isna() | (gdf["operator"] == "")
        gdf.loc[still_empty, "operator"] = self._default_rural

        return gdf

    def _spatial_join_operator(
        self, gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Spatial join to assign LDC name from boundary polygon."""
        # Use centroids for lines / polygons to avoid many-to-many joins
        centroids = gdf.copy()
        centroids["geometry"] = gdf.geometry.centroid

        boundaries_4326 = boundaries.to_crs("EPSG:4326") if boundaries.crs else boundaries
        centroids = centroids.to_crs("EPSG:4326")

        joined = gpd.sjoin(
            centroids[["geometry"]],
            boundaries_4326[["ldc_name", "geometry"]],
            how="left",
            predicate="within",
        )

        gdf["operator_from_boundary"] = joined["ldc_name"].values
        # Fill operator from boundary where not already set
        mask_from_boundary = ~joined["ldc_name"].isna()
        gdf.loc[mask_from_boundary.values, "operator"] = joined.loc[
            mask_from_boundary, "ldc_name"
        ].values

        return gdf

    @staticmethod
    def _normalise_operator(raw) -> str:
        if pd.isna(raw) or raw == "":
            return ""
        normalised = _KNOWN_LDCS.get(str(raw).lower().strip())
        if normalised is None and normalised is not False:
            return str(raw)  # Unknown but present — keep as-is
        return normalised or ""  # Gas/non-electric → empty
