"""Gridfinder runner — predict MV distribution lines for rural Ontario.

Source 18: gridfinder (Arderne et al. 2020)
Paper: https://doi.org/10.1038/s41597-019-0347-4
Repo:  https://github.com/carderne/gridfinder

Method: Night-time light imagery + road network → predicts MV line routing.
Used for rural Ontario where OSM has no distribution geometry.

Confidence: 0.30 (gridfinder_rural per settings.yaml)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np

from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

try:
    import gridfinder
    _GRIDFINDER_AVAILABLE = True
except ImportError:
    _GRIDFINDER_AVAILABLE = False
    log.info("gridfinder package not installed — rural inference will be skipped.")


class GridfinderRunner:
    """Run gridfinder MV line prediction for rural Ontario zones.

    When gridfinder is not installed, the runner gracefully returns an
    empty GeoDataFrame and logs an informational message.
    """

    def __init__(self) -> None:
        cfg = load_settings()
        gf_cfg = cfg["ml_inference"]["gridfinder"]
        self._threshold = gf_cfg["prediction_threshold"]

    def predict(
        self,
        bbox: Tuple[float, float, float, float],
        night_lights_path: Optional[Path],
        roads_gdf: gpd.GeoDataFrame,
        output_path: Optional[Path] = None,
    ) -> gpd.GeoDataFrame:
        """Run gridfinder for a rural Ontario bounding box.

        Parameters
        ----------
        bbox:               (west, south, east, north) in EPSG:4326.
        night_lights_path:  Path to NASA VNP46A1 GeoTIFF (may be None).
        roads_gdf:          Road network as LineStrings.
        output_path:        Optional path to write predicted lines.

        Returns
        -------
        GeoDataFrame of predicted MV distribution lines.
        Confidence = 0.30 (gridfinder_rural).
        """
        if not _GRIDFINDER_AVAILABLE:
            log.info("GridfinderRunner: package not available — returning empty result")
            return self._empty_gdf()

        if night_lights_path is None or not Path(night_lights_path).exists():
            log.warning("GridfinderRunner: night-lights raster not available — "
                        "using road-only heuristic")
            return self._road_heuristic(roads_gdf)

        log.info("GridfinderRunner: predicting MV lines for bbox %s", bbox)
        try:
            return self._run_gridfinder(bbox, night_lights_path, roads_gdf, output_path)
        except Exception as exc:
            log.error("GridfinderRunner: prediction failed: %s", exc)
            return self._empty_gdf()

    # ------------------------------------------------------------------
    # gridfinder integration
    # ------------------------------------------------------------------

    def _run_gridfinder(
        self,
        bbox: Tuple[float, float, float, float],
        night_lights_path: Path,
        roads_gdf: gpd.GeoDataFrame,
        output_path: Optional[Path],
    ) -> gpd.GeoDataFrame:
        """Integrate with gridfinder library."""
        import rasterio
        from rasterio.mask import mask as raster_mask
        from ..utils.geometry import bbox_to_polygon

        clip_poly = bbox_to_polygon(*bbox)

        # Load and clip night-lights raster
        with rasterio.open(night_lights_path) as src:
            try:
                clipped, transform = raster_mask(
                    src, [clip_poly], crop=True, nodata=0
                )
            except Exception:
                clipped = src.read(1)
                transform = src.transform

        night_lights_arr = clipped[0] if clipped.ndim == 3 else clipped

        # Convert roads to gridfinder-compatible format
        try:
            # gridfinder expects a rasterised road grid
            roads_arr = self._rasterise_roads(roads_gdf, night_lights_arr.shape, transform)
            predicted = gridfinder.run_model(
                ntl=night_lights_arr,
                roads=roads_arr,
                threshold=self._threshold,
            )
            return self._predicted_to_gdf(predicted, transform, bbox)
        except Exception as exc:
            log.warning("gridfinder model call failed: %s", exc)
            return self._road_heuristic(roads_gdf)

    @staticmethod
    def _rasterise_roads(
        roads: gpd.GeoDataFrame,
        shape: Tuple[int, int],
        transform,
    ) -> np.ndarray:
        """Rasterise road network to a binary grid matching night-lights dimensions."""
        from rasterio.features import rasterize
        from shapely.geometry import mapping

        geoms = [(mapping(geom), 1) for geom in roads.geometry if geom]
        if not geoms:
            return np.zeros(shape, dtype=np.uint8)

        return rasterize(
            geoms,
            out_shape=shape,
            transform=transform,
            dtype=np.uint8,
            fill=0,
        )

    @staticmethod
    def _predicted_to_gdf(predicted_arr, transform, bbox) -> gpd.GeoDataFrame:
        """Convert gridfinder binary prediction raster to vector GeoDataFrame."""
        from rasterio.features import shapes
        from shapely.geometry import shape

        mask = predicted_arr > 0
        geoms = [
            shape(geom)
            for geom, val in shapes(predicted_arr.astype(np.uint8), mask=mask, transform=transform)
            if val == 1
        ]

        if not geoms:
            return gpd.GeoDataFrame(columns=["geometry", "source", "confidence"],
                                    geometry="geometry", crs="EPSG:4326")

        from shapely.ops import unary_union
        from shapely.geometry import MultiLineString
        combined = unary_union(geoms)

        records = [{
            "geometry":   combined,
            "source":     "gridfinder_prediction",
            "confidence": 0.30,
            "edge_type":  "primary_feeder",
            "inferred":   True,
        }]
        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    def _road_heuristic(self, roads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Fallback: treat all roads as potential distribution routes."""
        if len(roads_gdf) == 0:
            return self._empty_gdf()

        result = roads_gdf[["geometry"]].copy()
        result["source"] = "road_heuristic_fallback"
        result["confidence"] = 0.20
        result["edge_type"] = "primary_feeder"
        result["inferred"] = True
        return result

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["geometry", "source", "confidence", "edge_type", "inferred"],
            geometry="geometry",
            crs="EPSG:4326",
        )
