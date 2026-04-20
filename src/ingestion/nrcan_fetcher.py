"""NRCan CanVec power layer fetcher — Source 15.

National-scale power infrastructure layer (coarser than OSM but useful
for rural Ontario HV line geometry validation).
https://open.canada.ca/data/en/dataset/8ba2aa2a-7bb9-4448-b4d7-f164409fe056
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import requests

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# CanVec feature catalogue: EN_1480009_0 = Electric Power Transmission Line
# NRCan publishes tiled GeoPackage for each NTS sheet; use provincial aggregate.
_NRCAN_CANVEC_BASE = (
    "https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/canvec/"
    "shp/EnMaElEc/"
)
_NRCAN_CANVEC_ON_URL = (
    "https://ftp.maps.canada.ca/pub/nrcan_rncan/vector/canvec/"
    "shp/EnMaElEc/canvec_ON_EnMaElEc.zip"
)


class NRCanFetcher:
    """Fetch NRCan CanVec electricity infrastructure for Ontario."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "nrcan",
            max_age_days=cfg["cache"]["max_age_days"],
        )

    def fetch_canvec_power(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        force: bool = False,
    ) -> gpd.GeoDataFrame:
        """Fetch NRCan CanVec power layer for Ontario.

        Returns GeoDataFrame with columns:
          facility_type, voltage_kv, operator, geometry, source, confidence
        """
        cache_key = "nrcan_canvec_power_on"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("NRCan CanVec: loading from cache")
            gdf = gpd.read_file(cached)
            if bbox:
                from ..utils.geometry import bbox_to_polygon
                poly = bbox_to_polygon(*bbox)
                gdf = gdf[gdf.geometry.intersects(poly)]
            return gdf

        log.info("NRCan CanVec: downloading Ontario power layer")
        try:
            resp = requests.get(_NRCAN_CANVEC_ON_URL, timeout=300, stream=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("NRCan CanVec download failed: %s", exc)
            return self._empty_gdf()

        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            # CanVec uses feature codes; 1480009 = electric transmission line
            shp_files = [
                n for n in zf.namelist()
                if n.endswith(".shp") and ("1480009" in n or "power" in n.lower())
            ]
            if not shp_files:
                # Fall back to any shapefile in the archive
                shp_files = [n for n in zf.namelist() if n.endswith(".shp")]

            if not shp_files:
                log.warning("NRCan CanVec: no shapefiles found in archive")
                return self._empty_gdf()

            tmp_dir = self.cache.cache_dir / "nrcan_canvec_tmp"
            tmp_dir.mkdir(exist_ok=True)
            zf.extractall(tmp_dir)

        gdfs = []
        for shp_name in shp_files:
            shp_path = tmp_dir / shp_name
            if shp_path.exists():
                try:
                    gdfs.append(gpd.read_file(shp_path).to_crs("EPSG:4326"))
                except Exception as exc:
                    log.warning("NRCan: could not read %s: %s", shp_path, exc)

        if not gdfs:
            return self._empty_gdf()

        import pandas as pd
        gdf = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
        )
        gdf = self._normalise_columns(gdf)

        out_path = self.cache.cache_dir / "nrcan_canvec_power_on.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="power")
        self.cache.put(cache_key, out_path, url=_NRCAN_CANVEC_ON_URL)
        log.info("NRCan CanVec: cached %d features", len(gdf))

        if bbox:
            from ..utils.geometry import bbox_to_polygon
            poly = bbox_to_polygon(*bbox)
            gdf = gdf[gdf.geometry.intersects(poly)]
        return gdf

    def _normalise_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        rename = {}
        for col in gdf.columns:
            cl = col.lower()
            if "voltage" in cl:
                rename[col] = "voltage_kv"
            elif "type" in cl or "class" in cl:
                rename[col] = "facility_type"
            elif "operator" in cl or "owner" in cl:
                rename[col] = "operator"

        gdf = gdf.rename(columns=rename)
        gdf["source"] = "NRCan_CanVec"
        gdf["confidence"] = 0.75  # Coarser than OSM; treat as inferred
        return gdf

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["facility_type", "voltage_kv", "operator", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
