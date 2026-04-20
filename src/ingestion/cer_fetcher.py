"""Canada Energy Regulator (CER) data fetcher — Source 5.

Interprovincial transmission tie lines (ON↔QC, ON↔MB, ON↔US).
https://www.cer-rec.gc.ca/en/data-analysis/energy-commodities/electricity/index.html
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

_CER_ELECTRICITY_URL = (
    "https://www.cer-rec.gc.ca/open/energy/electricity/"
    "national-energy-board-interprovincial-and-international-electricity-"
    "transmission-lines.zip"
)


class CERFetcher:
    """Fetch CER interprovincial transmission line data."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "cer",
            max_age_days=cfg["cache"]["max_age_days"],
        )

    def fetch_interprovincial_lines(self, force: bool = False) -> gpd.GeoDataFrame:
        """Fetch CER interprovincial transmission line shapefile.

        Returns GeoDataFrame filtered to Ontario interconnections.
        """
        cache_key = "cer_interprovincial_lines"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("CER: loading from cache")
            return gpd.read_file(cached)

        log.info("CER: downloading interprovincial transmission lines")
        try:
            resp = requests.get(_CER_ELECTRICITY_URL, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("CER download failed: %s", exc)
            return self._empty_gdf()

        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            shp_files = [n for n in zf.namelist() if n.endswith(".shp")]
            if not shp_files:
                return self._empty_gdf()
            tmp_dir = self.cache.cache_dir / "cer_tmp"
            tmp_dir.mkdir(exist_ok=True)
            zf.extractall(tmp_dir)
            shp_path = tmp_dir / shp_files[0]

        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")

        # Filter to lines touching Ontario
        from ..utils.geometry import bbox_to_polygon
        on_bbox = bbox_to_polygon(-95.2, 41.7, -74.3, 56.9)
        gdf = gdf[gdf.geometry.intersects(on_bbox)].copy()
        gdf["source"] = "CER_Interprovincial"
        gdf["confidence"] = 1.0
        gdf["node_type"] = "tx_line"

        out_path = self.cache.cache_dir / "cer_interprovincial_on.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="tx_lines")
        self.cache.put(cache_key, out_path, url=_CER_ELECTRICITY_URL)
        log.info("CER: cached %d Ontario interprovincial lines", len(gdf))
        return gdf

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["source", "confidence", "node_type", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
