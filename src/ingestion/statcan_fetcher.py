"""Statistics Canada data fetcher — Sources 9 & 10.

Source 9: Dissemination Block (DB) boundaries — finest geographic unit
  https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/
Source 10: Building-level parcel data (supplemented by Microsoft buildings)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# 2021 Census Dissemination Block boundary file (Ontario — cartographic)
_STATCAN_DB_URL = (
    "https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/"
    "boundary-limites/files-fichiers/ldb_000b21a_e.zip"
)

# Province code for Ontario = 35
_ONTARIO_PROVINCE_CODE = "35"


class StatCanFetcher:
    """Fetch Statistics Canada dissemination block and census data."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "statcan",
            max_age_days=cfg["cache"]["max_age_days"],
        )

    def fetch_dissemination_blocks(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        force: bool = False,
    ) -> gpd.GeoDataFrame:
        """Fetch Ontario Dissemination Block boundaries.

        Parameters
        ----------
        bbox: Optional spatial filter (west, south, east, north) in EPSG:4326
        force: Re-download if True

        Returns
        -------
        GeoDataFrame with columns: dbuid, pr_uid, cd_uid, csd_uid,
          pop_2021, dwell_2021, geometry (EPSG:4326)
        """
        cache_key = "statcan_db_ontario"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("StatCan DB: loading from cache")
            gdf = gpd.read_file(cached)
            if bbox:
                gdf = self._clip_to_bbox(gdf, bbox)
            return gdf

        log.info("StatCan DB: downloading dissemination blocks")
        try:
            resp = requests.get(_STATCAN_DB_URL, timeout=300, stream=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("StatCan DB download failed: %s", exc)
            return self._empty_db_gdf()

        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            shp_files = [n for n in zf.namelist() if n.endswith(".shp")]
            if not shp_files:
                return self._empty_db_gdf()
            tmp_dir = self.cache.cache_dir / "statcan_db_tmp"
            tmp_dir.mkdir(exist_ok=True)
            zf.extractall(tmp_dir)
            shp_path = tmp_dir / shp_files[0]

        gdf = gpd.read_file(shp_path)
        # Filter to Ontario (province code 35)
        pr_col = [c for c in gdf.columns if "pruid" in c.lower() or c.lower() == "pr_uid"]
        if pr_col:
            gdf = gdf[gdf[pr_col[0]].astype(str).str.startswith(_ONTARIO_PROVINCE_CODE)]

        gdf = gdf.to_crs("EPSG:4326")
        gdf = self._normalise_db_columns(gdf)

        out_path = self.cache.cache_dir / "statcan_db_ontario.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="dissemination_blocks")
        self.cache.put(cache_key, out_path, url=_STATCAN_DB_URL)
        log.info("StatCan DB: cached %d dissemination blocks", len(gdf))

        if bbox:
            gdf = self._clip_to_bbox(gdf, bbox)
        return gdf

    def _normalise_db_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardise column names."""
        col_map = {}
        for col in gdf.columns:
            cl = col.lower()
            if cl in ("dbuid", "db_uid"):
                col_map[col] = "dbuid"
            elif cl in ("pruid", "pr_uid", "provcode"):
                col_map[col] = "pr_uid"
            elif cl in ("cduid", "cd_uid"):
                col_map[col] = "cd_uid"
            elif cl in ("csduid", "csd_uid"):
                col_map[col] = "csd_uid"
            elif "pop" in cl:
                col_map[col] = "pop_2021"
            elif "dwell" in cl or "hh" in cl:
                col_map[col] = "dwell_2021"

        gdf = gdf.rename(columns=col_map)
        for col in ["dbuid", "pr_uid", "cd_uid", "csd_uid", "pop_2021", "dwell_2021"]:
            if col not in gdf.columns:
                gdf[col] = None

        gdf["source"] = "StatCan_2021_DB"
        gdf["confidence"] = 1.0
        return gdf

    def _clip_to_bbox(
        self, gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        from ..utils.geometry import bbox_to_polygon
        clip_poly = bbox_to_polygon(*bbox)
        return gdf[gdf.geometry.intersects(clip_poly)].copy()

    def _empty_db_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["dbuid", "pr_uid", "cd_uid", "csd_uid", "pop_2021",
                     "dwell_2021", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    # ------------------------------------------------------------------
    # Centroid-based building proxy (for service point density)
    # ------------------------------------------------------------------

    def db_centroids_as_service_proxies(
        self, db_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """Derive service-point proxy locations from DB centroids weighted by dwellings."""
        centroids = db_gdf.copy()
        centroids["geometry"] = db_gdf.geometry.centroid
        centroids["proxy_type"] = "db_centroid"
        centroids["estimated_dwellings"] = pd.to_numeric(
            centroids.get("dwell_2021", 1), errors="coerce"
        ).fillna(1)
        return centroids
