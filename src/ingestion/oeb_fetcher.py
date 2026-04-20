"""OEB data fetcher — Sources 1 & 2.

Source 1: OEB Distributor Service Areas (GIS shapefiles)
  https://www.oeb.ca/open-data/electricity-and-natural-gas-distributors-service-areas
Source 2: OEB Electricity RRR (CSV) — customer counts, SAIDI/SAIFI
  https://www.oeb.ca/open-data
"""
from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# OEB open-data portal endpoints (subject to OEB website updates)
_OEB_SERVICE_AREAS_URL = (
    "https://www.oeb.ca/sites/default/files/OEB_Electricity_Distributors_"
    "Service_Area.zip"
)
_OEB_RRR_BASE_URL = "https://www.oeb.ca/open-data"


class OEBFetcher:
    """Fetch and cache OEB distributor service areas and RRR data."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "oeb",
            max_age_days=cfg["cache"]["max_age_days"],
        )

    # ------------------------------------------------------------------
    # Source 1: Service Areas
    # ------------------------------------------------------------------

    def fetch_service_areas(self, force: bool = False) -> gpd.GeoDataFrame:
        """Download OEB electricity distributor service area polygons.

        Returns a GeoDataFrame in EPSG:4326 with columns:
          ldc_name, ldc_id, geometry
        """
        cache_key = "oeb_service_areas"
        cached = self.cache.get(cache_key)

        if cached and not force:
            log.info("OEB service areas: loading from cache %s", cached)
            return gpd.read_file(cached)

        log.info("OEB service areas: downloading from %s", _OEB_SERVICE_AREAS_URL)
        try:
            resp = requests.get(_OEB_SERVICE_AREAS_URL, timeout=120, stream=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("OEB service areas download failed: %s", exc)
            return self._fallback_service_areas()

        zip_bytes = io.BytesIO(resp.content)
        with zipfile.ZipFile(zip_bytes) as zf:
            shp_files = [n for n in zf.namelist() if n.endswith(".shp")]
            if not shp_files:
                log.error("No .shp in OEB service areas ZIP")
                return self._fallback_service_areas()

            tmp_dir = self.cache.cache_dir / "oeb_service_areas_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            zf.extractall(tmp_dir)
            shp_path = tmp_dir / shp_files[0]

        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        gdf = self._normalise_service_area_columns(gdf)

        out_path = self.cache.cache_dir / "oeb_service_areas.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="service_areas")
        self.cache.put(cache_key, out_path, url=_OEB_SERVICE_AREAS_URL)
        log.info("OEB service areas: cached %d LDC polygons", len(gdf))
        return gdf

    def _normalise_service_area_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardise column names across OEB shapefile versions."""
        col_map = {}
        for col in gdf.columns:
            cl = col.lower()
            if "name" in cl and "ldc" in cl:
                col_map[col] = "ldc_name"
            elif cl in ("distributor", "utility", "company"):
                col_map[col] = "ldc_name"
            elif "id" in cl and "ldc" in cl:
                col_map[col] = "ldc_id"

        gdf = gdf.rename(columns=col_map)
        if "ldc_name" not in gdf.columns:
            # Best-effort: use first non-geometry string column
            str_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == object]
            if str_cols:
                gdf["ldc_name"] = gdf[str_cols[0]]
            else:
                gdf["ldc_name"] = "Unknown"

        if "ldc_id" not in gdf.columns:
            gdf["ldc_id"] = range(len(gdf))

        gdf["source"] = "OEB_authoritative"
        gdf["confidence"] = 1.0
        return gdf[["ldc_name", "ldc_id", "source", "confidence", "geometry"]]

    def _fallback_service_areas(self) -> gpd.GeoDataFrame:
        """Return empty GDF with correct schema when download fails."""
        log.warning("OEB service areas: returning empty fallback GeoDataFrame")
        return gpd.GeoDataFrame(
            columns=["ldc_name", "ldc_id", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    # ------------------------------------------------------------------
    # Source 2: RRR — reliability metrics
    # ------------------------------------------------------------------

    def fetch_rrr_reliability(
        self, year: int = 2023, force: bool = False
    ) -> pd.DataFrame:
        """Fetch OEB RRR SAIDI/SAIFI data for a given reporting year.

        Returns a DataFrame with columns:
          ldc_name, year, saidi_minutes, saifi_interruptions,
          customers_residential, customers_commercial, customers_industrial
        """
        cache_key = f"oeb_rrr_{year}"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("OEB RRR %d: loading from cache", year)
            return pd.read_csv(cached)

        log.info("OEB RRR: fetching year %d reliability data", year)
        # OEB publishes RRR as Excel or CSV on their open data portal.
        # The URL pattern changes annually — we construct a best-effort URL.
        rrr_url = (
            f"https://www.oeb.ca/sites/default/files/OEB_RRR_Reliability_{year}.csv"
        )
        try:
            resp = requests.get(rrr_url, timeout=60)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            log.warning("OEB RRR download failed (%s): returning synthetic schema", exc)
            return self._synthetic_rrr_schema(year)

        df = self._normalise_rrr_columns(df, year)
        out_path = self.cache.cache_dir / f"oeb_rrr_{year}.csv"
        df.to_csv(out_path, index=False)
        self.cache.put(cache_key, out_path, url=rrr_url)
        log.info("OEB RRR %d: cached %d LDC records", year, len(df))
        return df

    def _normalise_rrr_columns(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename = {
            "distributor": "ldc_name",
            "utility": "ldc_name",
            "company_name": "ldc_name",
            "saidi": "saidi_minutes",
            "saifi": "saifi_interruptions",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["year"] = year
        df["source"] = "OEB_RRR"
        return df

    def _synthetic_rrr_schema(self, year: int) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "ldc_name", "year", "saidi_minutes", "saifi_interruptions",
                "customers_residential", "customers_commercial",
                "customers_industrial", "source",
            ]
        )
