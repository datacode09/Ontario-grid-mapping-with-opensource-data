"""Municipal open data fetcher — Sources 12, 13, 14.

Source 12: City of Mississauga Open Data (address points, parcels)
  https://www.mississauga.ca/services-and-programs/open-data/
Source 13: City of Toronto Open Data
  https://open.toronto.ca/
Source 14: Ontario Data Catalogue
  https://data.ontario.ca/
"""
from __future__ import annotations

import io
import json
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

# Mississauga open data endpoints (ArcGIS REST API)
_MISSISS_ADDRESSES_URL = (
    "https://opendata.arcgis.com/api/v2/datasets/"
    "88804a0d1a9b414c810eb65808b7ef1f_0/downloads/data"
    "?format=geojson&spatialRefId=4326"
)
# Toronto open data base
_TORONTO_OPENDATA_BASE = "https://ckan0.cf.opendata.inter.prod-toronto.ca"
_TORONTO_ADDRESS_RESOURCE = "address-points-toronto"

# Ontario Data Catalogue (CKAN API)
_ODC_BASE = "https://data.ontario.ca/api/3/action"
_CRITICAL_FACILITIES_DATASET = "critical-infrastructure"


class MunicipalFetcher:
    """Fetch municipal open data for pilot areas (Mississauga, Toronto)."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "municipal",
            max_age_days=cfg["cache"]["max_age_days"],
        )

    # ------------------------------------------------------------------
    # Source 12: Mississauga Address Points
    # ------------------------------------------------------------------

    def fetch_mississauga_addresses(
        self, force: bool = False
    ) -> gpd.GeoDataFrame:
        """Fetch Mississauga address points (authoritative geocoding).

        Returns GeoDataFrame with columns:
          address, postal_code, municipality, geometry, source, confidence
        """
        cache_key = "municipal_mississauga_addresses"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("Mississauga addresses: loading from cache")
            return gpd.read_file(cached)

        log.info("Mississauga addresses: downloading")
        try:
            resp = requests.get(_MISSISS_ADDRESSES_URL, timeout=120)
            resp.raise_for_status()
            gdf = gpd.read_file(io.BytesIO(resp.content))
        except Exception as exc:
            log.warning("Mississauga addresses download failed: %s", exc)
            return self._empty_address_gdf("Mississauga")

        gdf = gdf.to_crs("EPSG:4326")
        gdf = self._normalise_address_columns(gdf, municipality="Mississauga")

        out_path = self.cache.cache_dir / "mississauga_addresses.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="addresses")
        self.cache.put(cache_key, out_path, url=_MISSISS_ADDRESSES_URL)
        log.info("Mississauga addresses: cached %d points", len(gdf))
        return gdf

    # ------------------------------------------------------------------
    # Source 13: Toronto Address Points
    # ------------------------------------------------------------------

    def fetch_toronto_addresses(self, force: bool = False) -> gpd.GeoDataFrame:
        """Fetch Toronto address points via CKAN API."""
        cache_key = "municipal_toronto_addresses"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("Toronto addresses: loading from cache")
            return gpd.read_file(cached)

        log.info("Toronto addresses: fetching via CKAN API")
        try:
            search_url = f"{_TORONTO_OPENDATA_BASE}/api/3/action/package_show"
            resp = requests.get(
                search_url,
                params={"id": _TORONTO_ADDRESS_RESOURCE},
                timeout=30,
            )
            resp.raise_for_status()
            pkg = resp.json()["result"]
            resources = pkg.get("resources", [])
            geojson_resources = [
                r for r in resources
                if r.get("format", "").lower() in ("geojson", "json")
            ]
            if not geojson_resources:
                raise ValueError("No GeoJSON resource found")

            dl_url = geojson_resources[0]["url"]
            resp2 = requests.get(dl_url, timeout=300)
            resp2.raise_for_status()
            gdf = gpd.read_file(io.BytesIO(resp2.content))
        except Exception as exc:
            log.warning("Toronto addresses download failed: %s", exc)
            return self._empty_address_gdf("Toronto")

        gdf = gdf.to_crs("EPSG:4326")
        gdf = self._normalise_address_columns(gdf, municipality="Toronto")

        out_path = self.cache.cache_dir / "toronto_addresses.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="addresses")
        self.cache.put(cache_key, out_path)
        log.info("Toronto addresses: cached %d points", len(gdf))
        return gdf

    # ------------------------------------------------------------------
    # Source 14: Ontario Data Catalogue — Critical Facilities
    # ------------------------------------------------------------------

    def fetch_critical_facilities(self, force: bool = False) -> gpd.GeoDataFrame:
        """Fetch school, hospital and critical facility locations from ODC.

        These are used for load classification and SAR resilience-risk
        prioritisation of critical load zones.
        """
        cache_key = "odc_critical_facilities"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("Critical facilities: loading from cache")
            return gpd.read_file(cached)

        log.info("Critical facilities: fetching from Ontario Data Catalogue")
        datasets = {
            "schools": "https://data.ontario.ca/api/3/action/datastore_search?resource_id=schools",
            "hospitals": "https://data.ontario.ca/api/3/action/datastore_search?resource_id=hospitals",
        }

        gdfs = []
        for facility_type, url in datasets.items():
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                records = resp.json().get("result", {}).get("records", [])
                if records:
                    df = pd.DataFrame(records)
                    if "latitude" in df.columns and "longitude" in df.columns:
                        from shapely.geometry import Point
                        geometry = [
                            Point(float(row["longitude"]), float(row["latitude"]))
                            for _, row in df.iterrows()
                            if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude"))
                        ]
                        if geometry:
                            gdf = gpd.GeoDataFrame(
                                df.head(len(geometry)),
                                geometry=geometry,
                                crs="EPSG:4326",
                            )
                            gdf["facility_type"] = facility_type
                            gdfs.append(gdf)
            except Exception as exc:
                log.warning("Critical facilities %s failed: %s", facility_type, exc)

        if not gdfs:
            return self._empty_facilities_gdf()

        result = gpd.GeoDataFrame(
            pd.concat(gdfs, ignore_index=True), crs="EPSG:4326"
        )
        result["source"] = "Ontario_Data_Catalogue"
        result["confidence"] = 1.0
        result["is_critical_load"] = True

        out_path = self.cache.cache_dir / "critical_facilities.gpkg"
        result.to_file(out_path, driver="GPKG", layer="facilities")
        self.cache.put(cache_key, out_path)
        log.info("Critical facilities: cached %d records", len(result))
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_address_columns(
        self, gdf: gpd.GeoDataFrame, municipality: str
    ) -> gpd.GeoDataFrame:
        col_map = {}
        for col in gdf.columns:
            cl = col.lower()
            if "address" in cl or cl in ("add", "full_addr"):
                col_map[col] = "address"
            elif "postal" in cl or "pcode" in cl:
                col_map[col] = "postal_code"

        gdf = gdf.rename(columns=col_map)
        if "address" not in gdf.columns:
            gdf["address"] = ""
        if "postal_code" not in gdf.columns:
            gdf["postal_code"] = ""

        gdf["municipality"] = municipality
        gdf["source"] = f"OpenData_{municipality}"
        gdf["confidence"] = 1.0
        return gdf

    def _empty_address_gdf(self, municipality: str) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["address", "postal_code", "municipality", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    def _empty_facilities_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["facility_type", "source", "confidence", "is_critical_load", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
