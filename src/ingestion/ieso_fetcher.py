"""IESO data fetcher — Sources 3 & 4.

Source 3: IESO Transmission Facility Registry (line ratings, substation locations)
Source 4: IESO Generator Output and Capability
  https://www.ieso.ca/en/Power-Data/Data-Directory
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

_IESO_DATA_DIR = "https://www.ieso.ca/-/media/Files/IESO/Document-Library/data-directory"
_IESO_GENERATOR_URL = (
    "https://www.ieso.ca/-/media/Files/IESO/Document-Library/data-directory/"
    "Generator_Output_and_Capability.xlsx"
)
_IESO_TX_FACILITY_URL = (
    "https://www.ieso.ca/-/media/Files/IESO/Document-Library/data-directory/"
    "Transmission_Facility_Registry.xlsx"
)


class IESOFetcher:
    """Fetch IESO transmission facility registry and generator capability data."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "ieso",
            max_age_days=cfg["cache"]["max_age_days"],
        )

    # ------------------------------------------------------------------
    # Source 3: Transmission Facility Registry
    # ------------------------------------------------------------------

    def fetch_transmission_registry(self, force: bool = False) -> gpd.GeoDataFrame:
        """Fetch IESO transmission facility registry.

        Returns GeoDataFrame with columns:
          facility_name, voltage_kv, from_node, to_node,
          rating_mva, operator, geometry (LineString or Point), source, confidence
        """
        cache_key = "ieso_tx_registry"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("IESO TX registry: loading from cache")
            return gpd.read_file(cached)

        log.info("IESO TX registry: downloading from %s", _IESO_TX_FACILITY_URL)
        try:
            resp = requests.get(_IESO_TX_FACILITY_URL, timeout=120)
            resp.raise_for_status()
            df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
        except Exception as exc:
            log.warning("IESO TX registry download failed: %s", exc)
            return self._empty_tx_gdf()

        gdf = self._tx_registry_to_gdf(df)
        out_path = self.cache.cache_dir / "ieso_tx_registry.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="tx_facilities")
        self.cache.put(cache_key, out_path, url=_IESO_TX_FACILITY_URL)
        log.info("IESO TX registry: cached %d facilities", len(gdf))
        return gdf

    def _tx_registry_to_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        rename = {
            "station_name": "facility_name",
            "facility": "facility_name",
            "voltage_(kv)": "voltage_kv",
            "voltage_kv": "voltage_kv",
            "nominal_voltage": "voltage_kv",
            "rating_(mva)": "rating_mva",
            "from_station": "from_node",
            "to_station": "to_node",
            "latitude": "lat",
            "longitude": "lon",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        geometries = []
        for _, row in df.iterrows():
            lat = row.get("lat")
            lon = row.get("lon")
            if pd.notna(lat) and pd.notna(lon):
                geometries.append(Point(float(lon), float(lat)))
            else:
                geometries.append(None)

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        gdf["source"] = "IESO_TX_Registry"
        gdf["confidence"] = 1.0
        gdf["operator"] = gdf.get("operator", "Hydro One Transmission")
        return gdf

    def _empty_tx_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=[
                "facility_name", "voltage_kv", "from_node", "to_node",
                "rating_mva", "operator", "source", "confidence", "geometry"
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

    # ------------------------------------------------------------------
    # Source 4: Generator Output and Capability
    # ------------------------------------------------------------------

    def fetch_generator_capability(self, force: bool = False) -> gpd.GeoDataFrame:
        """Fetch IESO generator output and capability data.

        Returns GeoDataFrame with columns:
          generator_name, fuel_type, capacity_mw, lat, lon,
          operator, source, confidence, geometry
        """
        cache_key = "ieso_generators"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("IESO generators: loading from cache")
            return gpd.read_file(cached)

        log.info("IESO generators: downloading")
        try:
            resp = requests.get(_IESO_GENERATOR_URL, timeout=120)
            resp.raise_for_status()
            df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
        except Exception as exc:
            log.warning("IESO generator download failed: %s", exc)
            return self._empty_gen_gdf()

        gdf = self._generator_to_gdf(df)
        out_path = self.cache.cache_dir / "ieso_generators.gpkg"
        gdf.to_file(out_path, driver="GPKG", layer="generators")
        self.cache.put(cache_key, out_path, url=_IESO_GENERATOR_URL)
        log.info("IESO generators: cached %d units", len(gdf))
        return gdf

    def _generator_to_gdf(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename = {
            "generator": "generator_name",
            "plant_name": "generator_name",
            "fuel": "fuel_type",
            "fuel_type": "fuel_type",
            "capacity_(mw)": "capacity_mw",
            "nameplated_capacity_(mw)": "capacity_mw",
            "latitude": "lat",
            "longitude": "lon",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        geometries = [
            Point(float(row.get("lon", 0)), float(row.get("lat", 0)))
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon"))
            else None
            for _, row in df.iterrows()
        ]

        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        gdf["source"] = "IESO_Generator_Registry"
        gdf["confidence"] = 1.0
        gdf["node_type"] = "generator"
        return gdf

    def _empty_gen_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["generator_name", "fuel_type", "capacity_mw", "source", "confidence", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    # ------------------------------------------------------------------
    # Tie-line nodes (IESO control area borders)
    # ------------------------------------------------------------------

    def get_tie_line_nodes(self) -> gpd.GeoDataFrame:
        """Return known IESO tie-line connection points (hardcoded from public data)."""
        tie_lines = [
            {"name": "ON-QC B2B (Hawthorne)", "direction": "ON↔QC", "lat": 45.435, "lon": -75.619, "voltage_kv": 230},
            {"name": "ON-MB Tie (Richer)", "direction": "ON↔MB", "lat": 49.878, "lon": -96.587, "voltage_kv": 115},
            {"name": "ON-MI Tie (St Clair)", "direction": "ON↔MI", "lat": 42.973, "lon": -82.422, "voltage_kv": 345},
            {"name": "ON-MN Tie (Baudette)", "direction": "ON↔MN", "lat": 48.716, "lon": -94.587, "voltage_kv": 230},
            {"name": "ON-NY Tie (Niagara)", "direction": "ON↔NY", "lat": 43.084, "lon": -79.067, "voltage_kv": 345},
            {"name": "ON-OH Tie (Windsor)", "direction": "ON↔OH", "lat": 42.315, "lon": -83.036, "voltage_kv": 230},
        ]
        df = pd.DataFrame(tie_lines)
        geometry = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
        gdf["source"] = "IESO_TieLine_static"
        gdf["confidence"] = 1.0
        gdf["node_type"] = "tie_line"
        return gdf
