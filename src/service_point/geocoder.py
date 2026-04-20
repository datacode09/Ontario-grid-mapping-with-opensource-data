"""Address geocoder — convert civic addresses to WGS84 coordinates."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

_NOMINATIM_BASE = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_RATE_LIMIT_S = 1.1  # Nominatim usage policy: max 1 req/sec


class Geocoder:
    """Geocode Ontario addresses using Nominatim (OSM) with municipal data fallback.

    For Mississauga pilot area, prefers municipal open-data address points
    over external geocoding to reduce API calls and improve accuracy.
    """

    def __init__(
        self,
        municipal_addresses: Optional[gpd.GeoDataFrame] = None,
    ) -> None:
        self._municipal_addresses = municipal_addresses
        self._last_request_time: float = 0.0

    def geocode(self, address: str, municipality: str = "Ontario, Canada") -> Optional[Point]:
        """Geocode a single address to a WGS84 Point.

        First checks the municipal address dataset, then falls back to Nominatim.
        """
        # Try municipal dataset first (authoritative, no rate limit)
        if self._municipal_addresses is not None:
            pt = self._lookup_municipal(address)
            if pt:
                return pt

        # Nominatim fallback
        return self._nominatim_geocode(f"{address}, {municipality}")

    def geocode_dataframe(
        self,
        df: pd.DataFrame,
        address_col: str = "address",
        max_rows: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Geocode all addresses in a DataFrame.

        Returns GeoDataFrame with lat, lon, geometry, geocode_source columns added.
        """
        results = []
        total = min(len(df), max_rows) if max_rows else len(df)

        for i, (idx, row) in enumerate(df.iterrows()):
            if max_rows and i >= max_rows:
                break
            address = str(row.get(address_col, ""))
            pt = self.geocode(address)
            results.append({
                "address": address,
                "lat": pt.y if pt else None,
                "lon": pt.x if pt else None,
                "geometry": pt,
                "geocode_source": "municipal" if self._lookup_municipal(address) else "nominatim",
            })

            if (i + 1) % 100 == 0:
                log.info("Geocoder: processed %d/%d addresses", i + 1, total)

        return gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _lookup_municipal(self, address: str) -> Optional[Point]:
        """Fuzzy match against municipal address dataset."""
        if self._municipal_addresses is None:
            return None
        if "address" not in self._municipal_addresses.columns:
            return None

        addr_lower = address.lower().strip()
        matches = self._municipal_addresses[
            self._municipal_addresses["address"].str.lower().str.contains(
                addr_lower[:20], na=False
            )
        ]
        if len(matches):
            row = matches.iloc[0]
            geom = row.geometry
            if geom and not geom.is_empty:
                return geom if isinstance(geom, Point) else geom.centroid
        return None

    def _nominatim_geocode(self, query: str) -> Optional[Point]:
        """Geocode using Nominatim with rate limiting."""
        # Enforce rate limit
        elapsed = time.time() - self._last_request_time
        if elapsed < _NOMINATIM_RATE_LIMIT_S:
            time.sleep(_NOMINATIM_RATE_LIMIT_S - elapsed)

        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "countrycodes": "ca",
        }
        headers = {"User-Agent": "OntarioGridMapper/3.0 (open-source research tool)"}

        try:
            resp = requests.get(
                _NOMINATIM_BASE, params=params, headers=headers, timeout=15
            )
            resp.raise_for_status()
            self._last_request_time = time.time()

            results = resp.json()
            if results:
                lon = float(results[0]["lon"])
                lat = float(results[0]["lat"])
                return Point(lon, lat)
        except Exception as exc:
            log.debug("Nominatim geocode failed for '%s': %s", query, exc)

        return None
