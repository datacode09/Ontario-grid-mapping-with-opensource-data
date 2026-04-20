"""NASA Black Marble Night-Time Lights fetcher — Source 19.

VNP46A1 daily 500m resolution product.
Used as input for gridfinder (Source 18) and as independent proxy
for electrification density.
https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/VNP46A1/
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# NASA LAADS DAAC — requires Earthdata login token
_LAADS_BASE = "https://ladsweb.modaps.eosdis.nasa.gov"
_VNP46A1_COLLECTION = "5000"
_EARTHDATA_TOKEN_ENV = "EARTHDATA_TOKEN"


class NightLightsFetcher:
    """Fetch NASA Black Marble VNP46A1 night-time lights for Ontario."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "nightlights",
            max_age_days=cfg["cache"]["max_age_days"],
        )
        self._token = os.environ.get(_EARTHDATA_TOKEN_ENV, "")

    def fetch_annual_composite(
        self,
        year: int,
        bbox: Tuple[float, float, float, float],
        force: bool = False,
    ) -> Optional[Path]:
        """Fetch or compose annual night-time lights mosaic for Ontario bbox.

        Uses NASA LAADS DAAC. Requires NASA Earthdata account token set
        in environment variable EARTHDATA_TOKEN.

        Returns
        -------
        Path to GeoTIFF or None if unavailable.
        """
        cache_key = f"nightlights_vnp46a1_{year}_{bbox[0]:.2f}_{bbox[1]:.2f}"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("Night-time lights %d: loading from cache", year)
            return cached

        if not self._token:
            log.warning(
                "EARTHDATA_TOKEN not set — night-time lights unavailable. "
                "gridfinder will use road network only."
            )
            return None

        log.info("Night-time lights: fetching VNP46A1 for %d", year)
        tiles = self._get_ontario_h5_tiles(year)
        if not tiles:
            log.warning("Night-time lights: no tiles returned for %d", year)
            return None

        merged_path = self._mosaic_and_clip(tiles, bbox, year)
        if merged_path:
            self.cache.put(cache_key, merged_path)
        return merged_path

    def _get_ontario_h5_tiles(self, year: int) -> list[Path]:
        """Retrieve VNP46A1 HDF5 tiles covering Ontario.

        Ontario falls approximately in VIIRS tiles h10v04 and h11v04.
        """
        h_tiles = [10, 11]
        v_tile = 4
        doy = 1  # Day of year — use Jan 1 as representative annual tile

        paths = []
        for h in h_tiles:
            tile_name = f"VNP46A1.A{year}{doy:03d}.h{h:02d}v{v_tile:02d}.001.*.h5"
            url = (
                f"{_LAADS_BASE}/archive/allData/{_VNP46A1_COLLECTION}/"
                f"VNP46A1/{year}/{doy:03d}/"
            )
            try:
                # List directory to find actual filename
                resp = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    # Parse JSON file listing from LAADS DAAC
                    files = [
                        f["name"] for f in resp.json()
                        if f["name"].endswith(".h5")
                        and f"h{h:02d}v{v_tile:02d}" in f["name"]
                    ]
                    for fname in files:
                        dl_url = url + fname
                        local = self._download_tile(dl_url, fname)
                        if local:
                            paths.append(local)
            except Exception as exc:
                log.warning("Night-lights tile h%02dv%02d failed: %s", h, v_tile, exc)

        return paths

    def _download_tile(self, url: str, filename: str) -> Optional[Path]:
        cache_key = f"nl_tile_{filename}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self._token}"},
                stream=True,
                timeout=300,
            )
            resp.raise_for_status()
            tmp_path = self.cache.cache_dir / filename
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
            return self.cache.put(cache_key, tmp_path, url=url)
        except Exception as exc:
            log.warning("Night-lights tile download failed (%s): %s", url, exc)
            return None

    def _mosaic_and_clip(
        self,
        tile_paths: list[Path],
        bbox: Tuple[float, float, float, float],
        year: int,
    ) -> Optional[Path]:
        """Extract radiance band from HDF5 tiles, mosaic, and clip to bbox."""
        try:
            import h5py
            import rasterio
            from rasterio.merge import merge
            from rasterio.transform import from_bounds

            arrays = []
            for tile_path in tile_paths:
                with h5py.File(tile_path, "r") as hf:
                    # VNP46A1 DNB at-sensor radiance
                    radiance = hf[
                        "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m"
                    ][:]
                    arrays.append(radiance.astype(np.float32))

            merged = np.concatenate(arrays, axis=1) if len(arrays) > 1 else arrays[0]

            out_path = self.cache.cache_dir / f"nightlights_{year}.tif"
            # Approximate transform for Ontario's VIIRS tiles
            west, south, east, north = -96.0, 41.0, -74.0, 57.0
            from rasterio.transform import from_bounds as fb
            transform = fb(west, south, east, north, merged.shape[1], merged.shape[0])

            with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=merged.shape[0], width=merged.shape[1],
                count=1, dtype="float32",
                crs="EPSG:4326",
                transform=transform,
            ) as dst:
                dst.write(merged, 1)

            return out_path
        except ImportError as exc:
            log.warning("h5py/rasterio not available for night-lights processing: %s", exc)
            return None
        except Exception as exc:
            log.warning("Night-lights mosaic failed: %s", exc)
            return None
