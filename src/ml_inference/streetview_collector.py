"""Street-level imagery collector for GridMapping ML inference.

Source 16: Google Street View Static API (free tier: 28k images/month)
Alternative: Mapillary open imagery (fully open, less dense)
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional, Tuple

import aiohttp
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)


class StreetViewCollector:
    """Collect street-level imagery along road network segments.

    Supports Google Street View and Mapillary as providers.
    Images are saved to cache for downstream pole detection.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "street_view_cache",
            max_age_days=cfg["cache"]["max_age_days"],
        )
        ml_cfg = cfg["ml_inference"]["streetview"]
        self._provider = ml_cfg.get("provider", "mapillary")
        self._images_per_km = ml_cfg.get("images_per_km", 4)
        self._gsv_key = os.environ.get("GOOGLE_STREETVIEW_API_KEY", "")
        self._mapillary_token = os.environ.get("MAPILLARY_CLIENT_TOKEN", "")

    def collect_along_roads(
        self,
        roads_gdf: gpd.GeoDataFrame,
        max_images: int = 500,
    ) -> list[dict]:
        """Sample imagery points along road segments and download images.

        Parameters
        ----------
        roads_gdf:   GeoDataFrame of road LineStrings.
        max_images:  Maximum number of images to collect.

        Returns
        -------
        List of dicts: {lat, lon, image_path, provider, road_id}
        """
        sample_points = self._sample_points_along_roads(roads_gdf, max_images)
        log.info("StreetViewCollector: collecting %d imagery points", len(sample_points))

        if self._provider == "mapillary" and self._mapillary_token:
            return asyncio.run(self._collect_mapillary(sample_points))
        elif self._provider == "google" and self._gsv_key:
            return asyncio.run(self._collect_gsv(sample_points))
        else:
            log.warning(
                "No valid imagery API credentials found. "
                "Set GOOGLE_STREETVIEW_API_KEY or MAPILLARY_CLIENT_TOKEN."
            )
            return []

    # ------------------------------------------------------------------
    # Point sampling
    # ------------------------------------------------------------------

    def _sample_points_along_roads(
        self,
        roads_gdf: gpd.GeoDataFrame,
        max_images: int,
    ) -> list[dict]:
        """Sample evenly spaced points along road geometries."""
        from ..utils.geometry import to_lambert
        roads_lambert = to_lambert(roads_gdf)
        spacing_m = 1000.0 / self._images_per_km

        points = []
        for idx, row in roads_lambert.iterrows():
            geom = row.geometry
            if geom is None or geom.length == 0:
                continue
            n_samples = max(1, int(geom.length / spacing_m))
            for i in range(n_samples):
                dist = (i / n_samples) * geom.length
                pt_lambert = geom.interpolate(dist)
                from pyproj import Transformer
                t = Transformer.from_crs("EPSG:3347", "EPSG:4326", always_xy=True)
                lon, lat = t.transform(pt_lambert.x, pt_lambert.y)
                points.append({
                    "lat": lat,
                    "lon": lon,
                    "road_id": str(idx),
                })
                if len(points) >= max_images:
                    return points

        return points

    # ------------------------------------------------------------------
    # Mapillary async collector
    # ------------------------------------------------------------------

    async def _collect_mapillary(self, points: list[dict]) -> list[dict]:
        """Search Mapillary for imagery near each sampled point."""
        base_url = "https://graph.mapillary.com/images"
        results = []

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._query_mapillary_point(session, base_url, pt)
                for pt in points
            ]
            chunk_size = 20
            for i in range(0, len(tasks), chunk_size):
                batch = await asyncio.gather(*tasks[i:i + chunk_size], return_exceptions=True)
                for r in batch:
                    if isinstance(r, dict):
                        results.append(r)

        log.info("StreetViewCollector: collected %d Mapillary images", len(results))
        return results

    async def _query_mapillary_point(
        self, session: aiohttp.ClientSession, base_url: str, point: dict
    ) -> Optional[dict]:
        params = {
            "access_token": self._mapillary_token,
            "fields":       "id,thumb_1024_url,geometry",
            "bbox":         f"{point['lon']-0.001},{point['lat']-0.001},"
                            f"{point['lon']+0.001},{point['lat']+0.001}",
            "limit":        1,
        }
        cache_key = f"mapillary_{point['lat']:.5f}_{point['lon']:.5f}"
        cached = self.cache.get(cache_key)
        if cached:
            return {"lat": point["lat"], "lon": point["lon"],
                    "image_path": str(cached), "provider": "mapillary"}

        try:
            async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                features = data.get("data", [])
                if not features:
                    return None

                feature = features[0]
                img_url = feature.get("thumb_1024_url")
                if not img_url:
                    return None

                img_path = await self._download_image(session, img_url, cache_key)
                if img_path:
                    return {"lat": point["lat"], "lon": point["lon"],
                            "image_path": str(img_path), "provider": "mapillary",
                            "road_id": point.get("road_id")}
        except Exception as exc:
            log.debug("Mapillary query failed at (%.4f, %.4f): %s", point["lat"], point["lon"], exc)
        return None

    async def _download_image(
        self, session: aiohttp.ClientSession, url: str, cache_key: str
    ) -> Optional[Path]:
        tmp_path = self.cache.cache_dir / f"{cache_key}.jpg"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    tmp_path.write_bytes(content)
                    return self.cache.put(cache_key, tmp_path, url=url)
        except Exception as exc:
            log.debug("Image download failed (%s): %s", url, exc)
        return None

    # ------------------------------------------------------------------
    # Google Street View async collector
    # ------------------------------------------------------------------

    async def _collect_gsv(self, points: list[dict]) -> list[dict]:
        """Download Street View static images from Google API."""
        base_url = "https://maps.googleapis.com/maps/api/streetview"
        results = []

        async with aiohttp.ClientSession() as session:
            for pt in points:
                result = await self._query_gsv_point(session, base_url, pt)
                if result:
                    results.append(result)

        return results

    async def _query_gsv_point(
        self, session: aiohttp.ClientSession, base_url: str, point: dict
    ) -> Optional[dict]:
        cache_key = f"gsv_{point['lat']:.5f}_{point['lon']:.5f}"
        cached = self.cache.get(cache_key)
        if cached:
            return {"lat": point["lat"], "lon": point["lon"],
                    "image_path": str(cached), "provider": "google"}

        params = {
            "size":     "640x640",
            "location": f"{point['lat']},{point['lon']}",
            "fov":      "90",
            "pitch":    "10",
            "key":      self._gsv_key,
        }
        try:
            async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    tmp_path = self.cache.cache_dir / f"{cache_key}.jpg"
                    tmp_path.write_bytes(content)
                    img_path = self.cache.put(cache_key, tmp_path)
                    return {"lat": point["lat"], "lon": point["lon"],
                            "image_path": str(img_path), "provider": "google",
                            "road_id": point.get("road_id")}
        except Exception as exc:
            log.debug("GSV query failed: %s", exc)
        return None
