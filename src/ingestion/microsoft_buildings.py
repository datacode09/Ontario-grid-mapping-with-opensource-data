"""Microsoft Global Building Footprints fetcher — Source 11.

https://github.com/microsoft/CanadianBuildingFootprints
Province-level GeoJSON files (~600MB for Ontario).
These are the best open proxy for individual household service point locations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Generator, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import shape

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# Ontario building footprints — partitioned by province on GitHub LFS
_MSFT_BUILDINGS_BASE = (
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
)
_MSFT_ONTARIO_URL = (
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/"
    "CanadianBuildingFootprints/Ontario.geojsonl.gz"
)


class MicrosoftBuildingsFetcher:
    """Fetch Microsoft building footprints for Ontario.

    The Ontario file is large (~600 MB compressed). This fetcher supports
    streaming chunk-reads and spatial filtering so only buildings within
    the AOI bbox are loaded into memory.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "microsoft_buildings",
            max_age_days=cfg["cache"]["max_age_days"],
        )
        self._chunk_size = cfg["async"]["chunk_size_bytes"]

    def fetch_buildings(
        self,
        bbox: Tuple[float, float, float, float],
        force: bool = False,
    ) -> gpd.GeoDataFrame:
        """Fetch Microsoft building footprints within a bounding box.

        Downloads the Ontario GeoJSONL file (streaming) and filters to bbox.

        Parameters
        ----------
        bbox: (west, south, east, north) in EPSG:4326
        force: Re-download if True

        Returns
        -------
        GeoDataFrame with columns: building_id, geometry (Polygon), source, confidence
        """
        cache_key = f"msft_buildings_{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
        cached = self.cache.get(cache_key)
        if cached and not force:
            log.info("MSFT buildings: loading from cache %s", cached)
            return gpd.read_file(cached)

        log.info("MSFT buildings: streaming Ontario buildings for bbox %s", bbox)
        gdf = self._stream_filter_buildings(bbox)

        if len(gdf):
            out_path = self.cache.cache_dir / f"{cache_key}.gpkg"
            gdf.to_file(out_path, driver="GPKG", layer="buildings")
            self.cache.put(cache_key, out_path, url=_MSFT_ONTARIO_URL)
            log.info("MSFT buildings: cached %d footprints", len(gdf))
        return gdf

    def _stream_filter_buildings(
        self, bbox: Tuple[float, float, float, float]
    ) -> gpd.GeoDataFrame:
        """Stream the Ontario GeoJSONL file and keep only buildings in bbox."""
        west, south, east, north = bbox
        records = []

        try:
            import gzip

            resp = requests.get(_MSFT_ONTARIO_URL, stream=True, timeout=600)
            resp.raise_for_status()

            # GeoJSONL is newline-delimited JSON (one feature per line)
            buf = b""
            decompressor = gzip.GzipFile(fileobj=resp.raw)
            building_id = 0

            for line in decompressor:
                line = line.strip()
                if not line:
                    continue
                try:
                    feature = json.loads(line)
                except json.JSONDecodeError:
                    continue

                geom = shape(feature["geometry"])
                centroid = geom.centroid

                if (west <= centroid.x <= east and south <= centroid.y <= north):
                    records.append({
                        "building_id": f"msft_{building_id}",
                        "geometry": geom,
                        "source": "Microsoft_BuildingFootprints",
                        "confidence": 0.85,
                        "area_m2": None,
                    })
                building_id += 1

                if building_id % 100_000 == 0:
                    log.info("MSFT buildings: processed %dk features, %d in bbox",
                             building_id // 1000, len(records))

        except Exception as exc:
            log.warning("MSFT buildings stream failed: %s", exc)
            return self._empty_buildings_gdf()

        if not records:
            log.warning("MSFT buildings: no features found in bbox")
            return self._empty_buildings_gdf()

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

        # Compute footprint area in square metres
        from ..utils.geometry import to_lambert
        gdf_lambert = to_lambert(gdf)
        gdf["area_m2"] = gdf_lambert.geometry.area

        return gdf

    def _empty_buildings_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["building_id", "source", "confidence", "area_m2", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    def building_centroids(self, buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Return centroid points for each building polygon."""
        centroids = buildings_gdf.copy()
        centroids["building_poly"] = centroids["geometry"]
        centroids["geometry"] = buildings_gdf.geometry.centroid
        centroids["proxy_type"] = "building_centroid"
        return centroids
