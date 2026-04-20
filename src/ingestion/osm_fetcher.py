"""OSM data fetcher — Source 7.

Uses osmnx (preferred) or direct Overpass QL queries.
Extracts all power infrastructure tags for Ontario.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, Point

from ..utils.cache import FileCache
from ..utils.config_loader import load_settings
from ..utils.geometry import bbox_to_polygon
from ..utils.logger import get_logger

log = get_logger(__name__)

try:
    import osmnx as ox
    _OSMNX_AVAILABLE = True
except ImportError:
    _OSMNX_AVAILABLE = False
    log.warning("osmnx not available; falling back to direct Overpass queries")


_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Power tags to extract
_POWER_VALUES = [
    "line", "minor_line", "cable", "substation", "transformer",
    "pole", "tower", "generator", "switch",
]

# Ontario voltage levels
_DISTRIBUTION_VOLTAGES = [25000, 14400, 13800, 8000, 4160, 2400]
_TRANSMISSION_VOLTAGES = [500000, 230000, 115000]


class OSMFetcher:
    """Fetch Ontario power infrastructure from OpenStreetMap."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        cfg = load_settings()
        self.cache = FileCache(
            cache_dir or Path(cfg["paths"]["cache_dir"]) / "osm",
            max_age_days=cfg["cache"]["max_age_days"],
        )
        self.overpass_url = cfg["osm"]["overpass_url"]
        self.timeout = cfg["osm"]["timeout_s"]
        self.max_retries = cfg["osm"]["max_retries"]

    # ------------------------------------------------------------------
    # Primary public interface
    # ------------------------------------------------------------------

    def fetch_power_infrastructure(
        self,
        bbox: Tuple[float, float, float, float],
        force: bool = False,
    ) -> dict[str, gpd.GeoDataFrame]:
        """Fetch all OSM power features within bbox.

        Parameters
        ----------
        bbox: (west, south, east, north) in EPSG:4326
        force: Re-download even if cached

        Returns
        -------
        Dict with keys: 'lines', 'substations', 'transformers',
                        'poles', 'towers', 'generators', 'switches'
        """
        cache_key = f"osm_power_{bbox[0]:.3f}_{bbox[1]:.3f}_{bbox[2]:.3f}_{bbox[3]:.3f}"
        cached_path = self.cache.get(cache_key)

        if cached_path and not force:
            log.info("OSM: loading from cache %s", cached_path)
            return self._load_cached(cached_path)

        log.info("OSM: fetching power features for bbox %s", bbox)
        if _OSMNX_AVAILABLE:
            result = self._fetch_via_osmnx(bbox)
        else:
            result = self._fetch_via_overpass(bbox)

        out_path = self.cache.cache_dir / f"{cache_key}.gpkg"
        self._save_to_gpkg(result, out_path)
        self.cache.put(cache_key, out_path)
        return result

    # ------------------------------------------------------------------
    # osmnx path
    # ------------------------------------------------------------------

    def _fetch_via_osmnx(
        self, bbox: Tuple[float, float, float, float]
    ) -> dict[str, gpd.GeoDataFrame]:
        west, south, east, north = bbox
        tags = {"power": True}
        try:
            gdf = ox.features_from_bbox(
                bbox=(north, south, east, west), tags=tags
            )
        except Exception as exc:
            log.warning("osmnx fetch failed: %s — falling back to Overpass", exc)
            return self._fetch_via_overpass(bbox)

        return self._split_by_power_type(gdf)

    # ------------------------------------------------------------------
    # Direct Overpass path
    # ------------------------------------------------------------------

    def _build_overpass_query(self, bbox: Tuple[float, float, float, float]) -> str:
        west, south, east, north = bbox
        bbox_str = f"{south},{west},{north},{east}"
        union_parts = "\n".join(
            f'  node["power"="{v}"]({bbox_str});\n'
            f'  way["power"="{v}"]({bbox_str});\n'
            f'  relation["power"="{v}"]({bbox_str});'
            for v in _POWER_VALUES
        )
        return f"""
[out:json][timeout:{self.timeout}];
(
{union_parts}
);
out body;
>;
out skel qt;
"""

    def _fetch_via_overpass(
        self, bbox: Tuple[float, float, float, float]
    ) -> dict[str, gpd.GeoDataFrame]:
        query = self._build_overpass_query(bbox)
        raw = self._overpass_request(query)
        if not raw:
            return self._empty_result()

        return self._parse_overpass_json(raw)

    def _overpass_request(self, query: str) -> Optional[dict]:
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.overpass_url,
                    data={"data": query},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as exc:
                wait = 2 ** attempt
                log.warning("Overpass attempt %d failed: %s — retry in %ds", attempt + 1, exc, wait)
                time.sleep(wait)
        log.error("Overpass: all %d attempts failed", self.max_retries)
        return None

    def _parse_overpass_json(self, raw: dict) -> dict[str, gpd.GeoDataFrame]:
        """Parse Overpass JSON into split GeoDataFrames by power type."""
        nodes = {e["id"]: e for e in raw.get("elements", []) if e["type"] == "node"}
        ways = [e for e in raw.get("elements", []) if e["type"] == "way"]

        records: dict[str, list] = {k: [] for k in [
            "lines", "substations", "transformers", "poles", "towers", "generators", "switches"
        ]}

        def node_point(nid):
            n = nodes.get(nid)
            return Point(n["lon"], n["lat"]) if n else None

        for way in ways:
            tags = way.get("tags", {})
            power = tags.get("power", "")
            coords = [
                (nodes[n]["lon"], nodes[n]["lat"])
                for n in way.get("nodes", [])
                if n in nodes
            ]
            if len(coords) < 2:
                continue
            geom = LineString(coords)
            rec = {**tags, "osm_id": way["id"], "geometry": geom,
                   "source": "OSM", "confidence": self._infer_confidence(tags)}
            self._assign_record(records, power, rec)

        for nid, node in nodes.items():
            tags = node.get("tags", {})
            power = tags.get("power", "")
            geom = Point(node["lon"], node["lat"])
            rec = {**tags, "osm_id": nid, "geometry": geom,
                   "source": "OSM", "confidence": self._infer_confidence(tags)}
            self._assign_record(records, power, rec)

        return {
            key: gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
            if recs else self._empty_gdf()
            for key, recs in records.items()
        }

    def _assign_record(self, records: dict, power: str, rec: dict) -> None:
        mapping = {
            "line": "lines", "minor_line": "lines", "cable": "lines",
            "substation": "substations",
            "transformer": "transformers",
            "pole": "poles",
            "tower": "towers",
            "generator": "generators",
            "switch": "switches",
        }
        key = mapping.get(power)
        if key:
            records[key].append(rec)

    def _infer_confidence(self, tags: dict) -> float:
        cfg = load_settings()
        conf = cfg["confidence"]
        if tags.get("voltage"):
            return conf["osm_confirmed"]
        return conf["osm_inferred"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_by_power_type(self, gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
        """Split an osmnx GeoDataFrame by the power= tag value."""
        result = {}
        if "power" not in gdf.columns:
            return self._empty_result()

        line_types = {"line", "minor_line", "cable"}
        result["lines"] = gdf[gdf["power"].isin(line_types)].copy()
        result["substations"] = gdf[gdf["power"] == "substation"].copy()
        result["transformers"] = gdf[gdf["power"] == "transformer"].copy()
        result["poles"] = gdf[gdf["power"] == "pole"].copy()
        result["towers"] = gdf[gdf["power"] == "tower"].copy()
        result["generators"] = gdf[gdf["power"] == "generator"].copy()
        result["switches"] = gdf[gdf["power"] == "switch"].copy()

        for key, df in result.items():
            if len(df):
                df["source"] = "OSM"
                df["confidence"] = df.apply(
                    lambda row: self._infer_confidence(row.to_dict()), axis=1
                )
        return result

    def _empty_result(self) -> dict[str, gpd.GeoDataFrame]:
        return {k: self._empty_gdf() for k in [
            "lines", "substations", "transformers", "poles", "towers", "generators", "switches"
        ]}

    def _empty_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(columns=["osm_id", "source", "confidence", "geometry"],
                                geometry="geometry", crs="EPSG:4326")

    def _save_to_gpkg(self, result: dict, out_path: Path) -> None:
        for layer_name, gdf in result.items():
            if len(gdf):
                gdf.to_file(out_path, driver="GPKG", layer=layer_name)

    def _load_cached(self, path: Path) -> dict[str, gpd.GeoDataFrame]:
        import fiona
        layers = fiona.listlayers(path)
        result = {}
        for layer in layers:
            result[layer] = gpd.read_file(path, layer=layer)
        return result
