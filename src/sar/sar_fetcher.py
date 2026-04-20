"""SAR scene fetcher — download Sentinel-1 RTC from ASF DAAC (primary) or CDSE.

Source 20: Copernicus Sentinel-1 GRD/RTC via CDSE (fallback)
  https://dataspace.copernicus.eu/
Source 21: ASF DAAC Sentinel-1 RTC (primary, pre-processed Gamma0)
  https://search.asf.alaska.edu/
  Python: asf_search library

Scene selection: IW mode, VV+VH, intersecting Ontario corridor bbox.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import Polygon

from ..utils.cache import FileCache
from ..utils.config_loader import load_sar_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

try:
    import asf_search as asf
    _ASF_AVAILABLE = True
except ImportError:
    _ASF_AVAILABLE = False
    log.info("asf_search not installed — install requirements-sar.txt for SAR support")


class SARFetcher:
    """Download Sentinel-1 RTC scenes from ASF DAAC or CDSE fallback."""

    _SAR_DISCLAIMER = (
        "source=SAR_contextual_RS | is_exact_asset_geometry=false"
    )

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        cfg = load_sar_settings()
        self._sar_cfg = cfg.get("sar", {})
        self.output_dir = output_dir or Path(
            "data/raw/sar/sentinel1_rtc"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._asf_user = os.environ.get("ASF_USERNAME", "")
        self._asf_pass = os.environ.get("ASF_PASSWORD", "")

    def fetch_sentinel1_rtc(
        self,
        aoi_polygon: Polygon,
        start_date: str,
        end_date: str,
        polarisation: list = None,
        beam_mode: str = "IW",
        force: bool = False,
    ) -> list[Path]:
        """Query and download Sentinel-1 RTC scenes intersecting AOI.

        Parameters
        ----------
        aoi_polygon:   Shapely Polygon in EPSG:4326 defining the AOI.
        start_date:    ISO date string "YYYY-MM-DD".
        end_date:      ISO date string "YYYY-MM-DD".
        polarisation:  List of polarisations, default ["VV", "VH"].
        beam_mode:     SAR acquisition mode, default "IW" (Interferometric Wide).
        force:         Re-download if True.

        Returns
        -------
        List of local GeoTIFF file paths (VV and VH bands).
        Falls back to CDSE OData API if ASF quota is exceeded.
        """
        polarisation = polarisation or self._sar_cfg.get(
            "sentinel1", {}
        ).get("polarisation", ["VV", "VH"])

        log.info(
            "SARFetcher: searching Sentinel-1 RTC for %s to %s", start_date, end_date
        )

        if _ASF_AVAILABLE and self._asf_user:
            paths = self._fetch_via_asf(aoi_polygon, start_date, end_date, beam_mode, force)
            if paths:
                return paths
            log.warning("ASF fetch returned no results — trying CDSE fallback")

        return self._fetch_via_cdse(aoi_polygon, start_date, end_date, polarisation)

    # ------------------------------------------------------------------
    # ASF DAAC path
    # ------------------------------------------------------------------

    def _fetch_via_asf(
        self,
        aoi_polygon: Polygon,
        start_date: str,
        end_date: str,
        beam_mode: str,
        force: bool,
    ) -> list[Path]:
        """Search and download from ASF DAAC."""
        try:
            from shapely.geometry import mapping
            import json

            results = asf.search(
                intersectsWith=aoi_polygon.wkt,
                start=start_date,
                end=end_date,
                platform=asf.PLATFORM.SENTINEL1,
                beamMode=beam_mode,
                processingLevel="RTC_HI_RES",
                maxResults=self._sar_cfg.get("asf", {}).get("max_results", 50),
            )

            if not results:
                log.info("ASF: no RTC results — trying GRD")
                results = asf.search(
                    intersectsWith=aoi_polygon.wkt,
                    start=start_date,
                    end=end_date,
                    platform=asf.PLATFORM.SENTINEL1,
                    beamMode=beam_mode,
                    processingLevel="GRD_HD",
                    maxResults=50,
                )

            if not results:
                return []

            log.info("ASF: found %d scenes", len(results))
            session = asf.ASFSession().auth_with_creds(self._asf_user, self._asf_pass)
            downloaded = []

            for product in results[:5]:  # Limit initial download
                fname = product.properties.get("fileName", str(product))
                out_path = self.output_dir / fname
                if out_path.exists() and not force:
                    downloaded.append(out_path)
                    continue
                try:
                    product.download(path=self.output_dir, session=session)
                    downloaded.append(out_path)
                    log.info("ASF: downloaded %s", fname)
                except Exception as exc:
                    log.warning("ASF download failed for %s: %s", fname, exc)

            return downloaded

        except Exception as exc:
            log.warning("ASF search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # CDSE fallback path
    # ------------------------------------------------------------------

    def _fetch_via_cdse(
        self,
        aoi_polygon: Polygon,
        start_date: str,
        end_date: str,
        polarisation: list,
    ) -> list[Path]:
        """Search Copernicus Data Space Ecosystem (CDSE) OData API."""
        cdse_user = os.environ.get("CDSE_USERNAME", "")
        cdse_pass = os.environ.get("CDSE_PASSWORD", "")

        if not cdse_user or not cdse_pass:
            log.warning(
                "CDSE credentials not set. Set CDSE_USERNAME and CDSE_PASSWORD."
            )
            return []

        try:
            import requests
            # CDSE OData search endpoint
            base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
            bbox = aoi_polygon.bounds
            filter_str = (
                f"Collection/Name eq 'SENTINEL-1' and "
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
                f"and att/OData.CSC.StringAttribute/Value eq 'GRD') and "
                f"ContentDate/Start gt {start_date}T00:00:00.000Z and "
                f"ContentDate/Start lt {end_date}T23:59:59.000Z and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_polygon.wkt}')"
            )

            resp = requests.get(
                base_url,
                params={"$filter": filter_str, "$top": 20, "$orderby": "ContentDate/Start desc"},
                auth=(cdse_user, cdse_pass),
                timeout=60,
            )
            resp.raise_for_status()
            products = resp.json().get("value", [])
            log.info("CDSE: found %d products", len(products))

            downloaded = []
            for product in products[:3]:
                product_id = product.get("Id")
                name = product.get("Name", product_id)
                out_path = self.output_dir / f"{name}.zip"

                if out_path.exists():
                    downloaded.append(out_path)
                    continue

                dl_url = f"{base_url}({product_id})/$value"
                dl_resp = requests.get(
                    dl_url, auth=(cdse_user, cdse_pass), stream=True, timeout=300
                )
                if dl_resp.status_code == 200:
                    with open(out_path, "wb") as fh:
                        for chunk in dl_resp.iter_content(65536):
                            fh.write(chunk)
                    downloaded.append(out_path)
                    log.info("CDSE: downloaded %s", name)

            return downloaded

        except Exception as exc:
            log.error("CDSE fetch failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # PALSAR-2 (Source 24)
    # ------------------------------------------------------------------

    def fetch_palsar2_mosaic(
        self,
        bbox: tuple,
        year: int = 2023,
        output_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Download JAXA PALSAR-2 annual mosaic tile for the AOI.

        L-band (23cm) provides better penetration under forest canopy
        for northern Ontario corridor baselines.
        """
        output_dir = output_dir or self.output_dir.parent / "palsar2_mosaic"
        output_dir.mkdir(parents=True, exist_ok=True)

        west, south, east, north = bbox
        # JAXA tile naming: N{lat:02d}E{lon:03d}_F02DAR (3-degree tiles)
        tile_lat = int(south // 5) * 5
        tile_lon = int(west // 5) * 5
        tile_name = f"N{abs(tile_lat):02d}{'S' if tile_lat < 0 else 'N'}{abs(tile_lon):03d}E"

        url = (
            f"https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/data/{year}/"
            f"{tile_name}.tar.gz"
        )
        out_path = output_dir / f"palsar2_{year}_{tile_name}.tar.gz"

        if out_path.exists():
            return out_path

        try:
            import requests
            resp = requests.get(url, stream=True, timeout=300)
            if resp.status_code == 200:
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(65536):
                        fh.write(chunk)
                log.info("PALSAR-2: downloaded %s", tile_name)
                return out_path
            else:
                log.warning("PALSAR-2 tile not found: %s (HTTP %d)", url, resp.status_code)
                return None
        except Exception as exc:
            log.warning("PALSAR-2 download failed: %s", exc)
            return None
