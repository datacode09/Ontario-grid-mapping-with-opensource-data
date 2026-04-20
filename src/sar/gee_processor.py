"""Google Earth Engine SAR processing path — server-side S1 analysis.

GEE is the PREFERRED processing path when credentials are available.
Avoids large local GeoTIFF downloads by running all operations server-side.

Source 23: Google Earth Engine Public Catalog
  COPERNICUS/S1_GRD  — full Sentinel-1 time-series archive
"""
from __future__ import annotations

import os
from typing import Optional

from ..sar import _SAR_DISCLAIMER
from ..utils.config_loader import load_sar_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

try:
    import ee
    _GEE_AVAILABLE = True
except ImportError:
    _GEE_AVAILABLE = False
    log.info("earthengine-api not installed — GEE processing unavailable")


def run_gee_change_detection(
    aoi_polygon,
    before_start: str,
    before_end: str,
    after_start: str,
    after_end: str,
    export_drive_folder: str = "ontario_grid_sar",
) -> dict:
    """Server-side GEE pipeline for Sentinel-1 change detection.

    Steps:
      1. Load S1_GRD collection for AOI and both date windows
      2. Apply orbit metadata filter (IW, VV+VH)
      3. Apply Lee speckle filter (5×5 kernel)
      4. Compute median composites for before/after windows
      5. Compute log-ratio image
      6. Export change mask and ratio image to Google Drive as COG

    Parameters
    ----------
    aoi_polygon:          Shapely Polygon in EPSG:4326.
    before_start/end:     ISO date range for reference period.
    after_start/end:      ISO date range for event period.
    export_drive_folder:  Google Drive folder for export.

    Returns
    -------
    Dict with GEE task IDs for monitoring export progress.
    Returns empty dict if GEE unavailable or auth fails.
    """
    if not _GEE_AVAILABLE:
        log.warning("GEE processor: earthengine-api not available")
        return {}

    cfg = load_sar_settings().get("sar", {})
    gee_cfg = cfg.get("gee", {})

    if not gee_cfg.get("enabled", False):
        log.info("GEE processor: disabled in sar_settings.yaml")
        return {}

    try:
        _authenticate_gee()
    except Exception as exc:
        log.error("GEE authentication failed: %s", exc)
        return {}

    try:
        # Convert Shapely polygon to GEE geometry
        aoi_ee = ee.Geometry.Polygon(list(aoi_polygon.exterior.coords))

        collection = ee.ImageCollection(gee_cfg.get("collection", "COPERNICUS/S1_GRD"))

        def filter_collection(start, end):
            return (
                collection
                .filterBounds(aoi_ee)
                .filterDate(start, end)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                .select(["VV", "VH"])
            )

        before_col = filter_collection(before_start, before_end)
        after_col  = filter_collection(after_start, after_end)

        # Apply Lee speckle filter (approximated via focal_mean in GEE)
        def lee_filter(img):
            kernel_size = cfg.get("speckle_filter", {}).get("kernel_size", 5)
            smoothed = img.focal_mean(kernel_size, "square", "pixels")
            return smoothed.copyProperties(img, img.propertyNames())

        before_filtered = before_col.map(lee_filter).median()
        after_filtered  = after_col.map(lee_filter).median()

        # Log-ratio: after_dB - before_dB
        # S1_GRD values are already in dB in GEE
        ratio = after_filtered.subtract(before_filtered).select("VV").rename("ratio_vv")

        # Change mask: |ratio| > moderate threshold
        moderate_threshold = cfg.get("change_thresholds_db", {}).get("moderate", 3.0)
        change_mask = ratio.abs().gt(moderate_threshold).rename("change_mask")

        scale_m = gee_cfg.get("scale_m", 10)
        task_ids = {}

        # Export ratio image
        ratio_task = ee.batch.Export.image.toDrive(
            image=ratio,
            description="ontario_sar_change_ratio",
            folder=export_drive_folder,
            fileNamePrefix=f"change_ratio_{before_start}_{after_end}",
            region=aoi_ee.bounds(),
            scale=scale_m,
            crs="EPSG:32617",
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        ratio_task.start()
        task_ids["ratio_task_id"] = ratio_task.id

        # Export change mask
        mask_task = ee.batch.Export.image.toDrive(
            image=change_mask,
            description="ontario_sar_change_mask",
            folder=export_drive_folder,
            fileNamePrefix=f"change_mask_{before_start}_{after_end}",
            region=aoi_ee.bounds(),
            scale=scale_m,
            crs="EPSG:32617",
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True},
        )
        mask_task.start()
        task_ids["mask_task_id"] = mask_task.id

        log.info(
            "GEE: export tasks started — ratio=%s, mask=%s",
            ratio_task.id, mask_task.id
        )
        task_ids["source"] = "SAR_contextual_RS"
        task_ids["is_exact_asset_geometry"] = False
        task_ids["disclaimer"] = _SAR_DISCLAIMER
        return task_ids

    except Exception as exc:
        log.error("GEE change detection pipeline failed: %s", exc)
        return {}


def _authenticate_gee() -> None:
    """Authenticate with GEE using service account or user credentials."""
    project_id = os.environ.get("GEE_PROJECT_ID", "")
    sa_email   = os.environ.get("GEE_SERVICE_ACCOUNT", "")
    key_path   = os.environ.get("GEE_PRIVATE_KEY_PATH", "")

    if sa_email and key_path and os.path.exists(key_path):
        credentials = ee.ServiceAccountCredentials(sa_email, key_path)
        ee.Initialize(credentials, project=project_id)
        log.info("GEE: authenticated with service account")
    else:
        ee.Initialize(project=project_id)
        log.info("GEE: authenticated with user credentials")


def poll_gee_task(task_id: str) -> dict:
    """Check the status of a GEE export task."""
    if not _GEE_AVAILABLE:
        return {"status": "unavailable"}
    try:
        task = ee.batch.Task.list()
        for t in task:
            if t.id == task_id:
                return {"task_id": task_id, "status": t.state, "metadata": t.status()}
        return {"task_id": task_id, "status": "not_found"}
    except Exception as exc:
        return {"task_id": task_id, "status": "error", "error": str(exc)}
