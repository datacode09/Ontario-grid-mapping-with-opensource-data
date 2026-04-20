"""SAR enrichment module — optional contextual remote-sensing layer.

All public functions in this module return None gracefully if
SAR_ENABLED=false in sar_settings.yaml or via environment variable.

LABELLING REQUIREMENT: All SAR-derived products carry the following mandatory
disclaimer in every popup, report, and exported attribute:

    "This layer is a contextual remote-sensing product derived from SAR
    backscatter data. It does not represent exact electrical asset geometry
    and should not be used for engineering or operational decisions without
    utility-authoritative verification."

SAR confidence (sar_confidence) is NEVER merged with grid topology confidence.
These are independent attributes on separate fields.
"""
from __future__ import annotations

from ..utils.config_loader import is_sar_enabled
from ..utils.logger import get_logger

log = get_logger(__name__)

_SAR_DISCLAIMER = (
    "This layer is a contextual remote-sensing product derived from SAR "
    "backscatter data (Sentinel-1 / PALSAR-2). It does not represent exact "
    "electrical asset geometry and should not be used for engineering or "
    "operational decisions without utility-authoritative verification."
)

_SAR_POPUP_BLOCK = (
    "─────────────────────────────────────────────\n"
    "⚠️  CONTEXTUAL REMOTE-SENSING PRODUCT\n"
    "This layer is derived from SAR backscatter data\n"
    "(Sentinel-1 / PALSAR-2). It does not represent\n"
    "exact electrical asset geometry and must not be\n"
    "used for engineering or operational decisions\n"
    "without utility-authoritative verification.\n"
    "─────────────────────────────────────────────"
)


def check_sar_enabled() -> bool:
    """Return True if SAR module is enabled in configuration."""
    enabled = is_sar_enabled()
    if not enabled:
        log.debug("SAR module disabled (SAR_ENABLED=false)")
    return enabled


# Lazy imports to avoid loading SAR deps when module is disabled
def get_sar_fetcher():
    if not check_sar_enabled():
        return None
    from .sar_fetcher import SARFetcher
    return SARFetcher


def get_corridor_extractor():
    if not check_sar_enabled():
        return None
    from .corridor_extractor import CorridorExtractor
    return CorridorExtractor


def get_flood_detector():
    if not check_sar_enabled():
        return None
    from .flood_detector import FloodDetector
    return FloodDetector


def get_change_detector():
    if not check_sar_enabled():
        return None
    from .change_detector import ChangeDetector
    return ChangeDetector


def get_exposure_scorer():
    if not check_sar_enabled():
        return None
    from .exposure_scorer import ExposureScorer
    return ExposureScorer


def get_sar_exporter():
    if not check_sar_enabled():
        return None
    from .sar_exporter import SARExporter
    return SARExporter


__all__ = [
    "check_sar_enabled",
    "get_sar_fetcher",
    "get_corridor_extractor",
    "get_flood_detector",
    "get_change_detector",
    "get_exposure_scorer",
    "get_sar_exporter",
    "_SAR_DISCLAIMER",
    "_SAR_POPUP_BLOCK",
]
