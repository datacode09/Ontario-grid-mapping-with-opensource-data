"""Data ingestion layer — fetch all 27 open data sources."""
from .oeb_fetcher import OEBFetcher
from .ieso_fetcher import IESOFetcher
from .osm_fetcher import OSMFetcher
from .statcan_fetcher import StatCanFetcher
from .microsoft_buildings import MicrosoftBuildingsFetcher
from .nrcan_fetcher import NRCanFetcher
from .nightlights_fetcher import NightLightsFetcher
from .municipal_fetcher import MunicipalFetcher

__all__ = [
    "OEBFetcher",
    "IESOFetcher",
    "OSMFetcher",
    "StatCanFetcher",
    "MicrosoftBuildingsFetcher",
    "NRCanFetcher",
    "NightLightsFetcher",
    "MunicipalFetcher",
]
