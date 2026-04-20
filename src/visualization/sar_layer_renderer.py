"""SAR-specific layer renderer for Folium maps.

Renders SAR layers with mandatory visual treatment:
  - Hatched fill patterns (not solid)
  - Italic legend text
  - ⚠️ prefix on all popup titles
  - Mandatory disclaimer in every popup
  - Separate layer group "⚠️ SAR Context (Remote Sensing — Not Asset Geometry)"

The group is COLLAPSED BY DEFAULT.

Per the spec:
  SAR corridor footprints     → Cyan hatched polygon, 30% opacity
  SAR water / flood extent    → Blue hatched polygon, 50% opacity
  SAR change mask — moderate  → Yellow hatched polygon
  SAR change mask — significant → Orange hatched polygon
  SAR change mask — extreme   → Red hatched polygon
  Resilience risk — LOW/MEDIUM/HIGH/CRITICAL → Colored circles
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd

from ..sar import _SAR_POPUP_BLOCK, _SAR_DISCLAIMER
from ..utils.logger import get_logger

log = get_logger(__name__)

_SAR_GROUP_NAME = "⚠️ SAR Context (Remote Sensing — Not Asset Geometry)"
_SAR_BANNER = (
    "These layers are derived from satellite SAR data. "
    "They do not represent exact electrical assets. "
    "For contextual situational awareness only."
)

_RISK_COLORS = {
    "LOW":      "#00aa44",
    "MEDIUM":   "#ffcc00",
    "HIGH":     "#ff6600",
    "CRITICAL": "#cc0000",
}

# Hatched fill SVG patterns (embedded in Folium popups and layer styles)
_HATCH_CSS = """
<style>
.sar-corridor  { fill: cyan;   fill-opacity: 0.30; stroke: #00cccc; stroke-width: 1; stroke-dasharray: 4,2; }
.sar-flood     { fill: #3399ff; fill-opacity: 0.50; stroke: #0055cc; stroke-width: 1; stroke-dasharray: 4,2; }
.sar-moderate  { fill: #ffee00; fill-opacity: 0.50; stroke: #ccaa00; stroke-width: 1; stroke-dasharray: 4,2; }
.sar-significant { fill: #ff8800; fill-opacity: 0.50; stroke: #cc5500; stroke-width: 1; stroke-dasharray: 4,2; }
.sar-extreme   { fill: #cc0000; fill-opacity: 0.50; stroke: #880000; stroke-width: 1; stroke-dasharray: 4,2; }
</style>
"""


class SARLayerRenderer:
    """Render SAR-derived contextual layers in a dedicated Folium LayerGroup."""

    def __init__(self) -> None:
        self._sar_group = folium.FeatureGroup(
            name=_SAR_GROUP_NAME,
            show=False,  # Collapsed / hidden by default
        )

    def add_corridor_footprints(
        self,
        corridors_gdf: gpd.GeoDataFrame,
    ) -> "SARLayerRenderer":
        """Add SAR corridor footprint polygons."""
        if corridors_gdf is None or len(corridors_gdf) == 0:
            return self

        subgroup = folium.FeatureGroup(name="⚠️ SAR Corridor Footprints", show=False)

        for _, row in corridors_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            popup_html = self._build_corridor_popup(row)
            try:
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x: {
                        "fillColor": "cyan",
                        "color": "#00cccc",
                        "weight": 1,
                        "fillOpacity": 0.30,
                        "dashArray": "4,2",
                    },
                    tooltip=f"⚠️ SAR Corridor — {row.get('voltage_kv', '')} kV",
                    popup=folium.Popup(popup_html, max_width=400),
                ).add_to(subgroup)
            except Exception:
                continue

        subgroup.add_to(self._sar_group)
        return self

    def add_flood_extent(
        self,
        flood_gdf: Optional[gpd.GeoDataFrame] = None,
        flood_raster_path: Optional[Path] = None,
        event_date: str = "",
    ) -> "SARLayerRenderer":
        """Add SAR water/flood extent as hatched blue polygons."""
        if flood_gdf is None or len(flood_gdf) == 0:
            return self

        subgroup = folium.FeatureGroup(name="⚠️ SAR Flood / Water Extent", show=False)

        for _, row in flood_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            popup_html = self._build_flood_popup(row, event_date)
            try:
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x: {
                        "fillColor": "#3399ff",
                        "color": "#0055cc",
                        "weight": 1,
                        "fillOpacity": 0.50,
                        "dashArray": "4,2",
                    },
                    tooltip=f"⚠️ SAR Flood Extent — {event_date}",
                    popup=folium.Popup(popup_html, max_width=400),
                ).add_to(subgroup)
            except Exception:
                continue

        subgroup.add_to(self._sar_group)
        return self

    def add_change_mask(
        self,
        change_gdf: gpd.GeoDataFrame,
        tier: str = "moderate",
        before_date: str = "",
        after_date: str = "",
    ) -> "SARLayerRenderer":
        """Add SAR backscatter change mask polygons for a given tier."""
        if change_gdf is None or len(change_gdf) == 0:
            return self

        tier_colors = {
            "moderate":    ("#ffee00", "#ccaa00"),
            "significant": ("#ff8800", "#cc5500"),
            "extreme":     ("#cc0000", "#880000"),
        }
        fill_c, stroke_c = tier_colors.get(tier, ("#ffee00", "#ccaa00"))
        tier_label = tier.capitalize()
        subgroup = folium.FeatureGroup(
            name=f"⚠️ SAR Change ({tier_label})", show=False
        )

        for _, row in change_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            popup_html = self._build_change_popup(row, tier, before_date, after_date)
            try:
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x, fc=fill_c, sc=stroke_c: {
                        "fillColor": fc,
                        "color": sc,
                        "weight": 1,
                        "fillOpacity": 0.50,
                        "dashArray": "4,2",
                    },
                    tooltip=f"⚠️ SAR Change ({tier_label})",
                    popup=folium.Popup(popup_html, max_width=400),
                ).add_to(subgroup)
            except Exception:
                continue

        subgroup.add_to(self._sar_group)
        return self

    def add_resilience_risk(
        self,
        risk_gdf: gpd.GeoDataFrame,
    ) -> "SARLayerRenderer":
        """Add resilience risk point markers for each risk tier."""
        if risk_gdf is None or len(risk_gdf) == 0:
            return self

        for risk_label, color in _RISK_COLORS.items():
            tier_gdf = risk_gdf[risk_gdf.get("risk_label") == risk_label] if "risk_label" in risk_gdf.columns else gpd.GeoDataFrame()
            if len(tier_gdf) == 0:
                continue

            subgroup = folium.FeatureGroup(
                name=f"⚠️ Risk: {risk_label}", show=False
            )

            for _, row in tier_gdf.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                pt = geom if geom.geom_type == "Point" else geom.centroid
                popup_html = self._build_risk_popup(row)

                marker_kwargs = {"radius": 10, "color": color, "fill": True,
                                 "fill_color": color, "fill_opacity": 0.7}
                if risk_label == "CRITICAL":
                    # Pulsing effect via larger radius for critical
                    marker_kwargs["radius"] = 14

                folium.CircleMarker(
                    location=[pt.y, pt.x],
                    tooltip=f"⚠️ Risk: {risk_label}",
                    popup=folium.Popup(popup_html, max_width=450),
                    **marker_kwargs,
                ).add_to(subgroup)

            subgroup.add_to(self._sar_group)
        return self

    def attach_to_map(self, m: folium.Map) -> None:
        """Attach the SAR layer group to a Folium map."""
        self._sar_group.add_to(m)
        log.info("SARLayerRenderer: SAR layer group attached to map")

    # ------------------------------------------------------------------
    # Popup builders (all include mandatory disclaimer block)
    # ------------------------------------------------------------------

    def _build_corridor_popup(self, row) -> str:
        return (
            f"<b>⚠️ SAR Corridor</b><br>"
            f"Voltage: {row.get('voltage_kv', '')} kV<br>"
            f"Scene date: {row.get('sar_scene_date', 'N/A')}<br>"
            f"Mean VV: {row.get('mean_vv_db', 'N/A')} dB<br>"
            f"Buffer width: {row.get('buffer_width_m', '')} m<br>"
            f"SAR confidence: {row.get('sar_confidence', 'N/A')}<br>"
            f"<br><small><em>Source: {row.get('source', 'SAR_contextual_RS')}</em></small>"
            f"<br><pre style='font-size:9px;white-space:pre-wrap'>{_SAR_POPUP_BLOCK}</pre>"
        )

    def _build_flood_popup(self, row, event_date: str) -> str:
        return (
            f"<b>⚠️ SAR Flood Extent</b><br>"
            f"Detection date: {event_date}<br>"
            f"Threshold used: {row.get('threshold_db', -3.0)} dB<br>"
            f"CEMS validation: {row.get('cems_validated', 'Not validated')}<br>"
            f"SAR confidence: {row.get('sar_confidence', 'unvalidated')}<br>"
            f"<br><pre style='font-size:9px;white-space:pre-wrap'>{_SAR_POPUP_BLOCK}</pre>"
        )

    def _build_change_popup(self, row, tier: str, before: str, after: str) -> str:
        return (
            f"<b>⚠️ SAR Change ({tier.capitalize()})</b><br>"
            f"Date pair: {before} → {after}<br>"
            f"Mean ratio: {row.get('mean_ratio_db', '')} dB<br>"
            f"Max |ratio|: {row.get('max_abs_ratio_db', '')} dB<br>"
            f"Land cover: {row.get('land_cover', 'N/A')}<br>"
            f"SAR confidence: {row.get('sar_confidence', 'unvalidated')}<br>"
            f"<br><pre style='font-size:9px;white-space:pre-wrap'>{_SAR_POPUP_BLOCK}</pre>"
        )

    def _build_risk_popup(self, row) -> str:
        risk = row.get("risk_label", "")
        score = row.get("resilience_risk_score", "")
        sar_flood = row.get("sar_flood_exposure", "")
        sar_tier = row.get("sar_change_tier", "")
        eccc = row.get("in_official_floodzone", "")
        elev = row.get("elevation_m", "")
        saidi = row.get("saidi_percentile", "")

        return (
            f"<b>⚠️ Risk: {risk}</b><br>"
            f"Composite score: {score}<br>"
            f"<b>Score components:</b><br>"
            f"  SAR flood exposure: {sar_flood}<br>"
            f"  SAR change tier: {sar_tier}<br>"
            f"  In ECCC flood zone: {eccc}<br>"
            f"  Elevation: {elev} m<br>"
            f"  SAIDI percentile: {saidi}<br>"
            f"SAR confidence: {row.get('sar_confidence', 'N/A')}<br>"
            f"<br><small><em><i>Source: {row.get('source', 'SAR_contextual_RS')}</i></em></small>"
            f"<br><pre style='font-size:9px;white-space:pre-wrap'>{_SAR_POPUP_BLOCK}</pre>"
        )
