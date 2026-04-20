"""Interactive Folium HTML map renderer with full toggle-layer support.

Renders all 11 target layers per specification. SAR layers are rendered
in a dedicated LayerGroup with mandatory disclaimers (via SARLayerRenderer).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import folium
from folium.plugins import FeatureGroupSubGroup, MeasureControl
import branca.colormap as cm

from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

# Voltage-based line styles
_LINE_STYLES = {
    "hv_500": {"color": "#cc0000", "weight": 3, "opacity": 0.9},
    "hv_230": {"color": "#ff6600", "weight": 2, "opacity": 0.9},
    "hv_115": {"color": "#ffcc00", "weight": 2, "opacity": 0.8},
    "mv_27k": {"color": "#3399ff", "weight": 1.5, "opacity": 0.8},
    "mv_25k": {"color": "#3399ff", "weight": 1.5, "opacity": 0.8},
    "mv_14k": {"color": "#6699ff", "weight": 1.5, "opacity": 0.7},
    "mv_8k":  {"color": "#99aaff", "weight": 1.5, "opacity": 0.7},
    "default": {"color": "#999999", "weight": 1, "opacity": 0.6},
}


class FoliumRenderer:
    """Build an interactive Folium multi-layer map of Ontario's electricity grid."""

    def __init__(self, bbox=None) -> None:
        cfg = load_settings()
        self._cfg = cfg
        self._bbox = bbox or cfg["region"].get("bbox", {})
        center_lat = (self._bbox.get("south", 43.5) + self._bbox.get("north", 43.75)) / 2
        center_lon = (self._bbox.get("west", -79.9) + self._bbox.get("east", -79.5)) / 2

        self.m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="CartoDB positron",
        )
        folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(self.m)
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            name="Satellite",
            attr="Esri",
        ).add_to(self.m)

        MeasureControl(primary_length_unit="meters").add_to(self.m)

    # ------------------------------------------------------------------
    # Grid infrastructure layers
    # ------------------------------------------------------------------

    def add_transmission_lines(
        self,
        lines_gdf: gpd.GeoDataFrame,
        layer_name: str = "Transmission Lines",
        show: bool = True,
    ) -> "FoliumRenderer":
        """Add transmission lines colored by voltage tier."""
        if lines_gdf is None or len(lines_gdf) == 0:
            return self

        group = folium.FeatureGroup(name=layer_name, show=show)

        for _, row in lines_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            tier = str(row.get("voltage_tier", "default"))
            style = _LINE_STYLES.get(tier, _LINE_STYLES["default"])
            voltage_kv = row.get("voltage_kv", "")
            operator = row.get("operator", "")
            rating = row.get("rating_mva", "")
            confidence = row.get("confidence", "")

            popup_html = (
                f"<b>Transmission Line</b><br>"
                f"Voltage: {voltage_kv} kV<br>"
                f"Tier: {tier}<br>"
                f"Operator: {operator}<br>"
                f"Rating: {rating} MVA<br>"
                f"Source: {row.get('source', '')}<br>"
                f"Confidence: {confidence}"
            )

            try:
                coords = self._geom_to_folium_coords(geom)
                folium.PolyLine(
                    locations=coords,
                    color=style["color"],
                    weight=style["weight"],
                    opacity=style["opacity"],
                    tooltip=f"{voltage_kv} kV line",
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(group)
            except Exception as exc:
                log.debug("Skipping line: %s", exc)

        group.add_to(self.m)
        return self

    def add_substations(
        self,
        substations_gdf: gpd.GeoDataFrame,
        layer_name: str = "Substations",
        color: str = "#cc0000",
        radius: int = 8,
        show: bool = True,
    ) -> "FoliumRenderer":
        """Add substation point markers."""
        if substations_gdf is None or len(substations_gdf) == 0:
            return self

        group = folium.FeatureGroup(name=layer_name, show=show)

        for _, row in substations_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            name = str(row.get("facility_name", row.get("name", "Substation")))
            voltage = row.get("voltage_kv", "")
            operator = row.get("operator", "")
            risk_label = row.get("risk_label", "")
            risk_score = row.get("resilience_risk_score", "")

            popup_html = (
                f"<b>{name}</b><br>"
                f"Voltage: {voltage} kV<br>"
                f"Operator: {operator}<br>"
                f"Source: {row.get('source', '')}<br>"
                f"Confidence: {row.get('confidence', '')}"
            )
            if risk_label:
                popup_html += f"<br>⚠️ Risk: {risk_label} (score={risk_score})"

            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=f"{name} ({voltage} kV)",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(group)

        group.add_to(self.m)
        return self

    def add_poles(
        self,
        poles_gdf: gpd.GeoDataFrame,
        inferred: bool = False,
        show: bool = False,
    ) -> "FoliumRenderer":
        """Add utility poles as tiny markers."""
        if poles_gdf is None or len(poles_gdf) == 0:
            return self

        color = "#ff8800" if inferred else "#888888"
        name = "Poles (ML inferred)" if inferred else "Poles (OSM)"
        group = folium.FeatureGroup(name=name, show=show)

        for _, row in poles_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            tooltip = f"Pole — confidence: {row.get('confidence', '')}"
            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=2,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=tooltip,
            ).add_to(group)

        group.add_to(self.m)
        return self

    def add_secondary_transformers(
        self,
        tx_gdf: gpd.GeoDataFrame,
        inferred: bool = False,
        show: bool = True,
    ) -> "FoliumRenderer":
        """Add secondary transformer locations as small squares."""
        if tx_gdf is None or len(tx_gdf) == 0:
            return self

        name = "Secondary Transformers (inferred)" if inferred else "Secondary Transformers (OSM)"
        group = folium.FeatureGroup(name=name, show=show)

        for _, row in tx_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            popup_html = (
                f"<b>Secondary Transformer</b><br>"
                f"Source: {row.get('source', '')}<br>"
                f"Confidence: {row.get('confidence', '')}"
            )
            folium.RegularPolygonMarker(
                location=[pt.y, pt.x],
                number_of_sides=4,
                radius=6,
                color="#00aa44",
                fill=True,
                fill_color="#00aa44",
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=250),
            ).add_to(group)

        group.add_to(self.m)
        return self

    def add_buildings(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        show: bool = False,
    ) -> "FoliumRenderer":
        """Add building footprint polygons."""
        if buildings_gdf is None or len(buildings_gdf) == 0:
            return self

        group = folium.FeatureGroup(name="Building Footprints", show=show)
        style = {"fillColor": "#aad4f5", "color": "#4488aa",
                 "weight": 0.5, "fillOpacity": 0.4}

        for _, row in buildings_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            try:
                popup_html = (
                    f"Address: {row.get('address', '')}<br>"
                    f"Assigned TX: {row.get('assigned_tx_id', '')}<br>"
                    f"LDC: {row.get('ldc_name', '')}"
                )
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda x, s=style: s,
                    tooltip=row.get("address", "Building"),
                    popup=folium.Popup(popup_html, max_width=250),
                ).add_to(group)
            except Exception:
                continue

        group.add_to(self.m)
        return self

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def add_sar_layers(
        self, sar_renderer: "SARLayerRenderer"
    ) -> "FoliumRenderer":
        """Attach SAR layer group from SARLayerRenderer."""
        sar_renderer.attach_to_map(self.m)
        return self

    def save(self, output_path: Path | str) -> Path:
        """Save the map as an HTML file and return the path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        folium.LayerControl(collapsed=False).add_to(self.m)
        self.m.save(str(output_path))
        log.info("FoliumRenderer: map saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geom_to_folium_coords(geom) -> list:
        """Convert a Shapely LineString to [[lat, lon], ...] for Folium."""
        if geom.geom_type == "LineString":
            return [[c[1], c[0]] for c in geom.coords]
        elif geom.geom_type == "MultiLineString":
            return [[c[1], c[0]] for line in geom.geoms for c in line.coords]
        return []
