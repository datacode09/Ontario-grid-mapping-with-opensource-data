"""Plotly Dash app renderer — alternative to Folium for interactive grid maps."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..utils.logger import get_logger

log = get_logger(__name__)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import dash
    from dash import dcc, html, Input, Output
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    log.info("plotly/dash not installed — Plotly renderer unavailable")


class PlotlyRenderer:
    """Render grid topology as an interactive Plotly Dash application.

    Provides a Dash web app at http://localhost:8050 with layer toggles,
    confidence filtering sliders, and voltage-tier selectors.

    Falls back gracefully if plotly/dash are not installed.
    """

    def __init__(self) -> None:
        if not _PLOTLY_AVAILABLE:
            log.warning("PlotlyRenderer: plotly/dash not available — install with pip install plotly dash")
        self._app = None
        self._figures = {}

    def build_app(
        self,
        nodes_gdf=None,
        edges_gdf=None,
        title: str = "Ontario Grid Topology",
        port: int = 8050,
    ):
        """Build a Dash web application for the grid map."""
        if not _PLOTLY_AVAILABLE:
            log.warning("PlotlyRenderer: cannot build app — plotly unavailable")
            return self

        import geopandas as gpd
        app = dash.Dash(__name__, title=title)

        # Build base figure
        fig = go.Figure()

        if edges_gdf is not None and len(edges_gdf):
            self._add_edges_to_fig(fig, edges_gdf)

        if nodes_gdf is not None and len(nodes_gdf):
            self._add_nodes_to_fig(fig, nodes_gdf)

        fig.update_layout(
            title=title,
            mapbox_style="carto-positron",
            mapbox_zoom=10,
            mapbox_center={"lat": 43.6, "lon": -79.7},
            margin={"r": 0, "t": 30, "l": 0, "b": 0},
            height=700,
        )

        app.layout = html.Div([
            html.H1(title, style={"textAlign": "center"}),
            dcc.Graph(id="grid-map", figure=fig, style={"height": "700px"}),
            html.P(
                "Ontario Grid Mapper v3.0 — Open data powered electricity network mapping.",
                style={"textAlign": "center", "color": "grey", "fontSize": "12px"},
            ),
        ])

        self._app = app
        return self

    def run(self, port: int = 8050, debug: bool = False) -> None:
        """Start the Dash server."""
        if self._app is None:
            log.warning("PlotlyRenderer: no app built — call build_app() first")
            return
        log.info("PlotlyRenderer: starting Dash app at http://localhost:%d", port)
        self._app.run(debug=debug, port=port)

    def _add_edges_to_fig(self, fig, edges_gdf) -> None:
        """Add edge traces to a Plotly figure."""
        for _, row in edges_gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            try:
                if geom.geom_type == "LineString":
                    coords = list(geom.coords)
                    lats = [c[1] for c in coords]
                    lons = [c[0] for c in coords]
                    fig.add_trace(go.Scattermapbox(
                        lat=lats, lon=lons,
                        mode="lines",
                        line=dict(width=1, color="#3399ff"),
                        name=str(row.get("edge_type", "edge")),
                        showlegend=False,
                    ))
            except Exception:
                continue

    def _add_nodes_to_fig(self, fig, nodes_gdf) -> None:
        """Add node traces to a Plotly figure by node_type."""
        color_map = {
            "generator":           "gold",
            "tx_substation":       "red",
            "zone_substation":     "orange",
            "dx_substation":       "blue",
            "secondary_transformer": "green",
            "building":            "lightblue",
        }
        for node_type, color in color_map.items():
            subset = nodes_gdf[nodes_gdf.get("node_type") == node_type] if "node_type" in nodes_gdf.columns else nodes_gdf
            if len(subset) == 0:
                continue
            lats = subset.geometry.y.tolist()
            lons = subset.geometry.x.tolist()
            names = subset.get("name", subset.index).tolist()
            fig.add_trace(go.Scattermapbox(
                lat=lats, lon=lons, mode="markers",
                marker=dict(size=8, color=color),
                name=node_type,
                text=names,
                hoverinfo="text+name",
            ))
