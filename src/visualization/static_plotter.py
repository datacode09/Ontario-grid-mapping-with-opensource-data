"""Static matplotlib plotter for validation and QA plots."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..utils.logger import get_logger

log = get_logger(__name__)


class StaticPlotter:
    """Produce static matplotlib figures for data QA and coverage inspection."""

    def plot_coverage_overview(
        self,
        layers: dict[str, gpd.GeoDataFrame],
        title: str = "Ontario Grid Coverage",
        output_path: Optional[Path] = None,
        figsize: tuple = (12, 10),
    ) -> plt.Figure:
        """Plot all active layers on a single static map for QA."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        layer_styles = {
            "transmission_lines": {"color": "red",   "linewidth": 1.5, "alpha": 0.9},
            "primary_feeders":    {"color": "blue",  "linewidth": 0.8, "alpha": 0.7},
            "substations":        {"color": "red",   "marker": "o",   "markersize": 6},
            "poles":              {"color": "grey",  "marker": ".",   "markersize": 2},
            "transformers":       {"color": "green", "marker": "s",   "markersize": 4},
            "buildings":          {"color": "#aad4f5", "alpha": 0.3},
        }

        legend_handles = []
        for layer_name, gdf in layers.items():
            if gdf is None or len(gdf) == 0:
                continue

            style = layer_styles.get(layer_name, {"color": "purple", "alpha": 0.5})
            try:
                gdf.plot(ax=ax, **{k: v for k, v in style.items()
                                   if k in ("color", "linewidth", "alpha", "markersize")})
                legend_handles.append(
                    mpatches.Patch(color=style.get("color", "purple"), label=layer_name)
                )
            except Exception as exc:
                log.debug("Could not plot layer %s: %s", layer_name, exc)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            log.info("StaticPlotter: saved to %s", output_path)

        return fig

    def plot_confidence_distribution(
        self,
        layers: dict[str, gpd.GeoDataFrame],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot confidence score distributions per layer."""
        import pandas as pd

        all_records = []
        for name, gdf in layers.items():
            if gdf is None or "confidence" not in gdf.columns:
                continue
            for val in gdf["confidence"].dropna():
                all_records.append({"layer": name, "confidence": float(val)})

        if not all_records:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No confidence data", ha="center")
            return fig

        df = pd.DataFrame(all_records)
        fig, ax = plt.subplots(figsize=(10, 5))

        layers_present = df["layer"].unique()
        for layer in layers_present:
            sub = df[df["layer"] == layer]["confidence"]
            sub.hist(ax=ax, alpha=0.5, bins=20, label=layer)

        ax.set_title("Confidence Score Distribution by Layer")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        return fig
