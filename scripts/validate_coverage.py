#!/usr/bin/env python3
"""Coverage validation script — check data quality for all pipeline layers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import geopandas as gpd
import pandas as pd

from src.utils.config_loader import load_settings
from src.visualization.coverage_reporter import CoverageReporter


def main():
    cfg = load_settings()
    region = cfg["region"]["name"]
    output_dir = Path(cfg["paths"]["outputs"])

    reporter = CoverageReporter(region_name=region)
    layers = {}

    # Load cached layers
    cache_dir = Path(cfg["paths"]["cache_dir"])
    for subdir in ["oeb", "ieso", "osm", "statcan", "microsoft_buildings"]:
        for gpkg in (cache_dir / subdir).glob("*.gpkg"):
            try:
                import fiona
                for layer in fiona.listlayers(gpkg):
                    gdf = gpd.read_file(gpkg, layer=layer)
                    layers[f"{subdir}/{layer}"] = gdf
            except Exception as e:
                print(f"  Warning: could not load {gpkg}: {e}")

    df = reporter.generate_coverage_report(
        layers,
        output_path=output_dir / "coverage_report.csv"
    )
    reporter.print_summary(df)

    # Highlight empty layers
    empty = df[df["total_features"] == 0]
    if len(empty):
        print(f"\n⚠️  {len(empty)} empty layers — check downloads:")
        for _, row in empty.iterrows():
            print(f"   - {row['layer_name']}")
    else:
        print("\n✓ All layers have data.")


if __name__ == "__main__":
    main()
