"""Generate synthetic test fixtures (run once to create test GeoTIFFs)."""
import os
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent

def create_mock_sar_raster():
    """Create a 64x64 synthetic Sentinel-1 VV amplitude GeoTIFF."""
    try:
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        rows, cols = 64, 64
        np.random.seed(42)
        data = np.random.uniform(0.05, 0.5, (rows, cols)).astype(np.float32)
        transform = from_bounds(-79.8, 43.5, -79.6, 43.7, cols, rows)

        out_path = FIXTURE_DIR / "mock_s1_vv.tif"
        with rasterio.open(
            str(out_path), "w",
            driver="GTiff", height=rows, width=cols,
            count=1, dtype="float32",
            crs="EPSG:4326", transform=transform,
        ) as dst:
            dst.write(data, 1)

        print(f"Created: {out_path}")
    except ImportError as e:
        print(f"Cannot create SAR fixture (missing deps): {e}")
        # Create a placeholder
        placeholder = FIXTURE_DIR / "mock_s1_vv.tif.placeholder"
        placeholder.write_text("# SAR fixture not generated — install rasterio\n")


if __name__ == "__main__":
    create_mock_sar_raster()
