# Ontario Grid Mapper v3.0

**Production-grade Python framework** for mapping Ontario's electricity network from high-voltage transmission down to the household service point, using exclusively open data sources and ML inference where utility GIS data is unavailable.

Includes an optional **SAR (Synthetic Aperture Radar) enrichment module** providing utility-corridor context, flood exposure screening near substations, and land-surface change detection as contextual resilience-risk annotations.

> **Important SAR Disclaimer:** All SAR-derived layers are contextual remote-sensing products derived from Sentinel-1 / PALSAR-2 backscatter data. They **do not represent exact electrical asset geometry** and must **not** be used for engineering or operational decisions without utility-authoritative verification.

---

## Architecture Overview

```
ontario-grid-mapper/
├── src/
│   ├── ingestion/      # 8 data fetchers covering 27 sources
│   ├── processing/     # CRS normalisation, voltage classification, deduplication
│   ├── topology/       # NetworkX DiGraph construction and export
│   ├── ml_inference/   # GridMapping CNN + gridfinder rural inference
│   ├── service_point/  # Address → transformer → grid path lookup
│   ├── sar/            # Optional SAR enrichment module
│   └── visualization/  # Folium interactive maps + coverage reports
├── config/
│   ├── settings.yaml
│   └── sar_settings.yaml
└── scripts/
    ├── run_full_pipeline.sh
    └── run_sar_enrichment.sh
```

---

## Data Sources (27 total)

### Authoritative Regulatory & Utility
| # | Source | Use |
|---|--------|-----|
| 1 | [OEB Distributor Service Areas](https://www.oeb.ca/open-data/electricity-and-natural-gas-distributors-service-areas) | LDC boundary for every Ontario address |
| 2 | [OEB Electricity RRR](https://www.oeb.ca/open-data) | SAIDI/SAIFI, customer counts per LDC |
| 3 | [IESO Transmission Facility Registry](https://www.ieso.ca/en/Power-Data/Data-Directory) | TX line ratings, substation locations |
| 4 | [IESO Generator Output & Capability](https://www.ieso.ca/en/Power-Data/Data-Directory) | Generation facility source nodes |
| 5 | [CER Interprovincial Transmission](https://www.cer-rec.gc.ca) | ON↔QC, ON↔MB, ON↔US tie lines |

### Geospatial / GIS
| # | Source | Use |
|---|--------|-----|
| 6 | [Ontario GeoHub (LIO)](https://geohub.lio.gov.on.ca/) | Roads, DEM, land use, hydrology, wetlands |
| 7 | [OpenStreetMap via Overpass API](https://overpass-api.de/) | All power infrastructure tags |
| 8 | [Open Infrastructure Map](https://openinframap.org) | Visual QA (not a download source) |
| 9 | [StatCan Dissemination Blocks](https://www12.statcan.gc.ca) | Finest geographic unit for service proxy |
| 10 | [StatCan Building Parcel Data](https://www150.statcan.gc.ca/) | Supplement to building footprints |
| 11 | [Microsoft Building Footprints](https://github.com/microsoft/CanadianBuildingFootprints) | ~600 MB Ontario building polygons |
| 12 | [City of Mississauga Open Data](https://www.mississauga.ca/services-and-programs/open-data/) | Address points for Peel Region |
| 13 | [City of Toronto Open Data](https://open.toronto.ca/) | Address points, partial distribution data |
| 14 | [Ontario Data Catalogue](https://data.ontario.ca/) | Critical facilities (schools, hospitals) |
| 15 | [NRCan CanVec Power Layer](https://open.canada.ca) | National-scale HV line validation |

### ML / Inference
| # | Source | Use |
|---|--------|-----|
| 16 | [Google Street View API](https://developers.google.com/maps/documentation/streetview/) | GridMapping CNN input images |
| 17 | [Stanford GridMapping](https://github.com/wangzhecheng/GridMapping) | Pole detection + feeder reconstruction |
| 18 | [gridfinder](https://github.com/carderne/gridfinder) | Rural MV line prediction |
| 19 | [NASA Black Marble VNP46A1](https://ladsweb.modaps.eosdis.nasa.gov) | Night-time lights for gridfinder |

### SAR & Earth Observation (Optional Module)
| # | Source | Use |
|---|--------|-----|
| 20 | [Copernicus Sentinel-1 GRD/RTC (CDSE)](https://dataspace.copernicus.eu/) | Primary SAR source (fallback) |
| 21 | [ASF DAAC Sentinel-1 RTC](https://search.asf.alaska.edu/) | Pre-processed RTC GeoTIFFs (preferred) |
| 22 | [Copernicus CEMS Rapid Mapping](https://emergency.copernicus.eu/mapping/) | Flood extent validation reference |
| 23 | [Google Earth Engine](https://developers.google.com/earth-engine/) | Server-side S1 time-series processing |
| 24 | [JAXA PALSAR-2 Annual Mosaic](https://www.eorc.jaxa.jp/ALOS/) | L-band for forested northern corridors |
| 25 | [Copernicus DEM GLO-30](https://spacedata.copernicus.eu/) | SAR terrain correction + elevation |
| 26 | [ECCC Flood Hazard Zones](https://open.canada.ca) | Official flood polygon validation |
| 27 | [Ontario Integrated Hydrology (OIHN)](https://geohub.lio.gov.on.ca/) | Watershed context for flood routing |

---

## Installation

### Core Installation

```bash
git clone https://github.com/datacode09/ontario-grid-mapping-with-opensource-data
cd ontario-grid-mapping-with-opensource-data

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### SAR Module (Optional)

```bash
pip install -r requirements-sar.txt
```

### API Credentials

```bash
cp .env.example .env
# Edit .env and add your API keys:
#   GOOGLE_STREETVIEW_API_KEY  (optional — ML inference)
#   MAPILLARY_CLIENT_TOKEN     (optional — open alternative)
#   GEE_PROJECT_ID             (optional — SAR via GEE)
#   ASF_USERNAME / ASF_PASSWORD (optional — SAR via ASF)
#   CDSE_USERNAME / CDSE_PASSWORD (optional — SAR via CDSE)
```

---

## Pipeline Usage

### Step 1: Download All Sources

```bash
python -m src.main download --region "Mississauga" --cache-dir data/raw/
```

### Step 2–4: Build Grid Graph

```bash
python -m src.main build-graph \
    --region "Mississauga" \
    --enable-ml-inference \
    --output data/outputs/mississ_graph.gpkg
```

### Step 5: Address Lookup

```bash
python -m src.main lookup \
    --address "7086 Tamar Mews, Mississauga, ON" \
    --radius 200 \
    --output results.json
```

### Step 6: SAR Enrichment (Optional)

```bash
# Enable SAR module first:
export SAR_ENABLED=true

python -m src.main sar-enrich \
    --region "Mississauga" \
    --before-date 2024-03-01 \
    --after-date 2024-04-15 \
    --use-gee \
    --output data/outputs/sar_products/
```

### Step 7: Generate Interactive Map

```bash
python -m src.main visualize \
    --graph data/outputs/mississ_graph.gpkg \
    --sar-products data/outputs/sar_products/ \
    --output data/outputs/ontario_grid_map.html \
    --layers transmission,zone_substations,primary_feeders,\
              secondary_transformers,buildings,\
              sar_corridors,sar_flood,sar_change,sar_risk
```

### Full Pipeline (One Command)

```bash
bash scripts/run_full_pipeline.sh
```

---

## SAR Module Enable/Disable

The SAR module is **disabled by default** (`sar.enabled: false` in `config/sar_settings.yaml`).

**Enable via config:**
```yaml
# config/sar_settings.yaml
sar:
  enabled: true
```

**Enable via environment variable:**
```bash
export SAR_ENABLED=true
```

**Disable:** Set back to `false` or unset the environment variable. When disabled, all SAR functions return `None` gracefully and no SAR layers are rendered in the map.

---

## Data Confidence Model

Every graph node and edge carries a `confidence` attribute (topology confidence):

| Score | Source |
|-------|--------|
| 1.00 | OEB / IESO authoritative |
| 0.90 | OSM with voltage tag confirmed |
| 0.75 | OSM, no voltage tag (inferred from context) |
| 0.65 | GridMapping ML inference (urban) |
| 0.45 | GridMapping ML inference (suburban) |
| 0.30 | gridfinder night-lights (rural) |
| 0.20 | Voronoi proxy only, no transformer located |

SAR products carry a **separate, independent** `sar_confidence` attribute:

| Value | Meaning |
|-------|---------|
| `validated` | SAR water extent confirmed by CEMS/ECCC polygon |
| `plausible` | SAR result consistent with LIO wetlands/hydrology |
| `unvalidated` | SAR result has no independent corroboration |
| `uncertain` | Low coherence, possible spring-melt artefact |

> ⚠️ **SAR confidence is NEVER merged with grid topology confidence.** They appear on separate fields in all outputs, APIs, and map elements.

---

## Ontario-Specific Notes

1. **LDC Fragmentation**: Ontario has ~60 licensed electricity distributors. Always resolve LDC from OEB boundary shapefiles — never assume Hydro One.
2. **Distribution Voltages**: Primary feeders at 8.0/13.8 kV (older) and 14.4/25 kV (modern). Legacy 2.4/4.16 kV in Toronto/Ottawa.
3. **Underground Distribution**: Erin Mills (post-1975 Mississauga) has extensive underground residential distribution.
4. **Spring Flood Priority**: Credit River, Grand River, Trent River, and Ottawa River systems cause recurring spring flood exposure for nearby substations. SAR flood detection prioritises April–May scenes.
5. **SAR Artefacts**: Sentinel-1 C-band is sensitive to wet snow and spring melt (March–April). Scenes during this window are flagged `sar_confidence="uncertain"`.
6. **Northern Forested Corridors**: PALSAR-2 L-band (Source 24) used for boreal forest corridor baselines where C-band has limited canopy penetration.

---

## Known Data Gaps & Limitations

- **OEB service area boundaries**: URL may change with OEB portal updates. Check `src/ingestion/oeb_fetcher.py` if download fails.
- **IESO registry**: Published as Excel — column names change between annual releases. The fetcher handles known variants.
- **OSM coverage**: Distribution feeder coverage is ~60% in suburban areas. ML inference fills gaps where imagery is available.
- **Microsoft Buildings**: Ontario file is ~600 MB compressed. Initial download may take 10–30 minutes.
- **GridMapping CNN**: Requires PyTorch and model weights (~200 MB). Download from [figshare](https://figshare.com/articles/dataset/22723171). Falls back to OSM-only if unavailable.
- **GEE**: Requires free registration and authentication. Falls back to local rasterio processing (ASF RTC downloads).
- **SAR spring melt artefacts**: Scenes from March–April may show false-positive flood signals due to wet snow. Set `artefact_suppression.flag_spring_melt: true` in `sar_settings.yaml`.

---

## Output Files (Mississauga Pilot)

| File | Description |
|------|-------------|
| `ontario_grid_map.html` | Interactive multi-layer Folium map |
| `grid_graph.gpkg` | NetworkX graph as GeoPackage (nodes + edges layers) |
| `building_transformer_lookup.parquet` | Building → transformer Parquet lookup |
| `coverage_report.csv` | Per-layer data coverage statistics |
| `resilience_risk_index.geojson` | Per-substation SAR resilience risk scores |
| `corridor_footprints.geojson` | SAR corridor buffer polygons |
| `water_extent_20240415.tif` | SAR flood detection output (COG) |
| `change_mask_20240301_20240415.tif` | SAR backscatter change mask (COG) |

---

## Tests

```bash
pytest tests/ -v
```

To create SAR test fixtures (requires rasterio):
```bash
python tests/fixtures/create_fixtures.py
```

---

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_data_exploration.ipynb` | Load and visualise all open data sources |
| `02_topology_validation.ipynb` | Graph statistics and hierarchy completeness |
| `03_ml_inference_qa.ipynb` | OSM vs ML pole coverage comparison |
| `04_service_point_demo.ipynb` | Address lookup pipeline demo |
| `05_sar_corridor_analysis.ipynb` | SAR backscatter over transmission corridors |
| `06_sar_resilience_scoring.ipynb` | Flood exposure and risk index demo |

---

## License

MIT License. See `LICENSE` for details.

Data source licenses vary — see individual source attributions above. All outputs must carry attribution for OEB, IESO, OpenStreetMap contributors, Statistics Canada, Microsoft, and (when SAR module is used) ESA Copernicus / JAXA.

---

*Data sources last verified: April 2026*  
*Primary pilot area: Mississauga / Peel Region (Erin Mills focus)*  
*Scalable to: full Province of Ontario*
