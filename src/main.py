"""Ontario Grid Mapper — main CLI entrypoint.

Usage:
  python -m src.main download  --region Mississauga --cache-dir data/raw/
  python -m src.main build-graph --region Mississauga --enable-ml-inference
  python -m src.main lookup --address "7086 Tamar Mews, Mississauga, ON"
  python -m src.main sar-enrich --region Mississauga --before-date 2024-03-01
  python -m src.main visualize --graph data/outputs/mississ_graph.gpkg
  python -m src.main audit --region Mississauga --include-sar
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .utils.config_loader import load_settings, load_sar_settings, is_sar_enabled
from .utils.logger import get_logger

log = get_logger(__name__)


@click.group()
@click.version_option("3.0.0", prog_name="ontario-grid-mapper")
def cli():
    """Ontario Grid Mapper — map Ontario electricity network from open data."""
    pass


# ============================================================================
# Command: download
# ============================================================================

@cli.command("download")
@click.option("--region", default="Mississauga", help="Region name or 'ontario'")
@click.option("--cache-dir", default="data/raw", help="Cache directory for raw files")
@click.option("--force", is_flag=True, help="Force re-download even if cached")
@click.option("--sources", default="all",
              help="Comma-separated list of sources to download (all, oeb, ieso, osm, statcan, buildings, nrcan)")
def download(region, cache_dir, force, sources):
    """Download all open data sources for a region."""
    from .ingestion.oeb_fetcher import OEBFetcher
    from .ingestion.ieso_fetcher import IESOFetcher
    from .ingestion.osm_fetcher import OSMFetcher
    from .ingestion.statcan_fetcher import StatCanFetcher
    from .ingestion.microsoft_buildings import MicrosoftBuildingsFetcher
    from .ingestion.nrcan_fetcher import NRCanFetcher

    cfg = load_settings()
    bbox_cfg = cfg["region"].get("bbox", {})
    bbox = (
        bbox_cfg.get("west", -79.9),
        bbox_cfg.get("south", 43.45),
        bbox_cfg.get("east", -79.5),
        bbox_cfg.get("north", 43.75),
    )

    click.echo(f"Downloading open data for region: {region} (bbox: {bbox})")
    source_list = [s.strip().lower() for s in sources.split(",")]
    do_all = "all" in source_list

    if do_all or "oeb" in source_list:
        click.echo("  [1/6] OEB service areas...")
        try:
            OEBFetcher(Path(cache_dir) / "oeb").fetch_service_areas(force=force)
            click.echo("  ✓ OEB service areas")
        except Exception as e:
            click.echo(f"  ✗ OEB failed: {e}")

    if do_all or "ieso" in source_list:
        click.echo("  [2/6] IESO transmission registry...")
        try:
            IESOFetcher(Path(cache_dir) / "ieso").fetch_transmission_registry(force=force)
            IESOFetcher(Path(cache_dir) / "ieso").fetch_generator_capability(force=force)
            click.echo("  ✓ IESO data")
        except Exception as e:
            click.echo(f"  ✗ IESO failed: {e}")

    if do_all or "osm" in source_list:
        click.echo("  [3/6] OSM power infrastructure...")
        try:
            result = OSMFetcher(Path(cache_dir) / "osm").fetch_power_infrastructure(bbox, force=force)
            total = sum(len(v) for v in result.values())
            click.echo(f"  ✓ OSM: {total} features")
        except Exception as e:
            click.echo(f"  ✗ OSM failed: {e}")

    if do_all or "statcan" in source_list:
        click.echo("  [4/6] Statistics Canada dissemination blocks...")
        try:
            StatCanFetcher(Path(cache_dir) / "statcan").fetch_dissemination_blocks(bbox, force=force)
            click.echo("  ✓ StatCan DB")
        except Exception as e:
            click.echo(f"  ✗ StatCan failed: {e}")

    if do_all or "buildings" in source_list:
        click.echo("  [5/6] Microsoft building footprints...")
        try:
            bldgs = MicrosoftBuildingsFetcher(Path(cache_dir) / "microsoft_buildings").fetch_buildings(bbox, force=force)
            click.echo(f"  ✓ Buildings: {len(bldgs)} footprints")
        except Exception as e:
            click.echo(f"  ✗ Buildings failed: {e}")

    if do_all or "nrcan" in source_list:
        click.echo("  [6/6] NRCan CanVec power layer...")
        try:
            NRCanFetcher(Path(cache_dir) / "nrcan").fetch_canvec_power(bbox, force=force)
            click.echo("  ✓ NRCan CanVec")
        except Exception as e:
            click.echo(f"  ✗ NRCan failed: {e}")

    click.echo(f"\nDownload complete. Cache: {cache_dir}")


# ============================================================================
# Command: build-graph
# ============================================================================

@cli.command("build-graph")
@click.option("--region", default="Mississauga")
@click.option("--enable-ml-inference", is_flag=True, default=False)
@click.option("--output", default="data/outputs/grid_graph.gpkg")
@click.option("--cache-dir", default="data/raw")
def build_graph(region, enable_ml_inference, output, cache_dir):
    """Build full grid topology graph (transmission through distribution)."""
    from .ingestion.oeb_fetcher import OEBFetcher
    from .ingestion.ieso_fetcher import IESOFetcher
    from .ingestion.osm_fetcher import OSMFetcher
    from .processing.crs_normalizer import CRSNormalizer
    from .processing.voltage_classifier import VoltageClassifier
    from .processing.operator_tagger import OperatorTagger
    from .processing.substation_resolver import SubstationResolver
    from .topology.graph_builder import GraphBuilder
    from .topology.hierarchy_linker import HierarchyLinker
    from .topology.graph_exporter import GraphExporter

    cfg = load_settings()
    bbox = _bbox_from_config(cfg)

    click.echo(f"Building grid graph for {region}...")

    # Step 1: Ingest
    click.echo("  Ingesting data...")
    oeb = OEBFetcher(Path(cache_dir) / "oeb")
    ieso = IESOFetcher(Path(cache_dir) / "ieso")
    osm = OSMFetcher(Path(cache_dir) / "osm")

    ldc_gdf = oeb.fetch_service_areas()
    tx_gdf = ieso.fetch_transmission_registry()
    gen_gdf = ieso.fetch_generator_capability()
    osm_data = osm.fetch_power_infrastructure(bbox)

    # Step 2: Process
    click.echo("  Processing layers...")
    norm = CRSNormalizer()
    vc = VoltageClassifier()
    tagger = OperatorTagger(ldc_boundaries=ldc_gdf)
    resolver = SubstationResolver()

    if len(tx_gdf):
        tx_gdf = vc.classify(tx_gdf)
        tx_gdf = tagger.tag(tx_gdf)

    osm_lines = osm_data.get("lines", _empty_gdf())
    osm_subs = osm_data.get("substations", _empty_gdf())

    if len(osm_lines):
        osm_lines = vc.classify(osm_lines)
        osm_lines = tagger.tag(osm_lines)

    substations = resolver.resolve(
        osm_gdf=osm_subs,
        ieso_gdf=tx_gdf,
    )

    # Step 3: Build graph
    click.echo("  Building topology graph...")
    builder = GraphBuilder()
    if len(gen_gdf):
        builder.add_generators(gen_gdf)
    if len(substations):
        tx_ss = substations[substations.get("voltage_kv", 0) >= 100] if "voltage_kv" in substations.columns else substations
        builder.add_tx_substations(tx_ss)

    if len(osm_lines):
        builder.add_tx_lines(osm_lines[osm_lines.get("voltage_tier", "").str.startswith("hv")])
        builder.add_primary_feeders(osm_lines[~osm_lines.get("voltage_tier", "").str.startswith("hv")])

    G = builder.G
    linker = HierarchyLinker(G)
    linker.link_all()

    # Export
    click.echo(f"  Exporting to {output}...")
    exporter = GraphExporter(G)
    exporter.to_gpkg(Path(output))

    click.echo(
        f"\nGraph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    click.echo(f"Saved to: {output}")


# ============================================================================
# Command: lookup
# ============================================================================

@cli.command("lookup")
@click.option("--address", required=True, help="Civic address to look up")
@click.option("--radius", default=200.0, help="Transformer search radius (metres)")
@click.option("--output", default=None, help="Output JSON file path")
@click.option("--cache-dir", default="data/raw")
def lookup(address, radius, output, cache_dir):
    """Look up the probable serving transformer for an address."""
    from .ingestion.oeb_fetcher import OEBFetcher
    from .ingestion.osm_fetcher import OSMFetcher
    from .service_point.geocoder import Geocoder
    from .service_point.ldc_resolver import LDCResolver

    click.echo(f"Looking up: {address}")

    oeb = OEBFetcher(Path(cache_dir) / "oeb")
    ldc_gdf = oeb.fetch_service_areas()
    geocoder = Geocoder()
    point = geocoder.geocode(address)

    if point is None:
        click.echo("ERROR: Could not geocode address")
        sys.exit(1)

    result = {"address": address, "lat": point.y, "lon": point.x}

    if len(ldc_gdf):
        resolver = LDCResolver(ldc_gdf)
        ldc_info = resolver.resolve_point(point.x, point.y)
        if ldc_info:
            result["ldc_name"] = ldc_info["ldc_name"]

    result_str = json.dumps(result, indent=2)
    click.echo(result_str)

    if output:
        Path(output).write_text(result_str)
        click.echo(f"\nResult saved to: {output}")


# ============================================================================
# Command: sar-enrich
# ============================================================================

@cli.command("sar-enrich")
@click.option("--region", default="Mississauga")
@click.option("--before-date", required=True, help="Reference date YYYY-MM-DD")
@click.option("--after-date", required=True, help="Event date YYYY-MM-DD")
@click.option("--use-gee", is_flag=True, default=False, help="Use GEE processing path")
@click.option("--output", default="data/outputs/sar_products/")
def sar_enrich(region, before_date, after_date, use_gee, output):
    """Run SAR enrichment module (standalone)."""
    if not is_sar_enabled():
        click.echo(
            "SAR module is disabled. Set SAR_ENABLED=true in environment "
            "or sar:enabled: true in config/sar_settings.yaml"
        )
        sys.exit(0)

    from .sar.sar_fetcher import SARFetcher
    from .sar.corridor_extractor import extract_corridor_footprints
    from .sar.sar_exporter import SARExporter

    cfg = load_settings()
    bbox = _bbox_from_config(cfg)
    from .utils.geometry import bbox_to_polygon
    aoi = bbox_to_polygon(*bbox)

    click.echo(f"SAR enrichment for {region}: {before_date} → {after_date}")

    fetcher = SARFetcher(Path(output) / "sentinel1_rtc")
    before_scenes = fetcher.fetch_sentinel1_rtc(aoi, before_date, before_date)
    after_scenes  = fetcher.fetch_sentinel1_rtc(aoi, after_date, after_date)

    click.echo(f"  Downloaded: {len(before_scenes)} before, {len(after_scenes)} after scenes")

    if use_gee:
        from .sar.gee_processor import run_gee_change_detection
        tasks = run_gee_change_detection(aoi, before_date, before_date, after_date, after_date)
        if tasks:
            click.echo(f"  GEE tasks submitted: {tasks}")
        else:
            click.echo("  GEE: no tasks submitted (check GEE credentials)")

    exporter = SARExporter(Path(output))
    manifest = exporter.export_all()
    click.echo(f"\nSAR enrichment complete. Products: {list(manifest.keys())}")


# ============================================================================
# Command: visualize
# ============================================================================

@cli.command("visualize")
@click.option("--graph", default="data/outputs/grid_graph.gpkg", help="Graph GeoPackage")
@click.option("--sar-products", default=None, help="SAR products directory")
@click.option("--output", default="data/outputs/ontario_grid_map.html")
@click.option("--layers", default="all")
def visualize(graph, sar_products, output, layers):
    """Generate interactive Folium HTML map."""
    from .visualization.folium_renderer import FoliumRenderer
    from .visualization.sar_layer_renderer import SARLayerRenderer
    import geopandas as gpd

    cfg = load_settings()
    renderer = FoliumRenderer()

    # Load graph layers if available
    if Path(graph).exists():
        try:
            import fiona
            available_layers = fiona.listlayers(graph)
            for layer_name in available_layers:
                gdf = gpd.read_file(graph, layer=layer_name)
                if "node_type" in gdf.columns:
                    # Add nodes by type
                    tx_nodes = gdf[gdf["node_type"].isin(["tx_substation", "zone_substation"])]
                    if len(tx_nodes):
                        renderer.add_substations(tx_nodes, f"Substations ({layer_name})")
                elif "edge_type" in gdf.columns:
                    renderer.add_transmission_lines(gdf, "Transmission Lines")
        except Exception as e:
            click.echo(f"  Warning: could not load graph: {e}")

    # Add SAR layers if available and enabled
    if sar_products and is_sar_enabled():
        sar_renderer = SARLayerRenderer()
        sar_dir = Path(sar_products)

        corridor_path = sar_dir / "corridor_footprints.geojson"
        if corridor_path.exists():
            corridors = gpd.read_file(corridor_path)
            sar_renderer.add_corridor_footprints(corridors)

        risk_path = sar_dir / "resilience_risk_index.geojson"
        if risk_path.exists():
            risk_gdf = gpd.read_file(risk_path)
            sar_renderer.add_resilience_risk(risk_gdf)

        renderer.add_sar_layers(sar_renderer)

    out_path = renderer.save(output)
    click.echo(f"Map saved to: {out_path}")


# ============================================================================
# Command: audit
# ============================================================================

@cli.command("audit")
@click.option("--region", default="Mississauga")
@click.option("--include-sar", is_flag=True, default=False)
@click.option("--output", default="data/outputs/coverage_report.csv")
@click.option("--cache-dir", default="data/raw")
def audit(region, include_sar, output, cache_dir):
    """Generate coverage and SAR audit report."""
    from .visualization.coverage_reporter import CoverageReporter
    import geopandas as gpd

    reporter = CoverageReporter(region_name=region)
    layers = {}

    # Try to load cached layers for reporting
    oeb_path = Path(cache_dir) / "oeb" / "oeb_service_areas.gpkg"
    if oeb_path.exists():
        try:
            layers["oeb_service_areas"] = gpd.read_file(oeb_path)
        except Exception:
            pass

    sar_manifest = {}
    if include_sar and is_sar_enabled():
        sar_dir = Path("data/processed/sar_derived")
        for f in sar_dir.glob("*.geojson"):
            sar_manifest[f.stem] = str(f)

    df = reporter.generate_coverage_report(layers, sar_manifest, Path(output))
    reporter.print_summary(df)
    click.echo(f"\nCoverage report saved to: {output}")


# ============================================================================
# Helpers
# ============================================================================

def _bbox_from_config(cfg: dict) -> tuple:
    bbox = cfg["region"].get("bbox", {})
    return (
        bbox.get("west", -79.9),
        bbox.get("south", 43.45),
        bbox.get("east", -79.5),
        bbox.get("north", 43.75),
    )


def _empty_gdf():
    import geopandas as gpd
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")


if __name__ == "__main__":
    cli()
