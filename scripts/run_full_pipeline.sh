#!/usr/bin/env bash
# =============================================================================
# Ontario Grid Mapper — Full Pipeline Run
# Runs all 7 pipeline steps for the Mississauga pilot area.
# =============================================================================
set -euo pipefail

REGION="${REGION:-Mississauga}"
CACHE_DIR="${CACHE_DIR:-data/raw}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs}"
LOG_FILE="${OUTPUT_DIR}/pipeline.log"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================"
echo " Ontario Grid Mapper v3.0 — Full Pipeline"
echo " Region: ${REGION}"
echo " Cache:  ${CACHE_DIR}"
echo " Output: ${OUTPUT_DIR}"
echo "======================================================"

# Step 1: Download all open data sources
echo ""
echo "[Step 1/7] Downloading open data sources..."
python -m src.main download \
    --region "${REGION}" \
    --cache-dir "${CACHE_DIR}" \
    2>&1 | tee -a "${LOG_FILE}"

# Step 2-4: Build grid graph with ML inference
echo ""
echo "[Step 2-4/7] Building grid topology graph..."
python -m src.main build-graph \
    --region "${REGION}" \
    --enable-ml-inference \
    --cache-dir "${CACHE_DIR}" \
    --output "${OUTPUT_DIR}/grid_graph.gpkg" \
    2>&1 | tee -a "${LOG_FILE}"

# Step 5: Service point assignment (run via Python API in production)
echo ""
echo "[Step 5/7] Service point assignment completed as part of build-graph."

# Step 6: SAR enrichment (only if enabled)
if [ "${SAR_ENABLED:-false}" = "true" ]; then
    echo ""
    echo "[Step 6/7] SAR enrichment (enabled)..."
    python -m src.main sar-enrich \
        --region "${REGION}" \
        --before-date "${SAR_BEFORE_DATE:-2024-03-01}" \
        --after-date "${SAR_AFTER_DATE:-2024-04-15}" \
        --output "${OUTPUT_DIR}/sar_products/" \
        2>&1 | tee -a "${LOG_FILE}"
else
    echo ""
    echo "[Step 6/7] SAR enrichment skipped (SAR_ENABLED=false)"
fi

# Step 7: Generate interactive map
echo ""
echo "[Step 7/7] Generating interactive map..."
python -m src.main visualize \
    --graph "${OUTPUT_DIR}/grid_graph.gpkg" \
    --sar-products "${OUTPUT_DIR}/sar_products/" \
    --output "${OUTPUT_DIR}/ontario_grid_map.html" \
    2>&1 | tee -a "${LOG_FILE}"

# Coverage audit
echo ""
echo "[Audit] Generating coverage report..."
python -m src.main audit \
    --region "${REGION}" \
    --include-sar \
    --output "${OUTPUT_DIR}/coverage_report.csv" \
    2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "======================================================"
echo " Pipeline complete! Outputs:"
echo "   Map:      ${OUTPUT_DIR}/ontario_grid_map.html"
echo "   Graph:    ${OUTPUT_DIR}/grid_graph.gpkg"
echo "   Report:   ${OUTPUT_DIR}/coverage_report.csv"
echo "   Log:      ${LOG_FILE}"
echo "======================================================"
