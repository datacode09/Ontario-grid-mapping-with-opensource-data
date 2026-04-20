#!/usr/bin/env bash
# =============================================================================
# Ontario Grid Mapper — Standalone SAR Enrichment Module
# =============================================================================
set -euo pipefail

REGION="${REGION:-Mississauga}"
BEFORE_DATE="${SAR_BEFORE_DATE:-2024-03-01}"
AFTER_DATE="${SAR_AFTER_DATE:-2024-04-15}"
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/sar_products}"
USE_GEE="${USE_GEE:-false}"

echo "SAR Enrichment: ${REGION} | ${BEFORE_DATE} → ${AFTER_DATE}"
echo "GEE: ${USE_GEE}"

GEE_FLAG=""
if [ "${USE_GEE}" = "true" ]; then
    GEE_FLAG="--use-gee"
fi

python -m src.main sar-enrich \
    --region "${REGION}" \
    --before-date "${BEFORE_DATE}" \
    --after-date "${AFTER_DATE}" \
    ${GEE_FLAG} \
    --output "${OUTPUT_DIR}"

echo "SAR enrichment complete. Products in: ${OUTPUT_DIR}"
