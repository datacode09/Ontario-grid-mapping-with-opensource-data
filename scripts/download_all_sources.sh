#!/usr/bin/env bash
# =============================================================================
# Download all 27 open data sources for Ontario Grid Mapper
# =============================================================================
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-data/raw}"
REGION="${REGION:-Mississauga}"

echo "Downloading all open data sources for: ${REGION}"

python -m src.main download \
    --region "${REGION}" \
    --cache-dir "${CACHE_DIR}" \
    --sources all

echo "Download complete. Cached in: ${CACHE_DIR}"
