#!/bin/bash
# Script to test all Python library URLs from the spec
# Usage: bash test_python_library_urls.sh

echo "Testing Python Library URLs..."
echo "================================"
echo ""

# Array of URLs to test: name|pypi_url|github_url|docs_url
declare -a urls=(
    "sentinelsat|https://pypi.org/project/sentinelsat/|https://github.com/sentinelsat/sentinelsat|https://sentinelsat.readthedocs.io/"
    "pystac-client|https://pypi.org/project/pystac-client/|https://github.com/stac-utils/pystac-client|https://pystac-client.readthedocs.io/"
    "odc-stac|https://pypi.org/project/odc-stac/|https://github.com/opendatacube/odc-stac|https://odc-stac.readthedocs.io/"
    "sentinelhub|https://pypi.org/project/sentinelhub/|https://github.com/sentinel-hub/sentinelhub-py|https://sentinelhub-py.readthedocs.io/"
    "segmentation-models-pytorch|https://pypi.org/project/segmentation-models-pytorch/|https://github.com/qubvel/segmentation_models.pytorch|https://smp.readthedocs.io/"
    "wandb|https://pypi.org/project/wandb/|https://github.com/wandb/wandb|https://docs.wandb.ai/"
    "optuna|https://pypi.org/project/optuna/|https://github.com/optuna/optuna|https://optuna.readthedocs.io/"
    "dvc|https://pypi.org/project/dvc/|https://github.com/iterative/dvc|https://dvc.org/doc/"
    "torch|https://pypi.org/project/torch/|https://github.com/pytorch/pytorch|https://pytorch.org/docs/"
    "rasterio|https://pypi.org/project/rasterio/|https://github.com/rasterio/rasterio|https://rasterio.readthedocs.io/"
    "geopandas|https://pypi.org/project/geopandas/|https://github.com/geopandas/geopandas|https://geopandas.org/"
    "fastapi|https://pypi.org/project/fastapi/|https://github.com/tiangolo/fastapi|https://fastapi.tiangolo.com/"
    "streamlit|https://pypi.org/project/streamlit/|https://github.com/streamlit/streamlit|https://docs.streamlit.io/"
)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

total_success=0
total_fail=0

for url_entry in "${urls[@]}"; do
    IFS='|' read -r name pypi_url github_url docs_url <<< "$url_entry"

    echo "Testing: $name"
    echo "----------------------------------------"

    # Test PyPI
    echo -n "  PyPI: "
    pypi_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -L "$pypi_url" 2>&1)
    if [ $? -eq 0 ] && [ "$pypi_code" -ge 200 ] && [ "$pypi_code" -lt 400 ]; then
        echo -e "${GREEN}✓ OK (HTTP $pypi_code)${NC}"
        ((total_success++))
    else
        echo -e "${RED}✗ Failed (HTTP $pypi_code)${NC}"
        ((total_fail++))
    fi

    # Test GitHub
    echo -n "  GitHub: "
    github_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -L "$github_url" 2>&1)
    if [ $? -eq 0 ] && [ "$github_code" -ge 200 ] && [ "$github_code" -lt 400 ]; then
        echo -e "${GREEN}✓ OK (HTTP $github_code)${NC}"
        ((total_success++))
    else
        echo -e "${RED}✗ Failed (HTTP $github_code)${NC}"
        ((total_fail++))
    fi

    # Test Documentation
    echo -n "  Docs: "
    docs_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -L "$docs_url" 2>&1)
    if [ $? -eq 0 ] && [ "$docs_code" -ge 200 ] && [ "$docs_code" -lt 400 ]; then
        echo -e "${GREEN}✓ OK (HTTP $docs_code)${NC}"
        ((total_success++))
    else
        echo -e "${RED}✗ Failed (HTTP $docs_code)${NC}"
        ((total_fail++))
    fi

    echo ""
done

echo "================================"
echo "Summary:"
echo -e "  ${GREEN}Success: $total_success${NC}"
echo -e "  ${RED}Failed: $total_fail${NC}"
echo ""

if [ $total_fail -eq 0 ]; then
    echo -e "${GREEN}All URLs are accessible!${NC}"
    exit 0
else
    echo -e "${RED}Some URLs failed. Please check manually.${NC}"
    exit 1
fi
