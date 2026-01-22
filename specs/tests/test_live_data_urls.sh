#!/bin/bash
# Script to test all live data source URLs from section 2.2.2
# Usage: bash test_live_data_urls.sh

echo "Testing Live Data Source URLs (Section 2.2.2)..."
echo "================================================"
echo ""

# Array of URLs to test: name|url
declare -a urls=(
    "NASA FIRMS Portal|https://firms.modaps.eosdis.nasa.gov/"
    "NASA FIRMS API|https://firms.modaps.eosdis.nasa.gov/api/"
    "NASA FIRMS Data Download|https://firms.modaps.eosdis.nasa.gov/active_fire/"
    "EFFIS Portal|https://effis.jrc.ec.europa.eu/"
    "Eye on the Fire API|https://eyeonthefire.com/data-sources"
    "Copernicus EMS Data Portal|https://emergency.copernicus.eu/data"
    "JRC Dataset Portal|https://data.jrc.ec.europa.eu/"
    "NIFC Portal|https://www.nifc.gov/"
    "NIFC ArcGIS Map Services|https://www.nifc.gov/fire-information/nifc-maps"
    "PREVINCAT Server|https://previncat.ctfc.cat/en/index.html"
    "Ambee Fire API|https://www.getambee.com/api/fire"
    "Ambee Developer Portal|https://www.getambee.com/developers"
)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

total_success=0
total_fail=0
total_redirect=0

for url_entry in "${urls[@]}"; do
    IFS='|' read -r name url <<< "$url_entry"

    echo -e "${BLUE}Testing: $name${NC}"
    echo "  URL: $url"

    # Test with curl (with SSL verification, but allow insecure for testing)
    # Note: Use --insecure flag only for testing; remove in production
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -L --insecure "$url" 2>&1)
    exit_code=$?

    # Check for redirects
    redirect_url=$(curl -s -o /dev/null -w "%{redirect_url}" --max-time 15 -L --insecure "$url" 2>&1)

    if [ $exit_code -eq 0 ]; then
        if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 400 ]; then
            if [ -n "$redirect_url" ] && [ "$redirect_url" != "$url" ]; then
                echo -e "  Status: ${YELLOW}→ Redirect (HTTP $http_code)${NC}"
                echo -e "  Redirects to: $redirect_url"
                ((total_redirect++))
            else
                echo -e "  Status: ${GREEN}✓ OK (HTTP $http_code)${NC}"
                ((total_success++))
            fi
        elif [ "$http_code" -ge 400 ] && [ "$http_code" -lt 500 ]; then
            echo -e "  Status: ${RED}✗ Client Error (HTTP $http_code)${NC}"
            ((total_fail++))
        elif [ "$http_code" -ge 500 ]; then
            echo -e "  Status: ${RED}✗ Server Error (HTTP $http_code)${NC}"
            ((total_fail++))
        else
            echo -e "  Status: ${RED}✗ Unexpected (HTTP $http_code)${NC}"
            ((total_fail++))
        fi
    else
        echo -e "  Status: ${RED}✗ Connection Failed (Exit code: $exit_code)${NC}"
        ((total_fail++))
    fi

    echo ""
done

echo "================================================"
echo "Summary:"
echo -e "  ${GREEN}Success: $total_success${NC}"
echo -e "  ${YELLOW}Redirects: $total_redirect${NC}"
echo -e "  ${RED}Failed: $total_fail${NC}"
echo ""

if [ $total_fail -eq 0 ]; then
    echo -e "${GREEN}All URLs are accessible!${NC}"
    if [ $total_redirect -gt 0 ]; then
        echo -e "${YELLOW}Note: $total_redirect URL(s) redirect to different locations (this is normal for some services).${NC}"
    fi
    exit 0
else
    echo -e "${RED}Some URLs failed. Please check manually.${NC}"
    exit 1
fi
