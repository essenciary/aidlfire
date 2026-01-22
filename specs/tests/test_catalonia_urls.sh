#!/bin/bash
# Script to test all Catalonia data URLs from the spec
# Usage: bash test_catalonia_urls.sh

echo "Testing Catalonia Data URLs..."
echo "================================"
echo ""

# Array of URLs to test
declare -a urls=(
    "https://analisi.transparenciacatalunya.cat/api/views/bks7-dkfd/rows.csv?accessType=DOWNLOAD|Generalitat Fire Perimeters CSV"
    "https://datos.gob.es/en/catalogo/a09002970-mapa-de-proteccion-civil-de-cataluna-riesgo-de-incendios-forestales|Generalitat Civil Protection Risk Map"
    "https://www.icgc.cat/en/Geoinformation-and-Maps/Maps/Dataset-Land-cover-map-CatLC|ICGC CatLC Dataset"
    "https://ftp.icgc.cat/descarregues/CatLCNet|ICGC CatLC FTP Download"
    "https://www.nature.com/articles/s41597-022-01674-y|CatLC Nature Paper"
    "https://previncat.ctfc.cat/en/index.html|PREVINCAT Server"
    "https://www.mdpi.com/2072-4292/12/24/4124|PREVINCAT MDPI Paper"
    "https://data.jrc.ec.europa.eu/dataset/7d5a5041-efac-4762-b9d1-c0b290ab2ce7|Copernicus EMS Catalonia Fire 2022"
    "https://pdxscholar.library.pdx.edu/esm_fac/215/|WUI Map Catalonia"
    "https://zenodo.org/records/6424854|Fire Sondes Data"
    "https://zenodo.org/records/14979237|Drought & Wildfire Variables"
    "https://agricultura.gencat.cat/ca/ambits/medi-natural/incendis-forestals/|Generalitat Fire Data"
)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success_count=0
fail_count=0
redirect_count=0

for url_pair in "${urls[@]}"; do
    IFS='|' read -r url name <<< "$url_pair"

    echo -n "Testing: $name... "

    # Use curl with follow redirects, timeout, and silent mode
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -L "$url" 2>&1)

    if [ $? -eq 0 ]; then
        if [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
            echo -e "${GREEN}✓ OK (HTTP $http_code)${NC}"
            ((success_count++))
        elif [ "$http_code" -ge 300 ] && [ "$http_code" -lt 400 ]; then
            echo -e "${YELLOW}⚠ Redirect (HTTP $http_code)${NC}"
            ((redirect_count++))
        else
            echo -e "${RED}✗ Failed (HTTP $http_code)${NC}"
            ((fail_count++))
        fi
    else
        echo -e "${RED}✗ Connection failed${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "================================"
echo "Summary:"
echo -e "  ${GREEN}Success: $success_count${NC}"
echo -e "  ${YELLOW}Redirects: $redirect_count${NC}"
echo -e "  ${RED}Failed: $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}All URLs are accessible!${NC}"
    exit 0
else
    echo -e "${RED}Some URLs failed. Please check manually.${NC}"
    exit 1
fi
