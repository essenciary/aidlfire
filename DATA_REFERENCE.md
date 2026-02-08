# CEMS Wildfire Dataset - Data Reference Guide

This document provides a comprehensive reference for understanding and using the CEMS Wildfire Dataset. It covers the image files, metadata CSVs, and how all components connect together.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Dataset Overview](#dataset-overview)
- [Directory Structure](#directory-structure)
- [Image Files](#image-files)
  - [Satellite Imagery (S2L2A)](#satellite-imagery-s2l2a)
  - [Fire Masks (DEL, GRA)](#fire-masks-del-gra)
  - [Cloud Mask (CM)](#cloud-mask-cm)
  - [Land Cover Maps](#land-cover-maps)
- [Metadata Files](#metadata-files)
  - [satelliteData.csv](#satellitedatacsv)
  - [datasetPreConfigured.csv](#datasetpreconfiguredcsv)
  - [cloudCoverage.csv](#cloudcoveragecsv)
  - [JSON Metadata](#json-metadata)
- [Naming Convention](#naming-convention)
- [How Everything Connects](#how-everything-connects)
- [Common Use Cases](#common-use-cases)
- [Data Splits](#data-splits)

---

## Quick Start

**Want to train a fire segmentation model?** Here's what you need:

1. **Input images**: `*_S2L2A.tif` - 12-band Sentinel-2 imagery
2. **Target masks**: `*_DEL.tif` (binary) or `*_GRA.tif` (severity levels)
3. **Cloud filtering**: Use `*_CM.tif` to exclude cloudy pixels
4. **Metadata**: `csv_files/satelliteData.csv` for coordinates, dates, and statistics

---

## Dataset Overview

| Property | Value |
|----------|-------|
| **Fire Events** | ~275 unique EMSR activations |
| **Total Images** | 500+ georeferenced tiles |
| **Time Period** | June 2017 - April 2023 |
| **Satellite** | Sentinel-2 L2A (atmospherically corrected) |
| **Resolution** | ~10 meters per pixel |
| **Coverage** | Europe (Portugal, Spain, Italy, Greece, etc.) |
| **Format** | GeoTIFF (.tif) + PNG previews |

---

## Directory Structure

```
CEMS-Wildfire-Dataset/
├── csv_files/                      # Metadata files
│   ├── satelliteData.csv           # Main metadata (coordinates, dates, stats)
│   ├── datasetPreConfigured.csv    # Activation list with date ranges
│   └── cloudCoverage.csv           # Cloud/burn statistics per image
│
├── wildfires-cems/data/            # Image data (from HuggingFace)
│   ├── train/train/                # Training split
│   │   └── EMSR{XXX}/AOI{YY}/EMSR{XXX}_AOI{YY}_{ZZ}/
│   │       ├── *_S2L2A.tif         # Satellite image (12 bands)
│   │       ├── *_DEL.tif           # Delineation mask (binary)
│   │       ├── *_GRA.tif           # Grading mask (severity)
│   │       ├── *_CM.tif            # Cloud mask
│   │       ├── *_ESA_LC.tif        # Land cover (ESA WorldCover)
│   │       ├── *_S2L2A.json        # Acquisition metadata
│   │       └── *.png               # Preview images
│   ├── val/val/                    # Validation split
│   ├── test/test/                  # Test split
│   └── cat/                        # Catalonia data (regional subset)
│
└── fire-pipeline/                  # Processing tools
```

---

## Image Files

### Satellite Imagery (S2L2A)

**File**: `EMSR{XXX}_AOI{YY}_{ZZ}_S2L2A.tif`

The primary input data - 12-band Sentinel-2 Level-2A imagery.

| Band Index | Band Name | Wavelength (nm) | Resolution | Description |
|------------|-----------|-----------------|------------|-------------|
| 0 | B01 | 443 (Coastal) | 60m | Aerosol detection |
| 1 | B02 | 490 (Blue) | 10m | Water, coastal |
| 2 | B03 | 560 (Green) | 10m | Vegetation vigor |
| 3 | B04 | 665 (Red) | 10m | Chlorophyll absorption |
| 4 | B05 | 705 (Red Edge 1) | 20m | Vegetation stress |
| 5 | B06 | 740 (Red Edge 2) | 20m | Vegetation stress |
| 6 | B07 | 783 (Red Edge 3) | 20m | Leaf area index |
| 7 | B08 | 842 (NIR) | 10m | Vegetation biomass |
| 8 | B8A | 865 (NIR narrow) | 20m | Vegetation/water |
| 9 | B09 | 945 (Water Vapor) | 60m | Atmospheric correction |
| 10 | B11 | 1610 (SWIR1) | 20m | **Fire/burn detection** |
| 11 | B12 | 2190 (SWIR2) | 20m | **Fire/burn detection** |

**Key bands for fire detection**: B11 and B12 (SWIR) are most sensitive to fire and burn scars.

**Data format**:
- Type: Float32
- Value range: 0.0 - 1.0 (surface reflectance)
- CRS: EPSG:4326 (WGS84)

### Fire Masks (DEL, GRA)

#### DEL - Delineation Mask (Binary)

**File**: `*_DEL.tif`

Binary classification of burned area.

| Value | Meaning |
|-------|---------|
| 0 | Not burned |
| 1 | Burned |

**Use for**: Binary fire segmentation tasks.

#### GRA - Grading Mask (Severity)

**File**: `*_GRA.tif`

Multi-class severity assessment of fire damage.

| Value | Severity | Description |
|-------|----------|-------------|
| 0 | No damage | Unaffected area |
| 1 | Negligible | Minimal visible damage |
| 2 | Moderate | Possibly damaged |
| 3 | High | Clearly damaged |
| 4 | Destroyed | Complete destruction |

**Use for**: Multi-class severity segmentation or regression tasks.

**Source**: Expert annotations from CEMS Rapid Mapping service.

### Cloud Mask (CM)

**File**: `*_CM.tif`

Cloud/shadow mask generated using CloudSen12.

| Value | Meaning |
|-------|---------|
| 0 | Clear sky |
| 1 | Cloud (opaque) |
| 2 | Light cloud / haze / smoke |
| 3 | Cloud shadow |

**Use for**: Filtering out unreliable pixels during training and inference.

### Land Cover Maps

Three land cover products are included:

#### ESA WorldCover 2020

**File**: `*_ESA_LC.tif`

| Value | Class |
|-------|-------|
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare/sparse vegetation |
| 70 | Snow and ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss and lichen |

#### ESRI 10-Class LULC

**File**: `*_Esri10_LC.tif`

10-class land use/land cover classification.

#### Annual 9-Class

**File**: `*_Annual9_LC.tif`

9-class annual land cover product.

**Use for**: Stratified sampling, understanding fire behavior by land type, or multi-task learning.

---

## Metadata Files

### satelliteData.csv

**Location**: `csv_files/satelliteData.csv`
**Rows**: ~560 (one per image tile)

The primary metadata file containing geographic, temporal, and statistical information.

| Column | Type | Description |
|--------|------|-------------|
| `EMSR` | string | Emergency activation ID (e.g., "EMSR207") |
| `AOI` | string | Area of Interest (e.g., "AOI01") |
| `folder` | string | Data quality category: "optimal" or "colomba" |
| `folderPath` | string | Relative path to image directory |
| `activationDate` | datetime | When CEMS was activated for this fire |
| `interval_startDate` | datetime | Start of Sentinel-2 search window |
| `interval_endDate` | datetime | End of Sentinel-2 search window |
| `post_fire_acquisition` | string | Actual satellite acquisition timestamp |
| `GRA` | bool (0/1) | Grading mask available |
| `DEL` | bool (0/1) | Delineation mask available |
| `FEP` | bool (0/1) | First Estimate Product available |
| `left_Long` | float | Bounding box: left longitude |
| `bottom_Lat` | float | Bounding box: bottom latitude |
| `right_Long` | float | Bounding box: right longitude |
| `top_Lat` | float | Bounding box: top latitude |
| `centerBoxLong` | float | Center longitude |
| `centerBoxLat` | float | Center latitude |
| `resolution_x` | float | Pixel size in X (degrees) |
| `resolution_y` | float | Pixel size in Y (degrees) |
| `height` | int | Image height in pixels |
| `width` | int | Image width in pixels |
| `pixelBurned` | int | Number of burned pixels in DEL mask |
| `country` | string | Country name (Portugal, Spain, Italy, etc.) |
| `koppen_group` | string | Köppen climate classification group |
| `koppen_subgroup` | string | Köppen climate subgroup |

**Example query** - Find all fires in Spain:
```python
import pandas as pd
df = pd.read_csv("csv_files/satelliteData.csv")
spain_fires = df[df["country"] == "Spain"]
```

### datasetPreConfigured.csv

**Location**: `csv_files/datasetPreConfigured.csv`
**Rows**: ~274 (one per EMSR+AOI combination)

High-level activation information for downloading/configuring the dataset.

| Column | Type | Description |
|--------|------|-------------|
| `EMSR` | string | Emergency activation ID |
| `AOI` | string | Area of Interest |
| `folderType` | string | "optimal", "subOptimal_cloudy", or "subOptimal_FEP" |
| `folderPath` | string | Base folder path |
| `activationDate` | datetime | When CEMS was activated |
| `suggested_startDate` | datetime | Recommended search window start |
| `suggested_endDate` | datetime | Recommended search window end |

### cloudCoverage.csv

**Location**: `csv_files/cloudCoverage.csv`
**Rows**: ~998 (entries for both DEL and GRA masks)

Statistics about cloud cover and burn extent per image.

| Column | Type | Description |
|--------|------|-------------|
| `EMSR_AOI` | string | Combined identifier (e.g., "EMSR230_AOI01_01") |
| `folderPath` | string | Path to image directory |
| `startDate` | datetime | Search window start |
| `endDate` | datetime | Search window end |
| `height` | int | Image height |
| `width` | int | Image width |
| `sizeImage` | int | Total pixels (height × width) |
| `burnedPixel` | int | Pixels with burn mask > 0 |
| `cloudPixel` | int | Pixels with cloud mask > 0 |
| `countOverlap` | int | Pixels where burn and cloud overlap |
| `percentageCloud` | float | Fraction of image with clouds |
| `percentageOverlap` | float | Fraction of burned area obscured by clouds |
| `Type` | string | "DEL" or "GRA" |

### JSON Metadata

**File**: `*_S2L2A.json`

Contains Sentinel Hub API request details and acquisition metadata.

```json
{
  "payload": {
    "input": {
      "bounds": {
        "bbox": [lon_min, lat_min, lon_max, lat_max],
        "properties": {"crs": "EPSG:4326"}
      },
      "data": [{
        "dataFilter": {
          "maxCloudCoverage": 10,
          "timeRange": {"from": "...", "to": "..."}
        },
        "type": "sentinel-2-l2a"
      }]
    },
    "output": {"height": 580, "width": 589},
    "acquisition_date": ["2017/08/15_09:31:03"]
  },
  "timestamp": "2023-04-23T01:51:47.114008"
}
```

**Key fields**:
- `bbox`: Exact bounding box coordinates
- `acquisition_date`: When the satellite captured the image
- `timeRange`: Search window used to find the image

---

## Naming Convention

```
EMSR382_AOI01_02_S2L2A.tif
│      │     │  └── File type (S2L2A, DEL, GRA, CM, etc.)
│      │     └── Tile number (large AOIs are split into tiles)
│      └── Area of Interest (multiple AOIs per activation)
└── Emergency activation ID (unique per fire event)
```

**EMSR codes** are assigned by CEMS when a fire emergency is reported. Lower numbers are older events.

---

## How Everything Connects

```
                    ┌─────────────────────────────────────────┐
                    │        datasetPreConfigured.csv         │
                    │   (activation dates, download config)   │
                    └─────────────────┬───────────────────────┘
                                      │
                                      │ EMSR + AOI
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          satelliteData.csv                              │
│  (coordinates, country, climate, resolution, burned pixels, dates)      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  │ folderPath
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Image Directory                                  │
│  wildfires-cems/data/{split}/{split}/EMSR{XXX}/AOI{YY}/EMSR{XXX}_...   │
│                                                                         │
│   ├── *_S2L2A.tif    ◄─── Input (satellite imagery)                    │
│   ├── *_S2L2A.json   ◄─── Acquisition metadata                         │
│   ├── *_DEL.tif      ◄─── Target (binary mask)                         │
│   ├── *_GRA.tif      ◄─── Target (severity mask)                       │
│   ├── *_CM.tif       ◄─── Filter (cloud mask)                          │
│   └── *_ESA_LC.tif   ◄─── Context (land cover)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  │ EMSR_AOI identifier
                                  ▼
                    ┌─────────────────────────────────────────┐
                    │          cloudCoverage.csv              │
                    │   (cloud %, burn %, overlap stats)      │
                    └─────────────────────────────────────────┘
```

**Joining data example**:

```python
import pandas as pd

# Load metadata
satellite_df = pd.read_csv("csv_files/satelliteData.csv")
cloud_df = pd.read_csv("csv_files/cloudCoverage.csv")

# Create matching identifier in satellite_df
satellite_df["EMSR_AOI"] = satellite_df.apply(
    lambda r: f"{r['EMSR']}_{r['AOI']}_01", axis=1  # Adjust tile number as needed
)

# Join on identifier
merged = satellite_df.merge(
    cloud_df[cloud_df["Type"] == "DEL"],
    on="EMSR_AOI",
    how="left"
)
```

---

## Common Use Cases

### 1. Binary Fire Segmentation

```python
# Load image and mask
import rasterio

with rasterio.open("path/to/EMSR230_AOI01_01_S2L2A.tif") as src:
    image = src.read()  # Shape: (12, H, W)

with rasterio.open("path/to/EMSR230_AOI01_01_DEL.tif") as src:
    mask = src.read(1)  # Shape: (H, W), values 0 or 1
```

### 2. Filter by Country/Region

```python
import pandas as pd

df = pd.read_csv("csv_files/satelliteData.csv")

# Get Portuguese fires only
portugal = df[df["country"] == "Portugal"]

# Get fires in a specific bounding box (e.g., Catalonia)
catalonia = df[
    (df["country"] == "Spain") &
    (df["centerBoxLong"] >= 0.15) & (df["centerBoxLong"] <= 3.35) &
    (df["centerBoxLat"] >= 40.5) & (df["centerBoxLat"] <= 42.9)
]
```

### 3. Filter by Date Range

```python
df["activationDate"] = pd.to_datetime(df["activationDate"])

# Fires from 2022
fires_2022 = df[df["activationDate"].dt.year == 2022]

# Summer fires (June-September)
summer_fires = df[df["activationDate"].dt.month.isin([6, 7, 8, 9])]
```

### 4. Filter by Burn Size

```python
# Large fires only (>100,000 burned pixels)
large_fires = df[df["pixelBurned"] > 100000]

# Calculate actual area (approximate)
# At 10m resolution: area = pixelBurned * 10 * 10 = pixelBurned * 100 m²
df["burned_area_km2"] = df["pixelBurned"] * 100 / 1e6
```

### 5. Exclude Cloudy Images

```python
cloud_df = pd.read_csv("csv_files/cloudCoverage.csv")

# Keep only images with <10% cloud cover
clear_images = cloud_df[cloud_df["percentageCloud"] < 0.10]
```

---

## Data Splits

The dataset is pre-split into train/val/test:

| Split | Location | Purpose |
|-------|----------|---------|
| `train/train/` | Training data | Model training |
| `val/val/` | Validation data | Hyperparameter tuning |
| `test/test/` | Test data | Final evaluation |
| `cat/` | Catalonia subset | Regional testing (optional) |

**Note**: The nested structure (`train/train/`) results from the tar extraction process.

### Split Statistics

To get statistics per split:

```python
from pathlib import Path

data_root = Path("wildfires-cems/data")

for split in ["train", "val", "test"]:
    split_dir = data_root / split / split
    if split_dir.exists():
        emsr_folders = list(split_dir.glob("EMSR*"))
        print(f"{split}: {len(emsr_folders)} EMSR activations")
```

---

## Additional Resources

- **Original Repository**: [github.com/MatteoM95/CEMS-Wildfire-Dataset](https://github.com/MatteoM95/CEMS-Wildfire-Dataset)
- **HuggingFace Dataset**: [huggingface.co/datasets/links-ads/wildfires-cems](https://huggingface.co/datasets/links-ads/wildfires-cems)
- **CEMS Rapid Mapping**: [emergency.copernicus.eu](https://emergency.copernicus.eu/mapping/list-of-activations-rapid)
- **Sentinel-2 Documentation**: [sentinels.copernicus.eu](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2)

---

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{arnaudo2023burned,
  title={Robust Burned Area Delineation through Multitask Learning},
  author={Arnaudo, Edoardo and Barco, Luca and Merlo, Matteo and Rossi, Claudio},
  booktitle={Conference on Machine Learning and Principles and Practice
             of Knowledge Discovery in Databases},
  year={2023}
}
```
