# Urban Heat Island (UHI) Analysis

This repository contains code for analyzing Urban Heat Island intensity using both point-based sensor data and continuous temperature raster fields.

## Overview

The analysis implements two complementary approaches:

1. **Point-based UHI Analysis**: Uses sensor station temperature data to calculate UHI intensity by comparing urban and rural sensor measurements
2. **Raster-based UHI Analysis**: Uses continuous temperature fields to calculate spatial UHI intensity by comparing pixels to rural baseline

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- rasterio

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy rasterio
```

## Usage

### Quick Start

Run the complete analysis:
```bash
python uhi_analysis.py
```

### Input Data

The script expects the following inputs:

1. **Sensor Data**: CSV file with columns:
   - `station_id`: Unique identifier for each sensor
   - `date`: Timestamp
   - `temperature`: Temperature in °C
   - `latitude`, `longitude`: Sensor location
   - `RH`, `dew_point`: Optional additional measurements

2. **Urban/Rural Masks** (optional): GeoTIFF files defining urban and rural areas
   - If not provided, the script will use a simple latitude-based classification

3. **Temperature Rasters** (for Part 2): Directory containing hourly temperature GeoTIFF files
   - Expected naming: `mesh_recon_YYYYMMDD_HHMMSS.tif`

### Output

The analysis generates:

#### Part 1: Point-based Analysis
- `uhi_timeseries.png`: Time series showing urban/rural temperatures and UHI intensity
- `uhi_daytime_vs_nighttime.png`: Comparison of daytime vs nighttime UHI patterns

#### Part 2: Raster-based Analysis
- `uhi_intensity_*.tif`: UHI intensity raster for each timestep
- `raster_uhi_results.csv`: Summary statistics for each timestep
- `raster_uhi_timeseries.png`: Time series of raster-based UHI metrics
- `raster_uhi_daytime_vs_nighttime.png`: Daytime vs nighttime comparison

All outputs are saved to the `uhi_analysis_output/` directory.

## Methodology

### UHI Intensity Calculation

**Point-based**:
```
UHI_intensity = Mean(Urban_sensors) - Mean(Rural_sensors)
```

**Raster-based**:
```
UHI_intensity(pixel) = Temperature(pixel) - Mean(Rural_pixels)
```

### Daytime vs Nighttime Classification

- **Daytime**: 6:00 - 18:59 (6 AM to 6:59 PM)
- **Nighttime**: 19:00 - 5:59 (7 PM to 5:59 AM)

## Key Findings

The analysis provides:

1. Temporal patterns of UHI intensity throughout August 2023
2. Statistical comparison of daytime vs nighttime UHI intensity
3. Spatial distribution of UHI intensity across the study area
4. Hourly patterns showing peak UHI periods

## Creating Sample Masks

If you don't have urban/rural mask files, you can create sample masks based on your temperature rasters:

```bash
python create_sample_masks.py
```

This creates simple geometric masks for testing purposes.

## File Structure

```
.
├── uhi_analysis.py              # Main analysis script
├── create_sample_masks.py       # Helper script to create sample masks
├── original_2023/               # Sensor data directory
│   └── filtered_2023_aug_hourly.csv
├── Output_LOOCV_Irregular_Mesh/ # Temperature rasters directory
│   └── ensemble/
│       └── mesh_recon_*.tif
├── uhi_analysis_output/         # Generated output files
│   ├── *.png                    # Visualization plots
│   ├── *.tif                    # UHI intensity rasters
│   └── raster_uhi_results.csv   # Summary statistics
└── README.md                    # This file
```

## Notes

- The script is designed to handle missing data gracefully
- If mask files are not available, it falls back to a simple latitude-based classification
- Rasterio is optional for point-based analysis but required for raster-based analysis
- All generated visualizations are publication-quality (300 DPI)

## Author

Generated for the Ta-GAT repository (Urban Heat Island Analysis)
Date: February 2026
