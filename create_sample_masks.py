#!/usr/bin/env python3
"""
Create sample urban and rural masks for UHI analysis.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds

def create_sample_masks():
    """Create sample urban and rural masks based on temperature raster extent."""
    
    # Reference raster to match extent
    reference_tif = "Output_LOOCV_Irregular_Mesh/ensemble/mesh_recon_20230801_060000.tif"
    
    with rasterio.open(reference_tif) as src:
        # Get metadata
        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, nodata=0)
        
        height, width = src.shape
        bounds = src.bounds
        transform = src.transform
        
        print(f"Creating masks with shape: {height} x {width}")
        print(f"Bounds: {bounds}")
        
        # Create urban mask (central area - roughly center 60% of the image)
        urban_mask = np.zeros((height, width), dtype=np.uint8)
        h_margin = int(height * 0.2)
        w_margin = int(width * 0.2)
        urban_mask[h_margin:height-h_margin, w_margin:width-w_margin] = 1
        
        # Create rural mask (outer areas)
        rural_mask = np.zeros((height, width), dtype=np.uint8)
        # Top and bottom strips
        rural_mask[:h_margin, :] = 1
        rural_mask[height-h_margin:, :] = 1
        # Left and right strips (not overlapping with urban)
        rural_mask[:, :w_margin] = 1
        rural_mask[:, width-w_margin:] = 1
        
        print(f"Urban pixels: {np.sum(urban_mask)}")
        print(f"Rural pixels: {np.sum(rural_mask)}")
        
        # Save masks
        urban_mask_path = "ROI_WorldCover_2021_10m_urban_mask.tif"
        rural_mask_path = "ROI_WorldCover_2021_10m_rural_mask.tif"
        
        with rasterio.open(urban_mask_path, 'w', **profile) as dst:
            dst.write(urban_mask, 1)
        
        with rasterio.open(rural_mask_path, 'w', **profile) as dst:
            dst.write(rural_mask, 1)
        
        print(f"\nCreated masks:")
        print(f"  Urban: {urban_mask_path}")
        print(f"  Rural: {rural_mask_path}")

if __name__ == "__main__":
    create_sample_masks()
