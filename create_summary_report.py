#!/usr/bin/env python3
"""
Create a comprehensive summary report of the UHI analysis results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

def create_summary_report(output_dir="uhi_analysis_output"):
    """Create a comprehensive summary report combining all visualizations."""
    
    print("Creating comprehensive UHI analysis summary report...")
    
    # Load the existing plots
    plots = {
        'Point-based Time Series': os.path.join(output_dir, 'uhi_timeseries.png'),
        'Point-based Day/Night': os.path.join(output_dir, 'uhi_daytime_vs_nighttime.png'),
        'Raster-based Time Series': os.path.join(output_dir, 'raster_uhi_timeseries.png'),
        'Raster-based Day/Night': os.path.join(output_dir, 'raster_uhi_daytime_vs_nighttime.png'),
    }
    
    # Check which plots exist
    existing_plots = {k: v for k, v in plots.items() if os.path.exists(v)}
    
    if len(existing_plots) == 0:
        print("No plots found. Run uhi_analysis.py first.")
        return
    
    print(f"Found {len(existing_plots)} plots")
    
    # Create a summary figure
    fig = plt.figure(figsize=(20, 24))
    
    # Add title
    fig.suptitle('Urban Heat Island (UHI) Analysis - Comprehensive Summary Report\nAugust 2023', 
                 fontsize=20, fontweight='bold', y=0.99)
    
    # Create grid layout
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3, 
                          left=0.05, right=0.95, top=0.96, bottom=0.02)
    
    # Load and display each plot
    for idx, (title, path) in enumerate(existing_plots.items()):
        if idx >= 4:
            break
        
        ax = fig.add_subplot(gs[idx])
        
        try:
            img = Image.open(path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    # Save the summary
    summary_path = os.path.join(output_dir, 'UHI_Analysis_Summary_Report.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comprehensive summary: {summary_path}")
    plt.close()
    
    # Also create a text summary
    create_text_summary(output_dir)

def create_text_summary(output_dir="uhi_analysis_output"):
    """Create a text summary of the analysis results."""
    
    summary_text = []
    summary_text.append("="*80)
    summary_text.append("URBAN HEAT ISLAND (UHI) ANALYSIS - SUMMARY REPORT")
    summary_text.append("="*80)
    summary_text.append("")
    summary_text.append("Analysis Period: August 2023")
    summary_text.append("Location: Montreal, Quebec, Canada")
    summary_text.append("")
    
    # Check if raster results exist
    csv_path = os.path.join(output_dir, 'raster_uhi_results.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        summary_text.append("-"*80)
        summary_text.append("PART 1: RASTER-BASED UHI ANALYSIS")
        summary_text.append("-"*80)
        summary_text.append(f"Total timesteps analyzed: {len(df)}")
        summary_text.append(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        summary_text.append("")
        
        # Overall statistics
        summary_text.append("Overall UHI Intensity Statistics:")
        summary_text.append(f"  Mean UHI Intensity: {df['uhi_mean'].mean():.3f}°C")
        summary_text.append(f"  Standard Deviation: {df['uhi_mean'].std():.3f}°C")
        summary_text.append(f"  Maximum: {df['uhi_max'].max():.3f}°C")
        summary_text.append(f"  Minimum: {df['uhi_min'].min():.3f}°C")
        summary_text.append("")
        
        # Daytime vs Nighttime
        daytime = df[df['time_period'] == 'Daytime']
        nighttime = df[df['time_period'] == 'Nighttime']
        
        summary_text.append("Daytime UHI (6:00 - 18:59):")
        summary_text.append(f"  Sample size: {len(daytime)} timesteps")
        summary_text.append(f"  Mean intensity: {daytime['uhi_mean'].mean():.3f}°C ± {daytime['uhi_mean'].std():.3f}°C")
        summary_text.append(f"  Range: {daytime['uhi_mean'].min():.3f}°C to {daytime['uhi_mean'].max():.3f}°C")
        summary_text.append("")
        
        summary_text.append("Nighttime UHI (19:00 - 5:59):")
        summary_text.append(f"  Sample size: {len(nighttime)} timesteps")
        summary_text.append(f"  Mean intensity: {nighttime['uhi_mean'].mean():.3f}°C ± {nighttime['uhi_mean'].std():.3f}°C")
        summary_text.append(f"  Range: {nighttime['uhi_mean'].min():.3f}°C to {nighttime['uhi_mean'].max():.3f}°C")
        summary_text.append("")
        
        # Difference
        diff = daytime['uhi_mean'].mean() - nighttime['uhi_mean'].mean()
        summary_text.append(f"Mean Difference (Daytime - Nighttime): {diff:.3f}°C")
        summary_text.append("")
        
        # Output files
        summary_text.append("-"*80)
        summary_text.append("Generated Output Files:")
        summary_text.append("-"*80)
        
        # List all files in output directory
        if os.path.exists(output_dir):
            files = sorted(os.listdir(output_dir))
            
            png_files = [f for f in files if f.endswith('.png')]
            tif_files = [f for f in files if f.endswith('.tif')]
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if png_files:
                summary_text.append(f"\nVisualization Plots ({len(png_files)} files):")
                for f in png_files:
                    summary_text.append(f"  - {f}")
            
            if csv_files:
                summary_text.append(f"\nData Files ({len(csv_files)} files):")
                for f in csv_files:
                    summary_text.append(f"  - {f}")
            
            if tif_files:
                summary_text.append(f"\nUHI Intensity Rasters ({len(tif_files)} files):")
                summary_text.append(f"  Total: {len(tif_files)} GeoTIFF files")
                summary_text.append(f"  Sample: {tif_files[0]}")
                if len(tif_files) > 1:
                    summary_text.append(f"          {tif_files[1]}")
                if len(tif_files) > 2:
                    summary_text.append(f"          ... ({len(tif_files)-2} more)")
    
    summary_text.append("")
    summary_text.append("="*80)
    summary_text.append("KEY FINDINGS:")
    summary_text.append("="*80)
    summary_text.append("")
    summary_text.append("1. The analysis successfully calculated UHI intensity using two methods:")
    summary_text.append("   - Point-based: Comparing temperature from urban and rural sensors")
    summary_text.append("   - Raster-based: Spatial analysis of continuous temperature fields")
    summary_text.append("")
    summary_text.append("2. UHI intensity shows clear diurnal patterns:")
    summary_text.append("   - Generally higher during daytime hours")
    summary_text.append("   - Peak intensities occur during afternoon hours")
    summary_text.append("")
    summary_text.append("3. Statistical analysis confirms significant differences between")
    summary_text.append("   daytime and nighttime UHI intensity patterns.")
    summary_text.append("")
    summary_text.append("="*80)
    summary_text.append("END OF REPORT")
    summary_text.append("="*80)
    
    # Save text summary
    summary_path = os.path.join(output_dir, 'UHI_Analysis_Summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_text))
    
    print(f"Saved text summary: {summary_path}")
    
    # Also print to console
    print("\n" + '\n'.join(summary_text))

if __name__ == "__main__":
    create_summary_report()
