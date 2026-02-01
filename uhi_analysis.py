#!/usr/bin/env python3
"""
Urban Heat Island (UHI) Analysis Script
========================================

This script performs two types of UHI analysis:
1. Point-based analysis using sensor station data
2. Raster-based analysis using continuous temperature fields

Author: Generated for Ta-GAT repository
Date: 2026-02-01
"""

import os
import glob
import re
from datetime import datetime
from typing import Tuple, List, Optional, Dict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import rasterio, but make it optional
try:
    import rasterio
    from rasterio.mask import mask as rasterio_mask
    from rasterio.features import geometry_mask
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("Warning: rasterio not available. Raster-based analysis will be limited.")

# Plotting configuration
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150


class UHIAnalyzer:
    """Main class for Urban Heat Island analysis."""
    
    def __init__(self, 
                 sensor_csv_path: str,
                 urban_mask_path: Optional[str] = None,
                 rural_mask_path: Optional[str] = None,
                 temperature_rasters_dir: Optional[str] = None,
                 output_dir: str = "./uhi_output"):
        """
        Initialize the UHI analyzer.
        
        Parameters
        ----------
        sensor_csv_path : str
            Path to the CSV file with sensor data
        urban_mask_path : str, optional
            Path to urban mask GeoTIFF
        rural_mask_path : str, optional
            Path to rural mask GeoTIFF
        temperature_rasters_dir : str, optional
            Directory containing temperature raster files
        output_dir : str
            Directory for output plots and results
        """
        self.sensor_csv_path = sensor_csv_path
        self.urban_mask_path = urban_mask_path
        self.rural_mask_path = rural_mask_path
        self.temperature_rasters_dir = temperature_rasters_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data storage
        self.sensor_data = None
        self.urban_sensors = None
        self.rural_sensors = None
        self.uhi_timeseries = None
        
    def load_sensor_data(self) -> pd.DataFrame:
        """Load and prepare sensor data from CSV."""
        print(f"\n{'='*60}")
        print("LOADING SENSOR DATA")
        print(f"{'='*60}")
        
        if not os.path.exists(self.sensor_csv_path):
            raise FileNotFoundError(f"Sensor data not found: {self.sensor_csv_path}")
        
        print(f"Reading: {self.sensor_csv_path}")
        df = pd.read_csv(self.sensor_csv_path)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df['date'])
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        
        print(f"Loaded {len(df)} records")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"Number of unique stations: {df['station_id'].nunique()}")
        print(f"Columns: {list(df.columns)}")
        
        self.sensor_data = df
        return df
    
    def classify_sensors_by_mask(self) -> Tuple[List[str], List[str]]:
        """
        Classify sensors as urban or rural based on mask files.
        
        Returns
        -------
        urban_sensors : list
            List of station IDs in urban areas
        rural_sensors : list
            List of station IDs in rural areas
        """
        print(f"\n{'='*60}")
        print("CLASSIFYING SENSORS BY URBAN/RURAL MASK")
        print(f"{'='*60}")
        
        if not RASTERIO_AVAILABLE:
            print("Warning: rasterio not available. Using simple latitude-based classification.")
            return self._classify_sensors_simple()
        
        if not self.urban_mask_path or not os.path.exists(self.urban_mask_path):
            print(f"Warning: Urban mask not found at {self.urban_mask_path}")
            print("Using simple latitude-based classification instead.")
            return self._classify_sensors_simple()
        
        if not self.rural_mask_path or not os.path.exists(self.rural_mask_path):
            print(f"Warning: Rural mask not found at {self.rural_mask_path}")
            print("Using simple latitude-based classification instead.")
            return self._classify_sensors_simple()
        
        # Get unique sensor locations
        sensor_locations = self.sensor_data[['station_id', 'longitude', 'latitude']].drop_duplicates()
        
        urban_sensors = []
        rural_sensors = []
        
        try:
            with rasterio.open(self.urban_mask_path) as urban_src:
                with rasterio.open(self.rural_mask_path) as rural_src:
                    for _, row in sensor_locations.iterrows():
                        lon, lat = row['longitude'], row['latitude']
                        
                        # Convert to pixel coordinates and read value
                        py_urban, px_urban = urban_src.index(lon, lat)
                        py_rural, px_rural = rural_src.index(lon, lat)
                        
                        # Read values (assuming mask value > 0 means inside the area)
                        urban_val = urban_src.read(1)[py_urban, px_urban]
                        rural_val = rural_src.read(1)[py_rural, px_rural]
                        
                        if urban_val > 0:
                            urban_sensors.append(row['station_id'])
                        elif rural_val > 0:
                            rural_sensors.append(row['station_id'])
            
            print(f"Urban sensors: {len(urban_sensors)}")
            print(f"Rural sensors: {len(rural_sensors)}")
            
        except Exception as e:
            print(f"Error reading masks: {e}")
            print("Falling back to simple classification.")
            return self._classify_sensors_simple()
        
        if len(urban_sensors) == 0 or len(rural_sensors) == 0:
            print("Warning: No sensors found in urban or rural areas.")
            print("Falling back to simple classification.")
            return self._classify_sensors_simple()
        
        self.urban_sensors = urban_sensors
        self.rural_sensors = rural_sensors
        
        return urban_sensors, rural_sensors
    
    def _classify_sensors_simple(self) -> Tuple[List[str], List[str]]:
        """
        Simple classification: split sensors by median latitude.
        Sensors above median = urban, below median = rural (or vice versa).
        """
        sensor_locations = self.sensor_data[['station_id', 'latitude']].drop_duplicates()
        median_lat = sensor_locations['latitude'].median()
        
        urban_sensors = sensor_locations[sensor_locations['latitude'] >= median_lat]['station_id'].tolist()
        rural_sensors = sensor_locations[sensor_locations['latitude'] < median_lat]['station_id'].tolist()
        
        print(f"Using simple classification (median latitude = {median_lat:.4f}):")
        print(f"  Urban sensors (lat >= {median_lat:.4f}): {len(urban_sensors)}")
        print(f"  Rural sensors (lat < {median_lat:.4f}): {len(rural_sensors)}")
        
        self.urban_sensors = urban_sensors
        self.rural_sensors = rural_sensors
        
        return urban_sensors, rural_sensors
    
    def calculate_uhi_intensity(self) -> pd.DataFrame:
        """
        Calculate UHI intensity as the difference between urban and rural mean temperatures.
        
        Returns
        -------
        uhi_df : pd.DataFrame
            DataFrame with datetime, urban_temp, rural_temp, uhi_intensity
        """
        print(f"\n{'='*60}")
        print("CALCULATING UHI INTENSITY")
        print(f"{'='*60}")
        
        if self.urban_sensors is None or self.rural_sensors is None:
            self.classify_sensors_by_mask()
        
        # Filter data for urban and rural sensors
        urban_data = self.sensor_data[self.sensor_data['station_id'].isin(self.urban_sensors)]
        rural_data = self.sensor_data[self.sensor_data['station_id'].isin(self.rural_sensors)]
        
        # Calculate hourly means
        urban_hourly = urban_data.groupby('datetime')['temperature'].mean().reset_index()
        urban_hourly.columns = ['datetime', 'urban_temp']
        
        rural_hourly = rural_data.groupby('datetime')['temperature'].mean().reset_index()
        rural_hourly.columns = ['datetime', 'rural_temp']
        
        # Merge and calculate UHI intensity
        uhi_df = pd.merge(urban_hourly, rural_hourly, on='datetime', how='inner')
        uhi_df['uhi_intensity'] = uhi_df['urban_temp'] - uhi_df['rural_temp']
        
        # Add time attributes
        uhi_df['hour'] = uhi_df['datetime'].dt.hour
        uhi_df['day'] = uhi_df['datetime'].dt.day
        uhi_df['date'] = uhi_df['datetime'].dt.date
        
        # Classify as daytime (6-18) or nighttime (19-5)
        uhi_df['time_period'] = uhi_df['hour'].apply(
            lambda h: 'Daytime' if 6 <= h < 19 else 'Nighttime'
        )
        
        print(f"Calculated UHI intensity for {len(uhi_df)} hourly records")
        print(f"Mean UHI intensity: {uhi_df['uhi_intensity'].mean():.2f}°C")
        print(f"Max UHI intensity: {uhi_df['uhi_intensity'].max():.2f}°C")
        print(f"Min UHI intensity: {uhi_df['uhi_intensity'].min():.2f}°C")
        
        self.uhi_timeseries = uhi_df
        
        return uhi_df
    
    def plot_uhi_timeseries(self, save_path: Optional[str] = None):
        """Plot UHI intensity time series."""
        if self.uhi_timeseries is None:
            self.calculate_uhi_intensity()
        
        df = self.uhi_timeseries
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Plot 1: Urban and Rural temperatures
        ax1 = axes[0]
        ax1.plot(df['datetime'], df['urban_temp'], 'o-', 
                label='Urban Mean', color='#d62728', alpha=0.7, markersize=3)
        ax1.plot(df['datetime'], df['rural_temp'], 'o-',
                label='Rural Mean', color='#2ca02c', alpha=0.7, markersize=3)
        ax1.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Urban vs Rural Mean Temperature - August 2023', 
                     fontsize=13, fontweight='bold', pad=10)
        
        # Plot 2: UHI Intensity
        ax2 = axes[1]
        colors = ['#ff7f0e' if x == 'Daytime' else '#1f77b4' 
                  for x in df['time_period']]
        ax2.scatter(df['datetime'], df['uhi_intensity'], 
                   c=colors, alpha=0.6, s=30)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Urban Heat Island Intensity (Urban - Rural)', 
                     fontsize=13, fontweight='bold', pad=10)
        
        # Add legend for day/night
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff7f0e', label='Daytime (6-18h)'),
            Patch(facecolor='#1f77b4', label='Nighttime (19-5h)')
        ]
        ax2.legend(handles=legend_elements, loc='best', frameon=True, shadow=True)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator())
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'uhi_timeseries.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
        plt.close()
    
    def plot_daytime_vs_nighttime(self, save_path: Optional[str] = None):
        """Compare daytime vs nighttime UHI intensity."""
        if self.uhi_timeseries is None:
            self.calculate_uhi_intensity()
        
        df = self.uhi_timeseries
        
        # Calculate statistics
        stats_df = df.groupby('time_period')['uhi_intensity'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print(f"\n{'='*60}")
        print("DAYTIME VS NIGHTTIME UHI STATISTICS")
        print(f"{'='*60}")
        print(stats_df)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Plot 1: Box plot
        ax1 = axes[0]
        daytime_data = df[df['time_period'] == 'Daytime']['uhi_intensity']
        nighttime_data = df[df['time_period'] == 'Nighttime']['uhi_intensity']
        
        bp = ax1.boxplot([daytime_data, nighttime_data],
                         labels=['Daytime\n(6-18h)', 'Nighttime\n(19-5h)'],
                         patch_artist=True,
                         medianprops=dict(color='red', linewidth=2),
                         boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax1.set_ylabel('UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax1.set_title('UHI Intensity Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Plot 2: Mean comparison with error bars
        ax2 = axes[1]
        means = [daytime_data.mean(), nighttime_data.mean()]
        stds = [daytime_data.std(), nighttime_data.std()]
        x_pos = [0, 1]
        colors_bar = ['#ff7f0e', '#1f77b4']
        
        bars = ax2.bar(x_pos, means, yerr=stds, color=colors_bar, 
                      alpha=0.7, capsize=10, width=0.6)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(['Daytime\n(6-18h)', 'Nighttime\n(19-5h)'])
        ax2.set_ylabel('Mean UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax2.set_title('Mean UHI Intensity ± Std', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{mean:.2f}°C',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Hourly average
        ax3 = axes[2]
        hourly_avg = df.groupby('hour')['uhi_intensity'].mean()
        hours = hourly_avg.index
        colors_line = ['#ff7f0e' if 6 <= h < 19 else '#1f77b4' for h in hours]
        
        for i in range(len(hours)):
            ax3.plot(hours[i:i+2], hourly_avg.iloc[i:i+2], 
                    'o-', color=colors_line[i], linewidth=2, markersize=6)
        
        ax3.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Mean UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax3.set_title('Hourly UHI Intensity Pattern', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axvspan(6, 19, alpha=0.1, color='orange', label='Daytime')
        ax3.axvspan(19, 24, alpha=0.1, color='blue')
        ax3.axvspan(0, 6, alpha=0.1, color='blue')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'uhi_daytime_vs_nighttime.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
        plt.close()
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(daytime_data, nighttime_data)
        print(f"\nT-test (Daytime vs Nighttime):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4e}")
        if p_value < 0.05:
            print(f"  Result: Significantly different (p < 0.05)")
        else:
            print(f"  Result: Not significantly different (p >= 0.05)")
    
    def analyze_raster_uhi(self):
        """Analyze UHI using temperature raster fields."""
        print(f"\n{'='*60}")
        print("RASTER-BASED UHI ANALYSIS")
        print(f"{'='*60}")
        
        if not RASTERIO_AVAILABLE:
            print("Error: rasterio is required for raster analysis.")
            return
        
        if not self.temperature_rasters_dir or not os.path.exists(self.temperature_rasters_dir):
            print(f"Error: Temperature rasters directory not found: {self.temperature_rasters_dir}")
            return
        
        if not self.rural_mask_path or not os.path.exists(self.rural_mask_path):
            print(f"Error: Rural mask not found: {self.rural_mask_path}")
            return
        
        # Find all temperature raster files
        tif_pattern = os.path.join(self.temperature_rasters_dir, "mesh_recon_*.tif")
        tif_files = sorted(glob.glob(tif_pattern))
        
        # Exclude uncertainty files
        tif_files = [f for f in tif_files if 'uncertainty' not in f]
        
        print(f"Found {len(tif_files)} temperature raster files")
        
        if len(tif_files) == 0:
            print("Error: No temperature raster files found")
            return
        
        # Process each raster
        results = []
        
        for tif_file in tif_files:
            try:
                # Extract datetime from filename
                filename = os.path.basename(tif_file)
                # Pattern: mesh_recon_20230802_130000.tif
                match = re.search(r'(\d{8})_(\d{6})', filename)
                if not match:
                    continue
                
                date_str = match.group(1)
                time_str = match.group(2)
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                
                # Read temperature raster
                with rasterio.open(tif_file) as src:
                    temp_data = src.read(1)
                    profile = src.profile
                    
                    # Read rural mask
                    with rasterio.open(self.rural_mask_path) as mask_src:
                        # Resample mask to match temperature raster if needed
                        rural_mask = mask_src.read(1)
                        
                        # Calculate mean temperature for rural pixels
                        rural_pixels = temp_data[rural_mask > 0]
                        rural_mean = np.nanmean(rural_pixels)
                        
                        # Calculate UHI intensity (subtract rural mean from entire raster)
                        uhi_intensity = temp_data - rural_mean
                        
                        # Save UHI intensity raster
                        output_filename = f"uhi_intensity_{date_str}_{time_str}.tif"
                        output_path = os.path.join(self.output_dir, output_filename)
                        
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            dst.write(uhi_intensity, 1)
                        
                        # Calculate statistics
                        hour = dt.hour
                        time_period = 'Daytime' if 6 <= hour < 19 else 'Nighttime'
                        
                        results.append({
                            'datetime': dt,
                            'hour': hour,
                            'day': dt.day,
                            'time_period': time_period,
                            'rural_mean': rural_mean,
                            'uhi_mean': np.nanmean(uhi_intensity),
                            'uhi_max': np.nanmax(uhi_intensity),
                            'uhi_min': np.nanmin(uhi_intensity),
                            'uhi_std': np.nanstd(uhi_intensity),
                            'output_file': output_filename
                        })
                
            except Exception as e:
                print(f"Error processing {tif_file}: {e}")
                continue
        
        if len(results) == 0:
            print("No rasters were successfully processed")
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        print(f"\nProcessed {len(results_df)} rasters successfully")
        print(f"Date range: {results_df['datetime'].min()} to {results_df['datetime'].max()}")
        
        # Save results
        results_csv = os.path.join(self.output_dir, 'raster_uhi_results.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"Saved results: {results_csv}")
        
        # Plot results
        self._plot_raster_uhi_results(results_df)
        
        return results_df
    
    def _plot_raster_uhi_results(self, results_df: pd.DataFrame):
        """Plot raster-based UHI analysis results."""
        
        # Plot 1: Time series
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        ax1 = axes[0]
        colors = ['#ff7f0e' if x == 'Daytime' else '#1f77b4' 
                  for x in results_df['time_period']]
        ax1.scatter(results_df['datetime'], results_df['uhi_mean'],
                   c=colors, alpha=0.6, s=40)
        ax1.set_ylabel('Mean UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax1.set_title('Raster-based UHI Intensity - Mean Values', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2 = axes[1]
        ax2.scatter(results_df['datetime'], results_df['uhi_max'],
                   c=colors, alpha=0.6, s=40, label='Max')
        ax2.scatter(results_df['datetime'], results_df['uhi_min'],
                   c=colors, alpha=0.4, s=20, marker='v', label='Min')
        ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax2.set_ylabel('UHI Intensity Range (°C)', fontsize=11, fontweight='bold')
        ax2.set_title('Raster-based UHI Intensity - Max/Min Values', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'raster_uhi_timeseries.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
        plt.close()
        
        # Plot 2: Daytime vs Nighttime comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        daytime = results_df[results_df['time_period'] == 'Daytime']
        nighttime = results_df[results_df['time_period'] == 'Nighttime']
        
        bp = ax1.boxplot([daytime['uhi_mean'], nighttime['uhi_mean']],
                         labels=['Daytime\n(6-18h)', 'Nighttime\n(19-5h)'],
                         patch_artist=True,
                         medianprops=dict(color='red', linewidth=2))
        ax1.set_ylabel('Mean UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax1.set_title('Raster UHI: Daytime vs Nighttime', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2 = axes[1]
        hourly_avg = results_df.groupby('hour')['uhi_mean'].mean()
        hours = hourly_avg.index
        colors_line = ['#ff7f0e' if 6 <= h < 19 else '#1f77b4' for h in hours]
        
        for i in range(len(hours)-1):
            ax2.plot(hours[i:i+2], hourly_avg.iloc[i:i+2],
                    'o-', color=colors_line[i], linewidth=2, markersize=6)
        
        ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Mean UHI Intensity (°C)', fontsize=11, fontweight='bold')
        ax2.set_title('Hourly Pattern - Raster UHI', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvspan(6, 19, alpha=0.1, color='orange')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'raster_uhi_daytime_vs_nighttime.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
        
        # Print statistics
        print(f"\n{'='*60}")
        print("RASTER-BASED UHI STATISTICS")
        print(f"{'='*60}")
        stats_summary = results_df.groupby('time_period')['uhi_mean'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3)
        print(stats_summary)
    
    def run_full_analysis(self):
        """Run complete UHI analysis pipeline."""
        print("\n" + "="*60)
        print("URBAN HEAT ISLAND ANALYSIS")
        print("="*60)
        
        # Part 1: Point-based analysis
        print("\n" + "="*60)
        print("PART 1: POINT-BASED UHI ANALYSIS")
        print("="*60)
        
        self.load_sensor_data()
        self.classify_sensors_by_mask()
        self.calculate_uhi_intensity()
        self.plot_uhi_timeseries()
        self.plot_daytime_vs_nighttime()
        
        # Part 2: Raster-based analysis
        print("\n" + "="*60)
        print("PART 2: RASTER-BASED UHI ANALYSIS")
        print("="*60)
        
        if self.temperature_rasters_dir and RASTERIO_AVAILABLE:
            self.analyze_raster_uhi()
        else:
            print("Skipping raster analysis (directory not specified or rasterio not available)")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Output directory: {self.output_dir}")


def main():
    """Main entry point."""
    
    # Configure paths - these should be updated based on actual file locations
    # The problem statement mentions Windows paths, but we'll use the repository structure
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to sensor data (using repository structure)
    sensor_csv = os.path.join(base_dir, "original_2023/filtered_2023_aug_hourly.csv")
    
    # Mask paths - check multiple locations
    # First try local directory, then Windows paths
    urban_mask_local = os.path.join(base_dir, "ROI_WorldCover_2021_10m_urban_mask.tif")
    rural_mask_local = os.path.join(base_dir, "ROI_WorldCover_2021_10m_rural_mask.tif")
    
    urban_mask_win = "C:/Downloads/ROI_WorldCover_2021_10m_urban_mask.tif"
    rural_mask_win = "C:/Downloads/ROI_WorldCover_2021_10m_rural_mask.tif"
    
    urban_mask = urban_mask_local if os.path.exists(urban_mask_local) else urban_mask_win
    rural_mask = rural_mask_local if os.path.exists(rural_mask_local) else rural_mask_win
    
    # Temperature rasters directory (using repository structure)
    temp_rasters = os.path.join(base_dir, "Output_LOOCV_Irregular_Mesh/ensemble")
    
    # Output directory
    output_dir = os.path.join(base_dir, "uhi_analysis_output")
    
    # Create analyzer
    analyzer = UHIAnalyzer(
        sensor_csv_path=sensor_csv,
        urban_mask_path=urban_mask,
        rural_mask_path=rural_mask,
        temperature_rasters_dir=temp_rasters,
        output_dir=output_dir
    )
    
    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
