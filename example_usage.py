#!/usr/bin/env python3
"""
Example usage of the UHI analysis tools.

This script demonstrates how to use the UHI analysis implementation
with custom parameters and data sources.
"""

from uhi_analysis import UHIAnalyzer
import os

def example_basic_analysis():
    """Run basic UHI analysis with default settings."""
    
    print("="*80)
    print("EXAMPLE 1: Basic UHI Analysis")
    print("="*80)
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up paths
    sensor_csv = os.path.join(base_dir, "original_2023/filtered_2023_aug_hourly.csv")
    urban_mask = os.path.join(base_dir, "ROI_WorldCover_2021_10m_urban_mask.tif")
    rural_mask = os.path.join(base_dir, "ROI_WorldCover_2021_10m_rural_mask.tif")
    temp_rasters = os.path.join(base_dir, "Output_LOOCV_Irregular_Mesh/ensemble")
    output_dir = os.path.join(base_dir, "uhi_analysis_output")
    
    # Create analyzer
    analyzer = UHIAnalyzer(
        sensor_csv_path=sensor_csv,
        urban_mask_path=urban_mask,
        rural_mask_path=rural_mask,
        temperature_rasters_dir=temp_rasters,
        output_dir=output_dir
    )
    
    # Run complete analysis
    analyzer.run_full_analysis()
    
    print("\nAnalysis complete! Check the output directory for results.")
    print(f"Output directory: {output_dir}")


def example_point_based_only():
    """Run only the point-based analysis."""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Point-based Analysis Only")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    analyzer = UHIAnalyzer(
        sensor_csv_path=os.path.join(base_dir, "original_2023/filtered_2023_aug_hourly.csv"),
        output_dir=os.path.join(base_dir, "uhi_output_point_only")
    )
    
    # Load data
    analyzer.load_sensor_data()
    
    # Classify sensors (will use simple method if masks not available)
    analyzer.classify_sensors_by_mask()
    
    # Calculate and plot UHI
    analyzer.calculate_uhi_intensity()
    analyzer.plot_uhi_timeseries()
    analyzer.plot_daytime_vs_nighttime()
    
    print("\nPoint-based analysis complete!")


def example_custom_classification():
    """Example with custom sensor classification."""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Sensor Classification")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    analyzer = UHIAnalyzer(
        sensor_csv_path=os.path.join(base_dir, "original_2023/filtered_2023_aug_hourly.csv"),
        output_dir=os.path.join(base_dir, "uhi_output_custom")
    )
    
    # Load data
    analyzer.load_sensor_data()
    
    # Get unique stations
    stations = analyzer.sensor_data['station_id'].unique()
    
    # Custom classification: first half as urban, second half as rural
    mid = len(stations) // 2
    analyzer.urban_sensors = list(stations[:mid])
    analyzer.rural_sensors = list(stations[mid:])
    
    print(f"\nCustom classification:")
    print(f"  Urban sensors: {len(analyzer.urban_sensors)}")
    print(f"  Rural sensors: {len(analyzer.rural_sensors)}")
    
    # Calculate and plot
    analyzer.calculate_uhi_intensity()
    analyzer.plot_uhi_timeseries()
    analyzer.plot_daytime_vs_nighttime()
    
    print("\nCustom classification analysis complete!")


def example_data_exploration():
    """Example of exploring the data before analysis."""
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Data Exploration")
    print("="*80)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    analyzer = UHIAnalyzer(
        sensor_csv_path=os.path.join(base_dir, "original_2023/filtered_2023_aug_hourly.csv"),
        output_dir=os.path.join(base_dir, "uhi_output_explore")
    )
    
    # Load data
    df = analyzer.load_sensor_data()
    
    # Explore the data
    print("\n" + "-"*80)
    print("DATA EXPLORATION")
    print("-"*80)
    
    print("\nBasic Statistics:")
    print(df[['temperature', 'RH', 'dew_point']].describe())
    
    print("\nTemperature by Station:")
    station_stats = df.groupby('station_id')['temperature'].agg(['mean', 'std', 'min', 'max'])
    print(station_stats.round(2))
    
    print("\nTemperature by Hour of Day:")
    hourly_stats = df.groupby('hour')['temperature'].agg(['mean', 'std'])
    print(hourly_stats.round(2))
    
    print("\nData Coverage:")
    coverage = df.groupby('station_id')['datetime'].count()
    print(f"Average records per station: {coverage.mean():.0f}")
    print(f"Total hours in August: 744 (31 days × 24 hours)")
    print(f"Coverage: {(coverage.mean()/744)*100:.1f}%")


if __name__ == "__main__":
    # Choose which example to run
    import sys
    
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        print("Usage: python example_usage.py [example_number]")
        print("\nAvailable examples:")
        print("  1 - Basic complete analysis")
        print("  2 - Point-based analysis only")
        print("  3 - Custom sensor classification")
        print("  4 - Data exploration")
        print("\nRunning example 1 by default...")
        example = "1"
    
    if example == "1":
        example_basic_analysis()
    elif example == "2":
        example_point_based_only()
    elif example == "3":
        example_custom_classification()
    elif example == "4":
        example_data_exploration()
    else:
        print(f"Unknown example: {example}")
        print("Valid options: 1, 2, 3, 4")
