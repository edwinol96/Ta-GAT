#!/usr/bin/env python3
"""
Test the forecast mode with synthetic/mock data.

This script creates synthetic data and a simple model to verify
that the forecast mode functionality works correctly.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from datetime import datetime, timedelta


class DualStreamGAT(torch.nn.Module):
    """GNN model (same as in forecast_mode.py)"""
    def __init__(self, num_static: int, num_time: int, hidden_dim: int, heads: int = 3, dropout: float = 0.4):
        super().__init__()
        self.geo_gat = GATv2Conv(in_channels=2, out_channels=hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.sem_in_dim = num_static + num_time + 1
        self.sem_gat = GATv2Conv(in_channels=self.sem_in_dim, out_channels=hidden_dim, heads=heads, dropout=dropout, concat=True)
        fusion_dim = (hidden_dim * heads) * 2
        self.norm = torch.nn.LayerNorm(fusion_dim)
        self.lin_out = torch.nn.Linear(fusion_dim, 1)

    def forward(self, x_semantic, x_coords, edge_index):
        h_geo = self.geo_gat(x_coords, edge_index)
        h_sem = self.sem_gat(x_semantic, edge_index)
        combined = torch.cat([h_geo, h_sem], dim=1)
        combined = self.norm(combined)
        combined = F.elu(combined)
        return self.lin_out(combined)


def create_synthetic_data(n_sensors=10, n_timesteps=25, output_dir="./test_data"):
    """
    Create synthetic temperature data for testing
    
    Simulates:
    - Multiple sensors at different locations
    - Hourly temperature readings over time
    - Diurnal temperature patterns
    - Spatial correlation between sensors
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating synthetic data...")
    print(f"  Sensors: {n_sensors}")
    print(f"  Timesteps: {n_timesteps}")
    
    # Generate sensor locations (random lat/lon in a small area)
    np.random.seed(42)
    base_lat, base_lon = 45.5, -73.6  # Montreal area
    sensor_lats = base_lat + np.random.randn(n_sensors) * 0.1
    sensor_lons = base_lon + np.random.randn(n_sensors) * 0.1
    
    # Normalize coordinates
    lat_norm = (sensor_lats - sensor_lats.mean()) / (sensor_lats.std() + 1e-6)
    lon_norm = (sensor_lons - sensor_lons.mean()) / (sensor_lons.std() + 1e-6)
    
    # Generate sensor IDs
    sensor_ids = [f"SENSOR_{i:03d}" for i in range(n_sensors)]
    
    # Generate synthetic static features (e.g., elevation, vegetation)
    static_features = {
        'elevation': np.random.uniform(50, 200, n_sensors),
        'vegetation': np.random.uniform(0.2, 0.8, n_sensors),
        'urban_density': np.random.uniform(0.1, 0.9, n_sensors),
    }
    
    # Generate timestamps (hourly)
    start_time = datetime(2023, 8, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_timesteps)]
    
    # Create temperature data with realistic patterns
    data_rows = []
    
    for t_idx, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Diurnal temperature pattern (sinusoidal)
        base_temp = 20.0  # Base temperature
        diurnal_variation = 5.0 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at ~2 PM
        seasonal_variation = 3.0 * np.sin(2 * np.pi * day_of_year / 365)
        
        for s_idx, sensor_id in enumerate(sensor_ids):
            # Sensor-specific offset based on location and features
            spatial_offset = 0.5 * (sensor_lats[s_idx] - base_lat) * 10  # Latitude effect
            urban_offset = 1.5 * static_features['urban_density'][s_idx]  # Urban heat island
            vegetation_offset = -1.0 * static_features['vegetation'][s_idx]  # Cooling effect
            
            # Add some random noise
            noise = np.random.randn() * 0.3
            
            # Calculate temperature
            temperature = (base_temp + diurnal_variation + seasonal_variation +
                          spatial_offset + urban_offset + vegetation_offset + noise)
            
            # Create row
            row = {
                'dt': timestamp,
                'id_station': sensor_id,
                'temperature': temperature,
                'latitude': sensor_lats[s_idx],
                'longitude': sensor_lons[s_idx],
                'lat_norm': lat_norm[s_idx],
                'lon_norm': lon_norm[s_idx],
                **{f'static_{k}': v[s_idx] for k, v in static_features.items()}
            }
            data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "synthetic_temperature_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved data to: {csv_path}")
    
    return csv_path, n_sensors


def create_synthetic_model(n_static=3, output_dir="./test_data"):
    """
    Create a simple trained model for testing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nCreating synthetic model...")
    
    # Model parameters
    n_time = 6
    hidden_dim = 32
    heads = 3
    dropout = 0.4
    
    # Create model
    model = DualStreamGAT(n_static, n_time, hidden_dim, heads, dropout)
    
    # Initialize with small random weights (would be trained in practice)
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
    
    # Normalization parameters (from synthetic data statistics)
    y_mean = 20.0
    y_std = 5.0
    
    # Save model
    model_path = os.path.join(output_dir, "synthetic_model.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "y_mean": y_mean,
        "y_std": y_std,
        "n_static": n_static,
        "n_time": n_time,
        "hidden_dim": hidden_dim,
        "heads": heads,
        "dropout": dropout,
        "model_class": "DualStreamGAT"
    }, model_path)
    
    print(f"  Saved model to: {model_path}")
    return model_path


def run_test():
    """Run the full test"""
    print("=" * 80)
    print("FORECAST MODE TEST WITH SYNTHETIC DATA")
    print("=" * 80)
    
    # Create test directory
    test_dir = "./test_forecast_output"
    data_dir = "./test_data"
    
    # Create synthetic data and model
    data_path, n_sensors = create_synthetic_data(n_sensors=10, n_timesteps=25, output_dir=data_dir)
    model_path = create_synthetic_model(n_static=3, output_dir=data_dir)
    
    # Import and run forecast mode
    print("\nRunning forecast mode...")
    print("-" * 80)
    
    try:
        # Import here to ensure the module is available
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from forecast_mode import run_forecast_experiment
        
        # Run with minimal settings for quick test
        run_forecast_experiment(
            model_path=model_path,
            data_path=data_path,
            output_dir=test_dir,
            num_initial_steps=5,
            total_steps=15,
            n_runs=2,
            seed=42
        )
        
        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nTest outputs saved to: {test_dir}")
        print("Check the generated plots and summary file.")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
