#!/usr/bin/env python3
"""
Forecast Mode Testing for GNN Temperature Prediction Model

This script tests the trained GNN model in "forecast mode":
1. Selects a test case and randomly masks 3 sensors
2. Uses first 5-6 time steps as initial input
3. Performs autoregressive rollout prediction (feeding predictions back)
4. Evaluates separately for masked and unmasked sensors
5. Generates plots and error metrics
"""

import os
import sys
import random
import argparse
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting defaults
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})
plt.rcParams['font.family'] = 'serif'


class DualStreamGAT(torch.nn.Module):
    """GNN model from the original notebook"""
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


def load_model_and_data(model_path: str, data_path: str):
    """Load trained model and test data"""
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model parameters
    n_static = checkpoint['n_static']
    n_time = checkpoint['n_time']
    hidden_dim = checkpoint['hidden_dim']
    heads = checkpoint['heads']
    dropout = checkpoint['dropout']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    
    # Initialize model
    model = DualStreamGAT(n_static, n_time, hidden_dim, heads, dropout)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Load data
    df = pd.read_csv(data_path)
    df['dt'] = pd.to_datetime(df['dt'])
    
    return model, df, y_mean, y_std, n_static, n_time


def prepare_temporal_encoding(timestamp):
    """Create temporal features (sin/cos encoding of hour and day-of-year)"""
    return np.array([
        np.sin(2*np.pi*timestamp.hour/24), 
        np.cos(2*np.pi*timestamp.hour/24),
        np.sin(4*np.pi*timestamp.hour/24), 
        np.cos(4*np.pi*timestamp.hour/24),
        np.sin(2*np.pi*timestamp.dayofyear/365), 
        np.cos(2*np.pi*timestamp.dayofyear/365)
    ], dtype=np.float32)


def select_test_sequence(df: pd.DataFrame, min_length: int = 15) -> pd.DataFrame:
    """
    Select a test sequence with sufficient length
    Returns a dataframe with consecutive timestamps
    """
    # Get unique timestamps
    timestamps = sorted(df['dt'].unique())
    
    # Find consecutive sequences
    best_seq = None
    current_seq = [timestamps[0]]
    
    for i in range(1, len(timestamps)):
        time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # hours
        
        if time_diff == 1.0:  # consecutive hourly data
            current_seq.append(timestamps[i])
        else:
            if len(current_seq) >= min_length:
                best_seq = current_seq
                break
            current_seq = [timestamps[i]]
    
    # Check last sequence
    if best_seq is None and len(current_seq) >= min_length:
        best_seq = current_seq
    
    if best_seq is None:
        raise ValueError(f"No consecutive sequence of length >= {min_length} found")
    
    # Return data for this sequence
    return df[df['dt'].isin(best_seq)].copy()


def forecast_rollout(
    model: torch.nn.Module,
    initial_data: pd.DataFrame,
    masked_sensors: List[str],
    num_initial_steps: int,
    total_steps: int,
    edge_index: torch.Tensor,
    y_mean: float,
    y_std: float,
    n_static: int,
    n_time: int,
    device: torch.device
) -> Tuple[Dict, Dict]:
    """
    Perform autoregressive forecast rollout
    
    Args:
        model: Trained GNN model
        initial_data: DataFrame with initial timesteps
        masked_sensors: List of sensor IDs to mask
        num_initial_steps: Number of initial steps to use as input
        total_steps: Total number of steps to predict
        edge_index: Graph edges
        y_mean, y_std: Normalization parameters
        n_static, n_time: Feature dimensions
        device: torch device
        
    Returns:
        predictions: Dict mapping sensor_id -> array of predictions
        ground_truth: Dict mapping sensor_id -> array of true values
    """
    model.eval()
    
    # Get unique timestamps and sensors
    timestamps = sorted(initial_data['dt'].unique())[:total_steps]
    all_sensors = sorted(initial_data['id_station'].unique())
    sensor_to_idx = {sid: idx for idx, sid in enumerate(all_sensors)}
    n_sensors = len(all_sensors)
    
    # Initialize storage
    predictions = {sid: [] for sid in all_sensors}
    ground_truth = {sid: [] for sid in all_sensors}
    
    # Get static features (assuming they don't change)
    # Extract static features for each sensor
    static_cols = [col for col in initial_data.columns if col not in 
                   ['dt', 'id_station', 'temperature', 'latitude', 'longitude', 'lat_norm', 'lon_norm']]
    
    # Build static feature matrix
    sensor_static = {}
    coord_features = {}
    for sid in all_sensors:
        sensor_data = initial_data[initial_data['id_station'] == sid].iloc[0]
        static_features = []
        for col in static_cols:
            if col in sensor_data:
                static_features.append(sensor_data[col])
        sensor_static[sid] = np.array(static_features, dtype=np.float32) if static_features else np.zeros(n_static, dtype=np.float32)
        
        # Store coordinates
        if 'lat_norm' in sensor_data and 'lon_norm' in sensor_data:
            coord_features[sid] = np.array([sensor_data['lat_norm'], sensor_data['lon_norm']], dtype=np.float32)
        else:
            coord_features[sid] = np.zeros(2, dtype=np.float32)
    
    # Initialize temperature state (using real values for first num_initial_steps)
    current_temps = np.zeros(n_sensors, dtype=np.float32)
    
    with torch.no_grad():
        for step_idx, timestamp in enumerate(timestamps):
            # Get ground truth for this timestep
            ts_data = initial_data[initial_data['dt'] == timestamp]
            
            # Update current temperatures
            if step_idx < num_initial_steps:
                # Use real data for initial steps
                for _, row in ts_data.iterrows():
                    sid = row['id_station']
                    if sid in sensor_to_idx:
                        idx = sensor_to_idx[sid]
                        # Mask the selected sensors
                        if sid in masked_sensors:
                            current_temps[idx] = 0.0
                        else:
                            current_temps[idx] = (row['temperature'] - y_mean) / y_std
            
            # Build input features for all sensors
            X_static = np.zeros((n_sensors, n_static), dtype=np.float32)
            X_time = np.zeros((n_sensors, n_time), dtype=np.float32)
            X_coord = np.zeros((n_sensors, 2), dtype=np.float32)
            
            # Temporal encoding
            time_encoding = prepare_temporal_encoding(timestamp)
            
            for sid in all_sensors:
                idx = sensor_to_idx[sid]
                X_static[idx] = sensor_static[sid]
                X_time[idx] = time_encoding
                X_coord[idx] = coord_features[sid]
            
            # Prepare input tensor
            X_temp = current_temps.reshape(-1, 1)
            X_static_time = np.concatenate([X_static, X_time], axis=1)
            
            # Convert to tensors
            x_static_time = torch.from_numpy(X_static_time).to(device)
            x_temp = torch.from_numpy(X_temp).to(device)
            x_coord = torch.from_numpy(X_coord).to(device)
            x_sem = torch.cat([x_static_time, x_temp], dim=1)
            
            # Forward pass
            output = model(x_sem, x_coord, edge_index).detach().cpu().numpy().flatten()
            
            # Denormalize predictions
            predictions_denorm = output * y_std + y_mean
            
            # Store predictions and ground truth
            for sid in all_sensors:
                idx = sensor_to_idx[sid]
                predictions[sid].append(predictions_denorm[idx])
                
                # Get ground truth
                sensor_data = ts_data[ts_data['id_station'] == sid]
                if not sensor_data.empty:
                    ground_truth[sid].append(sensor_data['temperature'].values[0])
                else:
                    ground_truth[sid].append(np.nan)
            
            # Update temperature state for next iteration (after initial steps)
            if step_idx >= num_initial_steps - 1:
                # Use predictions for next step
                current_temps = output.copy()
    
    # Convert lists to arrays
    predictions = {sid: np.array(vals) for sid, vals in predictions.items()}
    ground_truth = {sid: np.array(vals) for sid, vals in ground_truth.items()}
    
    return predictions, ground_truth


def build_simple_edge_index(n_sensors: int, k_neighbors: int = 6) -> torch.Tensor:
    """
    Build a simple k-nearest neighbor graph based on sensor indices
    This is a simplified version - in practice, you'd use spatial distances
    """
    edges_src = []
    edges_dst = []
    
    for i in range(n_sensors):
        # Connect to k nearest neighbors (circularly)
        for j in range(1, k_neighbors + 1):
            neighbor = (i + j) % n_sensors
            edges_src.append(i)
            edges_dst.append(neighbor)
            # Make it undirected
            edges_src.append(neighbor)
            edges_dst.append(i)
    
    return torch.tensor([edges_src, edges_dst], dtype=torch.long)


def evaluate_forecast(
    predictions: Dict,
    ground_truth: Dict,
    masked_sensors: List[str],
    num_initial_steps: int
) -> Dict:
    """
    Evaluate predictions separately for masked and unmasked sensors
    
    Returns metrics dictionary
    """
    # Split into masked and unmasked sensors
    masked_preds = []
    masked_true = []
    unmasked_preds = []
    unmasked_true = []
    
    for sensor_id, pred_vals in predictions.items():
        true_vals = ground_truth[sensor_id]
        
        # Only evaluate after initial steps
        pred_vals = pred_vals[num_initial_steps:]
        true_vals = true_vals[num_initial_steps:]
        
        # Remove NaN values
        valid_mask = ~np.isnan(true_vals)
        pred_vals = pred_vals[valid_mask]
        true_vals = true_vals[valid_mask]
        
        if len(pred_vals) == 0:
            continue
        
        if sensor_id in masked_sensors:
            masked_preds.extend(pred_vals)
            masked_true.extend(true_vals)
        else:
            unmasked_preds.extend(pred_vals)
            unmasked_true.extend(true_vals)
    
    # Calculate metrics
    metrics = {}
    
    if len(unmasked_preds) > 0:
        unmasked_preds = np.array(unmasked_preds)
        unmasked_true = np.array(unmasked_true)
        metrics['unmasked_mae'] = np.mean(np.abs(unmasked_preds - unmasked_true))
        metrics['unmasked_rmse'] = np.sqrt(np.mean((unmasked_preds - unmasked_true)**2))
    else:
        metrics['unmasked_mae'] = np.nan
        metrics['unmasked_rmse'] = np.nan
    
    if len(masked_preds) > 0:
        masked_preds = np.array(masked_preds)
        masked_true = np.array(masked_true)
        metrics['masked_mae'] = np.mean(np.abs(masked_preds - masked_true))
        metrics['masked_rmse'] = np.sqrt(np.mean((masked_preds - masked_true)**2))
    else:
        metrics['masked_mae'] = np.nan
        metrics['masked_rmse'] = np.nan
    
    return metrics


def plot_forecast_results(
    predictions: Dict,
    ground_truth: Dict,
    masked_sensors: List[str],
    num_initial_steps: int,
    output_dir: str,
    run_id: int = 0
):
    """
    Generate plots for masked and unmasked sensors
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot masked sensors
    n_masked = len(masked_sensors)
    if n_masked > 0:
        fig, axes = plt.subplots(n_masked, 1, figsize=(12, 4 * n_masked))
        if n_masked == 1:
            axes = [axes]
        
        for idx, sensor_id in enumerate(masked_sensors):
            ax = axes[idx]
            pred = predictions[sensor_id]
            true = ground_truth[sensor_id]
            
            # Time steps
            steps = np.arange(len(pred))
            
            # Plot
            ax.plot(steps, true, 'o-', label='True Temperature', color='#333333', linewidth=2, markersize=4)
            ax.plot(steps, pred, 's--', label='Predicted Temperature', color='#1f77b4', linewidth=2, markersize=4)
            
            # Mark initial steps
            ax.axvline(x=num_initial_steps - 0.5, color='red', linestyle='--', linewidth=2, label='Forecast Start')
            ax.axvspan(0, num_initial_steps - 0.5, alpha=0.2, color='gray', label='Initial Steps')
            
            # Calculate MAE for forecast period
            forecast_pred = pred[num_initial_steps:]
            forecast_true = true[num_initial_steps:]
            valid_mask = ~np.isnan(forecast_true)
            if valid_mask.sum() > 0:
                mae = np.mean(np.abs(forecast_pred[valid_mask] - forecast_true[valid_mask]))
                ax.text(0.02, 0.98, f'Forecast MAE: {mae:.3f}°C', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Temperature (°C)', fontsize=12)
            ax.set_title(f'Masked Sensor: {sensor_id}', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'forecast_masked_sensors_run{run_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot a few unmasked sensors
    unmasked_sensors = [s for s in predictions.keys() if s not in masked_sensors]
    n_to_plot = min(3, len(unmasked_sensors))
    
    if n_to_plot > 0:
        selected_unmasked = random.sample(unmasked_sensors, n_to_plot)
        
        fig, axes = plt.subplots(n_to_plot, 1, figsize=(12, 4 * n_to_plot))
        if n_to_plot == 1:
            axes = [axes]
        
        for idx, sensor_id in enumerate(selected_unmasked):
            ax = axes[idx]
            pred = predictions[sensor_id]
            true = ground_truth[sensor_id]
            
            steps = np.arange(len(pred))
            
            ax.plot(steps, true, 'o-', label='True Temperature', color='#333333', linewidth=2, markersize=4)
            ax.plot(steps, pred, 's--', label='Predicted Temperature', color='#2ca02c', linewidth=2, markersize=4)
            
            ax.axvline(x=num_initial_steps - 0.5, color='red', linestyle='--', linewidth=2, label='Forecast Start')
            ax.axvspan(0, num_initial_steps - 0.5, alpha=0.2, color='gray', label='Initial Steps')
            
            # Calculate MAE for forecast period
            forecast_pred = pred[num_initial_steps:]
            forecast_true = true[num_initial_steps:]
            valid_mask = ~np.isnan(forecast_true)
            if valid_mask.sum() > 0:
                mae = np.mean(np.abs(forecast_pred[valid_mask] - forecast_true[valid_mask]))
                ax.text(0.02, 0.98, f'Forecast MAE: {mae:.3f}°C', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Temperature (°C)', fontsize=12)
            ax.set_title(f'Unmasked Sensor: {sensor_id}', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'forecast_unmasked_sensors_run{run_id}.png'), dpi=300, bbox_inches='tight')
        plt.close()


def run_forecast_experiment(
    model_path: str,
    data_path: str,
    output_dir: str,
    num_initial_steps: int = 6,
    total_steps: int = 20,
    n_runs: int = 5,
    seed: int = 42
):
    """
    Run complete forecast mode experiment with multiple random maskings
    """
    print("=" * 80)
    print("FORECAST MODE TESTING")
    print("=" * 80)
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load model and data
    print(f"\n1. Loading model from: {model_path}")
    print(f"   Loading data from: {data_path}")
    model, df, y_mean, y_std, n_static, n_time = load_model_and_data(model_path, data_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"   Device: {device}")
    
    # Select test sequence
    print(f"\n2. Selecting test sequence (min length: {total_steps})")
    test_seq = select_test_sequence(df, min_length=total_steps)
    print(f"   Selected {len(test_seq['dt'].unique())} timestamps")
    print(f"   Date range: {test_seq['dt'].min()} to {test_seq['dt'].max()}")
    print(f"   Sensors available: {len(test_seq['id_station'].unique())}")
    
    # Get all sensors
    all_sensors = sorted(test_seq['id_station'].unique())
    
    # Build edge index (simplified version - ideally load from training)
    n_sensors = len(all_sensors)
    edge_index = build_simple_edge_index(n_sensors, k_neighbors=6).to(device)
    print(f"   Built graph with {n_sensors} nodes and {edge_index.shape[1]} edges")
    
    # Run multiple experiments with different random maskings
    print(f"\n3. Running {n_runs} forecast experiments with different masked sensors")
    all_metrics = []
    
    for run_id in range(n_runs):
        print(f"\n   --- Run {run_id + 1}/{n_runs} ---")
        
        # Randomly select 3 sensors to mask
        masked_sensors = random.sample(all_sensors, min(3, len(all_sensors)))
        print(f"   Masked sensors: {masked_sensors}")
        
        # Run forecast
        print(f"   Running autoregressive forecast rollout...")
        predictions, ground_truth = forecast_rollout(
            model=model,
            initial_data=test_seq,
            masked_sensors=masked_sensors,
            num_initial_steps=num_initial_steps,
            total_steps=total_steps,
            edge_index=edge_index,
            y_mean=y_mean,
            y_std=y_std,
            n_static=n_static,
            n_time=n_time,
            device=device
        )
        
        # Evaluate
        metrics = evaluate_forecast(predictions, ground_truth, masked_sensors, num_initial_steps)
        all_metrics.append(metrics)
        
        print(f"   Results:")
        print(f"     Unmasked sensors - MAE: {metrics['unmasked_mae']:.3f}°C, RMSE: {metrics['unmasked_rmse']:.3f}°C")
        print(f"     Masked sensors   - MAE: {metrics['masked_mae']:.3f}°C, RMSE: {metrics['masked_rmse']:.3f}°C")
        
        # Generate plots
        plot_forecast_results(predictions, ground_truth, masked_sensors, num_initial_steps, output_dir, run_id)
    
    # Compute average metrics
    print(f"\n4. Summary across {n_runs} runs:")
    print("=" * 80)
    
    avg_unmasked_mae = np.nanmean([m['unmasked_mae'] for m in all_metrics])
    avg_unmasked_rmse = np.nanmean([m['unmasked_rmse'] for m in all_metrics])
    avg_masked_mae = np.nanmean([m['masked_mae'] for m in all_metrics])
    avg_masked_rmse = np.nanmean([m['masked_rmse'] for m in all_metrics])
    
    std_unmasked_mae = np.nanstd([m['unmasked_mae'] for m in all_metrics])
    std_unmasked_rmse = np.nanstd([m['unmasked_rmse'] for m in all_metrics])
    std_masked_mae = np.nanstd([m['masked_mae'] for m in all_metrics])
    std_masked_rmse = np.nanstd([m['masked_rmse'] for m in all_metrics])
    
    print(f"\nUnmasked Sensors (visible during input):")
    print(f"  MAE:  {avg_unmasked_mae:.3f} ± {std_unmasked_mae:.3f}°C")
    print(f"  RMSE: {avg_unmasked_rmse:.3f} ± {std_unmasked_rmse:.3f}°C")
    
    print(f"\nMasked Sensors (hidden from input):")
    print(f"  MAE:  {avg_masked_mae:.3f} ± {std_masked_mae:.3f}°C")
    print(f"  RMSE: {avg_masked_rmse:.3f} ± {std_masked_rmse:.3f}°C")
    
    print(f"\nPlots saved to: {output_dir}")
    print("=" * 80)
    
    # Save summary
    summary_path = os.path.join(output_dir, 'forecast_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("FORECAST MODE TESTING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {data_path}\n")
        f.write(f"Number of initial steps: {num_initial_steps}\n")
        f.write(f"Total steps predicted: {total_steps}\n")
        f.write(f"Number of runs: {n_runs}\n\n")
        f.write("AVERAGE METRICS:\n")
        f.write(f"Unmasked Sensors:\n")
        f.write(f"  MAE:  {avg_unmasked_mae:.3f} ± {std_unmasked_mae:.3f}°C\n")
        f.write(f"  RMSE: {avg_unmasked_rmse:.3f} ± {std_unmasked_rmse:.3f}°C\n\n")
        f.write(f"Masked Sensors:\n")
        f.write(f"  MAE:  {avg_masked_mae:.3f} ± {std_masked_mae:.3f}°C\n")
        f.write(f"  RMSE: {avg_masked_rmse:.3f} ± {std_masked_rmse:.3f}°C\n\n")
        f.write("INDIVIDUAL RUNS:\n")
        for i, m in enumerate(all_metrics):
            f.write(f"\nRun {i+1}:\n")
            f.write(f"  Unmasked - MAE: {m['unmasked_mae']:.3f}, RMSE: {m['unmasked_rmse']:.3f}\n")
            f.write(f"  Masked   - MAE: {m['masked_mae']:.3f}, RMSE: {m['masked_rmse']:.3f}\n")
    
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Test GNN model in forecast mode')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--initial-steps', type=int, default=6, help='Number of initial steps to use')
    parser.add_argument('--total-steps', type=int, default=20, help='Total number of steps to predict')
    parser.add_argument('--n-runs', type=int, default=5, help='Number of runs with different random maskings')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_forecast_experiment(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_initial_steps=args.initial_steps,
        total_steps=args.total_steps,
        n_runs=args.n_runs,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
