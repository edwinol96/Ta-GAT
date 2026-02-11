#!/usr/bin/env python3
"""
SpatiotemporalGAT with Masking Strategy and Forecast Mode Testing

This script implements:
1. A GNN+LSTM model (SpatiotemporalGAT) for temperature prediction
2. Training with masking strategy (masks 3 sensors and reconstructs values)
3. Forecast mode testing with autoregressive rollout
4. Evaluation and visualization for both masked and unmasked sensors

Usage:
    python spatiotemporal_gat_forecast.py
"""

import os
import random
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, atan2

# Plotting defaults
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.alpha': 0.5})
plt.rcParams['font.family'] = 'serif'

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Configuration for the SpatiotemporalGAT model."""
    
    # Input paths - using repository data locations
    csv_path: str = "original_2023/filtered_2023_aug_hourly.csv"
    neighbor_csv_path: str = "original_2023/nearest_6_neighbors.csv"
    output_dir: str = "output_spatiotemporal_gat"
    
    # Data settings
    seq_len: int = 6          # Input sequence length (time steps)
    pred_len: int = 12        # Prediction horizon
    train_ratio: float = 0.8
    
    # Model settings
    hidden_dim: int = 64
    lstm_hidden: int = 128
    num_lstm_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    
    # Training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 15
    
    # Masking settings
    num_masked_sensors: int = 3
    
    # Evaluation settings
    num_test_trials: int = 5  # Number of different random mask choices to evaluate
    
    # Graph settings
    k_neighbors: int = 6
    
    seed: int = 42


# ============================================================================
# Utility Functions
# ============================================================================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on earth (in km)."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_sensor_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess sensor data from CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # Standardize column names
    col_mapping = {
        'station_id': 'id_station',
        'date': 'datetime',
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime', 'temperature'])
    
    # Sort by datetime
    df = df.sort_values(['datetime', 'id_station']).reset_index(drop=True)
    
    return df


def build_adjacency_from_neighbors(df: pd.DataFrame, neighbor_csv: str, k: int = 6) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build adjacency matrix from neighbor CSV or by computing k-nearest neighbors."""
    stations = df['id_station'].unique()
    station_to_idx = {s: i for i, s in enumerate(stations)}
    n_stations = len(stations)
    
    adj = np.zeros((n_stations, n_stations), dtype=np.float32)
    
    if os.path.exists(neighbor_csv):
        # Load from neighbor CSV
        neighbors_df = pd.read_csv(neighbor_csv)
        for _, row in neighbors_df.iterrows():
            target = row['target_id']
            neighbor = row['neighbor_id']
            if target in station_to_idx and neighbor in station_to_idx:
                i, j = station_to_idx[target], station_to_idx[neighbor]
                dist = row.get('distance_km', 1.0)
                weight = 1.0 / (dist + 0.1)  # Inverse distance weighting
                adj[i, j] = weight
                adj[j, i] = weight  # Symmetric
    else:
        # Compute k-nearest neighbors based on coordinates
        station_coords = df.groupby('id_station').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        coords = station_coords[['latitude', 'longitude']].values
        nbrs = NearestNeighbors(n_neighbors=min(k + 1, n_stations)).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        for i in range(n_stations):
            for j_idx in range(1, len(indices[i])):  # Skip self
                j = indices[i][j_idx]
                dist = distances[i][j_idx]
                weight = 1.0 / (dist * 111 + 0.1)  # Approximate km conversion
                adj[i, j] = weight
                adj[j, i] = weight
    
    # Normalize adjacency
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj = adj / row_sums
    
    return adj, station_to_idx


def prepare_sequences(
    df: pd.DataFrame,
    station_to_idx: Dict[str, int],
    seq_len: int,
    pred_len: int
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], np.ndarray, np.ndarray]:
    """
    Prepare sequences for training/testing.
    
    Returns:
        X: Input sequences (num_samples, num_stations, seq_len)
        Y: Target values (num_samples, num_stations, pred_len)
        timestamps: List of timestamps for each sample
        temp_mean, temp_std: Normalization statistics
    """
    n_stations = len(station_to_idx)
    timestamps = sorted(df['datetime'].unique())
    
    # Create pivot table: rows=timestamps, columns=stations
    pivot = df.pivot_table(
        index='datetime',
        columns='id_station',
        values='temperature',
        aggfunc='mean'
    ).reindex(columns=[s for s in station_to_idx.keys()])
    
    # Fill missing values with interpolation
    pivot = pivot.interpolate(method='linear', axis=0).bfill().ffill()
    
    data = pivot.values  # (time_steps, num_stations)
    
    # Normalize
    temp_mean = np.nanmean(data)
    temp_std = np.nanstd(data)
    temp_std = max(temp_std, 1e-6)
    data_norm = (data - temp_mean) / temp_std
    
    # Create sequences
    X_list, Y_list, ts_list = [], [], []
    
    for i in range(len(data_norm) - seq_len - pred_len + 1):
        x_seq = data_norm[i:i + seq_len].T  # (num_stations, seq_len)
        y_seq = data_norm[i + seq_len:i + seq_len + pred_len].T  # (num_stations, pred_len)
        
        if not np.any(np.isnan(x_seq)) and not np.any(np.isnan(y_seq)):
            X_list.append(x_seq)
            Y_list.append(y_seq)
            ts_list.append(pivot.index[i + seq_len - 1])
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    return X, Y, ts_list, temp_mean, temp_std


# ============================================================================
# Model Definition
# ============================================================================
class SpatiotemporalGAT(nn.Module):
    """
    Graph Attention Network with LSTM for spatiotemporal prediction.
    
    Architecture:
    1. GAT layers process spatial relationships at each time step
    2. LSTM processes temporal sequences for each node
    3. Output layer predicts future values
    """
    
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # GAT layers for spatial processing
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, num_nodes, seq_len)
            edge_index: Graph edges (2, num_edges)
            mask: Optional mask for nodes (batch, num_nodes) - 1 for masked, 0 for observed
        
        Returns:
            Predictions (batch, num_nodes, pred_len)
        """
        batch_size, num_nodes, seq_len = x.shape
        
        # Apply mask - set masked sensor values to 0
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x * (1 - mask_expanded)
        
        # Process each time step through GAT
        temporal_features = []
        for t in range(seq_len):
            x_t = x[:, :, t:t+1]  # (batch, num_nodes, 1)
            
            # Reshape for GAT: (batch * num_nodes, 1)
            x_flat = x_t.view(-1, 1)
            
            # Project input
            h = self.input_proj(x_flat)  # (batch * num_nodes, hidden_dim)
            
            # Create batched edge_index
            edge_batch = edge_index.clone()
            all_edges = []
            for b in range(batch_size):
                offset = b * num_nodes
                all_edges.append(edge_batch + offset)
            batched_edges = torch.cat(all_edges, dim=1)
            
            # GAT layers
            h = F.elu(self.gat1(h, batched_edges))
            h = self.dropout(h)
            h = self.gat2(h, batched_edges)
            h = self.layer_norm(h)
            
            # Reshape back: (batch, num_nodes, hidden_dim)
            h = h.view(batch_size, num_nodes, -1)
            temporal_features.append(h)
        
        # Stack temporal features: (batch, num_nodes, seq_len, hidden_dim)
        temporal_features = torch.stack(temporal_features, dim=2)
        
        # Process each node's temporal sequence through LSTM
        outputs = []
        for n in range(num_nodes):
            node_seq = temporal_features[:, n, :, :]  # (batch, seq_len, hidden_dim)
            lstm_out, _ = self.lstm(node_seq)  # (batch, seq_len, lstm_hidden)
            last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden)
            pred = self.fc_out(last_out)  # (batch, pred_len)
            outputs.append(pred)
        
        # Stack outputs: (batch, num_nodes, pred_len)
        output = torch.stack(outputs, dim=1)
        
        return output


# ============================================================================
# Training Functions
# ============================================================================
def create_graph_data(
    adj_matrix: np.ndarray,
    device: torch.device
) -> torch.Tensor:
    """Create edge_index from adjacency matrix."""
    # Get edge indices where adjacency > 0
    src, dst = np.where(adj_matrix > 0)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index


def train_epoch(
    model: SpatiotemporalGAT,
    train_loader: DataLoader,
    edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_masked: int = 3
) -> float:
    """Train for one epoch with masking strategy."""
    model.train()
    total_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        batch_size, num_nodes, _ = batch_x.shape
        
        # Create random masks - mask 'num_masked' sensors per sample
        mask = torch.zeros(batch_size, num_nodes, device=device)
        for b in range(batch_size):
            masked_indices = torch.randperm(num_nodes)[:num_masked]
            mask[b, masked_indices] = 1.0
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch_x, edge_index, mask=mask)
        
        # Compute loss - focus on masked sensors for reconstruction
        # Loss = reconstruction loss on masked + prediction loss on all
        mask_expanded = mask.unsqueeze(-1).expand_as(batch_y)
        
        # Masked reconstruction loss
        masked_loss = F.mse_loss(pred * mask_expanded, batch_y * mask_expanded, reduction='sum')
        masked_count = mask_expanded.sum() + 1e-6
        masked_loss = masked_loss / masked_count
        
        # Overall prediction loss (all sensors)
        pred_loss = F.mse_loss(pred, batch_y)
        
        # Combined loss
        loss = 0.5 * masked_loss + 0.5 * pred_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    model: SpatiotemporalGAT,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    num_masked: int = 3
) -> Tuple[float, float, float]:
    """
    Evaluate model with masking.
    
    Returns:
        total_loss, masked_mae, unmasked_mae
    """
    model.eval()
    total_loss = 0.0
    masked_errors = []
    unmasked_errors = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            batch_size, num_nodes, _ = batch_x.shape
            
            # Create random masks
            mask = torch.zeros(batch_size, num_nodes, device=device)
            for b in range(batch_size):
                masked_indices = torch.randperm(num_nodes)[:num_masked]
                mask[b, masked_indices] = 1.0
            
            pred = model(batch_x, edge_index, mask=mask)
            loss = F.mse_loss(pred, batch_y)
            total_loss += loss.item()
            
            # Compute errors for masked and unmasked separately
            mask_expanded = mask.unsqueeze(-1).expand_as(batch_y)
            
            # MAE for masked sensors
            masked_diff = torch.abs(pred - batch_y) * mask_expanded
            masked_sum = masked_diff.sum() / (mask_expanded.sum() + 1e-6)
            masked_errors.append(masked_sum.item())
            
            # MAE for unmasked sensors
            unmasked_mask = 1 - mask_expanded
            unmasked_diff = torch.abs(pred - batch_y) * unmasked_mask
            unmasked_sum = unmasked_diff.sum() / (unmasked_mask.sum() + 1e-6)
            unmasked_errors.append(unmasked_sum.item())
    
    return (
        total_loss / len(val_loader),
        np.mean(masked_errors),
        np.mean(unmasked_errors)
    )


def train_model(
    model: SpatiotemporalGAT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index: torch.Tensor,
    config: Config,
    device: torch.device
) -> Dict[str, List[float]]:
    """Train the model with early stopping."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'masked_mae': [],
        'unmasked_mae': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(
            model, train_loader, edge_index, optimizer, device,
            num_masked=config.num_masked_sensors
        )
        
        val_loss, masked_mae, unmasked_mae = evaluate(
            model, val_loader, edge_index, device,
            num_masked=config.num_masked_sensors
        )
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['masked_mae'].append(masked_mae)
        history['unmasked_mae'].append(unmasked_mae)
        
        if epoch % 10 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"Masked MAE={masked_mae:.4f}, Unmasked MAE={unmasked_mae:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


# ============================================================================
# Forecast Mode Functions
# ============================================================================
def forecast_with_masking(
    model: SpatiotemporalGAT,
    initial_sequence: np.ndarray,
    edge_index: torch.Tensor,
    masked_indices: List[int],
    total_pred_steps: int,
    device: torch.device,
    temp_mean: float,
    temp_std: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform autoregressive forecasting with masked sensors.
    
    Args:
        model: Trained model
        initial_sequence: Initial input (num_nodes, seq_len) - normalized
        edge_index: Graph edges
        masked_indices: Indices of sensors to mask
        total_pred_steps: Total number of steps to predict
        device: Torch device
        temp_mean, temp_std: Normalization statistics
    
    Returns:
        predictions: (num_nodes, total_pred_steps) - original scale
        masked_predictions: (len(masked_indices), total_pred_steps) - original scale
    """
    model.eval()
    
    num_nodes, seq_len = initial_sequence.shape
    pred_len = model.pred_len
    
    # Create mask tensor
    mask = torch.zeros(1, num_nodes, device=device)
    for idx in masked_indices:
        mask[0, idx] = 1.0
    
    # Apply mask to initial sequence
    current_seq = initial_sequence.copy()
    for idx in masked_indices:
        current_seq[idx, :] = 0.0
    
    all_predictions = []
    
    steps_predicted = 0
    while steps_predicted < total_pred_steps:
        # Prepare input
        x = torch.tensor(current_seq, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(x, edge_index, mask=mask)  # (1, num_nodes, pred_len)
        
        pred_np = pred.cpu().numpy().squeeze(0)  # (num_nodes, pred_len)
        
        # Take predictions for remaining steps
        steps_to_add = min(pred_len, total_pred_steps - steps_predicted)
        all_predictions.append(pred_np[:, :steps_to_add])
        steps_predicted += steps_to_add
        
        # Update sequence for next iteration (autoregressive)
        if steps_predicted < total_pred_steps:
            # Shift sequence and add predictions
            new_seq = np.concatenate([
                current_seq[:, steps_to_add:],
                pred_np[:, :steps_to_add]
            ], axis=1)
            
            # Ensure masked sensors stay masked
            for idx in masked_indices:
                new_seq[idx, :seq_len - steps_to_add] = current_seq[idx, steps_to_add:]
            
            current_seq = new_seq
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=1)[:, :total_pred_steps]
    
    # Convert to original scale
    predictions_original = predictions * temp_std + temp_mean
    
    # Extract masked sensor predictions
    masked_predictions = predictions_original[masked_indices, :]
    
    return predictions_original, masked_predictions


def run_forecast_evaluation(
    model: SpatiotemporalGAT,
    test_data: Tuple[np.ndarray, np.ndarray],
    edge_index: torch.Tensor,
    config: Config,
    device: torch.device,
    temp_mean: float,
    temp_std: float,
    station_names: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Run forecast mode evaluation with multiple random mask choices.
    
    Returns:
        Dictionary with evaluation results
    """
    X_test, Y_test = test_data
    num_samples, num_nodes, seq_len = X_test.shape
    pred_len = Y_test.shape[2]
    
    all_results = []
    
    for trial in range(config.num_test_trials):
        print(f"\n=== Forecast Trial {trial + 1}/{config.num_test_trials} ===")
        
        # Pick a random test sample
        sample_idx = random.randint(0, num_samples - 1)
        
        # Randomly select 3 sensors to mask
        masked_indices = random.sample(range(num_nodes), config.num_masked_sensors)
        unmasked_indices = [i for i in range(num_nodes) if i not in masked_indices]
        
        print(f"Sample index: {sample_idx}")
        print(f"Masked sensors: {[station_names[i] for i in masked_indices]}")
        
        # Get initial sequence and ground truth
        initial_seq = X_test[sample_idx]  # (num_nodes, seq_len)
        ground_truth = Y_test[sample_idx]  # (num_nodes, pred_len) - normalized
        
        # Run forecast
        predictions, masked_preds = forecast_with_masking(
            model, initial_seq, edge_index, masked_indices,
            pred_len, device, temp_mean, temp_std
        )
        
        # Convert ground truth to original scale
        ground_truth_original = ground_truth * temp_std + temp_mean
        
        # Compute errors
        # For masked sensors
        masked_gt = ground_truth_original[masked_indices, :]
        masked_mae = np.mean(np.abs(masked_preds - masked_gt))
        masked_rmse = np.sqrt(np.mean((masked_preds - masked_gt) ** 2))
        
        # For unmasked sensors
        unmasked_preds = predictions[unmasked_indices, :]
        unmasked_gt = ground_truth_original[unmasked_indices, :]
        unmasked_mae = np.mean(np.abs(unmasked_preds - unmasked_gt))
        unmasked_rmse = np.sqrt(np.mean((unmasked_preds - unmasked_gt) ** 2))
        
        print(f"Masked sensors - MAE: {masked_mae:.3f}°C, RMSE: {masked_rmse:.3f}°C")
        print(f"Unmasked sensors - MAE: {unmasked_mae:.3f}°C, RMSE: {unmasked_rmse:.3f}°C")
        
        result = {
            'trial': trial,
            'sample_idx': sample_idx,
            'masked_indices': masked_indices,
            'masked_mae': masked_mae,
            'masked_rmse': masked_rmse,
            'unmasked_mae': unmasked_mae,
            'unmasked_rmse': unmasked_rmse,
            'predictions': predictions,
            'ground_truth': ground_truth_original,
            'masked_preds': masked_preds,
            'masked_gt': masked_gt
        }
        all_results.append(result)
        
        # Plot results for this trial
        plot_forecast_results(
            result, station_names, pred_len,
            os.path.join(output_dir, f'trial_{trial + 1}')
        )
    
    # Summary statistics
    summary = {
        'masked_mae_mean': np.mean([r['masked_mae'] for r in all_results]),
        'masked_mae_std': np.std([r['masked_mae'] for r in all_results]),
        'masked_rmse_mean': np.mean([r['masked_rmse'] for r in all_results]),
        'masked_rmse_std': np.std([r['masked_rmse'] for r in all_results]),
        'unmasked_mae_mean': np.mean([r['unmasked_mae'] for r in all_results]),
        'unmasked_mae_std': np.std([r['unmasked_mae'] for r in all_results]),
        'unmasked_rmse_mean': np.mean([r['unmasked_rmse'] for r in all_results]),
        'unmasked_rmse_std': np.std([r['unmasked_rmse'] for r in all_results]),
        'trials': all_results
    }
    
    return summary


def plot_forecast_results(
    result: Dict[str, Any],
    station_names: List[str],
    pred_len: int,
    output_dir: str
) -> None:
    """Plot forecast results for a single trial."""
    os.makedirs(output_dir, exist_ok=True)
    
    masked_indices = result['masked_indices']
    predictions = result['predictions']
    ground_truth = result['ground_truth']
    
    time_steps = np.arange(1, pred_len + 1)
    
    # Plot masked sensors
    fig, axes = plt.subplots(len(masked_indices), 1, figsize=(12, 4 * len(masked_indices)))
    if len(masked_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(masked_indices):
        ax = axes[i]
        ax.plot(time_steps, ground_truth[idx], 'b-o', label='True', linewidth=2, markersize=6)
        ax.plot(time_steps, predictions[idx], 'r--s', label='Predicted', linewidth=2, markersize=6)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Masked Sensor: {station_names[idx][:30]}...' if len(station_names[idx]) > 30 else f'Masked Sensor: {station_names[idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'masked_sensors_forecast.png'), dpi=150)
    plt.close()
    
    # Plot a few unmasked sensors (first 3)
    unmasked_indices = [i for i in range(len(station_names)) if i not in masked_indices][:3]
    
    fig, axes = plt.subplots(len(unmasked_indices), 1, figsize=(12, 4 * len(unmasked_indices)))
    if len(unmasked_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(unmasked_indices):
        ax = axes[i]
        ax.plot(time_steps, ground_truth[idx], 'b-o', label='True', linewidth=2, markersize=6)
        ax.plot(time_steps, predictions[idx], 'g--s', label='Predicted', linewidth=2, markersize=6)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Unmasked Sensor: {station_names[idx][:30]}...' if len(station_names[idx]) > 30 else f'Unmasked Sensor: {station_names[idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unmasked_sensors_forecast.png'), dpi=150)
    plt.close()


def plot_training_history(history: Dict[str, List[float]], output_dir: str) -> None:
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training and validation loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE for masked and unmasked
    ax = axes[0, 1]
    ax.plot(history['masked_mae'], label='Masked MAE', linewidth=2)
    ax.plot(history['unmasked_mae'], label='Unmasked MAE', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (normalized)')
    ax.set_title('MAE by Sensor Type')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale loss
    ax = axes[1, 0]
    ax.semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    ax.semilogy(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training and Validation Loss (Log Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = (
        f"Final Training Metrics\n"
        f"{'='*30}\n"
        f"Final Train Loss: {history['train_loss'][-1]:.4f}\n"
        f"Final Val Loss: {history['val_loss'][-1]:.4f}\n"
        f"Final Masked MAE: {history['masked_mae'][-1]:.4f}\n"
        f"Final Unmasked MAE: {history['unmasked_mae'][-1]:.4f}\n"
        f"{'='*30}\n"
        f"Best Val Loss: {min(history['val_loss']):.4f}\n"
        f"Best Epoch: {np.argmin(history['val_loss'])}"
    )
    ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    plt.close()


def print_summary(summary: Dict[str, Any]) -> None:
    """Print summary of forecast evaluation."""
    print("\n" + "=" * 60)
    print("FORECAST EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nMasked Sensors (hidden during prediction):")
    print(f"  MAE:  {summary['masked_mae_mean']:.3f} ± {summary['masked_mae_std']:.3f} °C")
    print(f"  RMSE: {summary['masked_rmse_mean']:.3f} ± {summary['masked_rmse_std']:.3f} °C")
    print(f"\nUnmasked Sensors (visible during prediction):")
    print(f"  MAE:  {summary['unmasked_mae_mean']:.3f} ± {summary['unmasked_mae_std']:.3f} °C")
    print(f"  RMSE: {summary['unmasked_rmse_mean']:.3f} ± {summary['unmasked_rmse_std']:.3f} °C")
    print("=" * 60)


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main function to run the SpatiotemporalGAT training and evaluation."""
    
    # Get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    config = Config()
    config.csv_path = os.path.join(base_dir, config.csv_path)
    config.neighbor_csv_path = os.path.join(base_dir, config.neighbor_csv_path)
    config.output_dir = os.path.join(base_dir, config.output_dir)
    
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n1. Loading sensor data...")
    df = load_sensor_data(config.csv_path)
    print(f"   Loaded {len(df)} records from {df['id_station'].nunique()} stations")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Build adjacency
    print("\n2. Building graph structure...")
    adj_matrix, station_to_idx = build_adjacency_from_neighbors(
        df, config.neighbor_csv_path, config.k_neighbors
    )
    station_names = list(station_to_idx.keys())
    num_nodes = len(station_names)
    print(f"   Number of nodes: {num_nodes}")
    print(f"   Number of edges: {(adj_matrix > 0).sum()}")
    
    # Create edge index for PyTorch Geometric
    edge_index = create_graph_data(adj_matrix, device)
    
    # Prepare sequences
    print("\n3. Preparing sequences...")
    X, Y, timestamps, temp_mean, temp_std = prepare_sequences(
        df, station_to_idx, config.seq_len, config.pred_len
    )
    print(f"   Total samples: {len(X)}")
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {Y.shape}")
    print(f"   Temperature stats: mean={temp_mean:.2f}, std={temp_std:.2f}")
    
    # Train/test split
    split_idx = int(len(X) * config.train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Create data loaders
    train_dataset = list(zip(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    ))
    test_dataset = list(zip(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32)
    ))
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    print("\n4. Initializing model...")
    model = SpatiotemporalGAT(
        num_nodes=num_nodes,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        hidden_dim=config.hidden_dim,
        lstm_hidden=config.lstm_hidden,
        num_lstm_layers=config.num_lstm_layers,
        heads=config.heads,
        dropout=config.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n5. Training model with masking strategy...")
    print(f"   Masking {config.num_masked_sensors} sensors during training")
    
    history = train_model(
        model, train_loader, val_loader, edge_index, config, device
    )
    
    # Plot training history
    plot_training_history(history, config.output_dir)
    print(f"   Training history saved to {config.output_dir}/training_history.png")
    
    # Save model
    model_path = os.path.join(config.output_dir, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'station_names': station_names
    }, model_path)
    print(f"   Model saved to {model_path}")
    
    # Run forecast evaluation
    print("\n6. Running forecast mode evaluation...")
    summary = run_forecast_evaluation(
        model,
        (X_test, Y_test),
        edge_index,
        config,
        device,
        temp_mean,
        temp_std,
        station_names,
        config.output_dir
    )
    
    # Print summary
    print_summary(summary)
    
    # Save summary to file
    summary_path = os.path.join(config.output_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("FORECAST EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Sequence length: {config.seq_len}\n")
        f.write(f"  Prediction length: {config.pred_len}\n")
        f.write(f"  Number of masked sensors: {config.num_masked_sensors}\n")
        f.write(f"  Number of test trials: {config.num_test_trials}\n\n")
        f.write(f"Masked Sensors (hidden during prediction):\n")
        f.write(f"  MAE:  {summary['masked_mae_mean']:.3f} ± {summary['masked_mae_std']:.3f} °C\n")
        f.write(f"  RMSE: {summary['masked_rmse_mean']:.3f} ± {summary['masked_rmse_std']:.3f} °C\n\n")
        f.write(f"Unmasked Sensors (visible during prediction):\n")
        f.write(f"  MAE:  {summary['unmasked_mae_mean']:.3f} ± {summary['unmasked_mae_std']:.3f} °C\n")
        f.write(f"  RMSE: {summary['unmasked_rmse_mean']:.3f} ± {summary['unmasked_rmse_std']:.3f} °C\n")
    
    print(f"\n   Summary saved to {summary_path}")
    print("\n✓ Done!")
    
    return model, summary


if __name__ == "__main__":
    model, summary = main()
