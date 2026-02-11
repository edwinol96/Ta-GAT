# Forecast Mode Testing for GNN Temperature Prediction

This document explains how to use the forecast mode testing functionality for the trained GNN model.

## Overview

The forecast mode testing evaluates the model's ability to:
1. **Predict future temperatures** in an autoregressive manner (using its own predictions as input)
2. **Reconstruct masked sensor readings** for sensors it never saw during the forecast period
3. **Maintain accuracy** on visible (unmasked) sensors during the forecast

## Key Features

- **Autoregressive Rollout**: Uses only the first 5-6 time steps as input, then predicts forward step-by-step
- **Sensor Masking**: Randomly masks 3 sensors to test reconstruction capability
- **Separate Evaluation**: Reports metrics separately for masked vs. unmasked sensors
- **Multiple Runs**: Repeats experiments with different random maskings for robust results
- **Visualization**: Generates plots showing true vs. predicted temperatures over time

## Usage

### Basic Usage

```bash
python forecast_mode.py \
    --model path/to/model_state.pth \
    --data path/to/test_data.csv \
    --output path/to/output_directory
```

### Advanced Options

```bash
python forecast_mode.py \
    --model path/to/model_state.pth \
    --data path/to/test_data.csv \
    --output path/to/output_directory \
    --initial-steps 6 \          # Number of initial steps to use as input (default: 6)
    --total-steps 20 \           # Total number of steps to predict (default: 20)
    --n-runs 5 \                 # Number of runs with different maskings (default: 5)
    --seed 42                    # Random seed for reproducibility (default: 42)
```

## Input Requirements

### Model File (`model_state.pth`)
The model checkpoint must contain:
- `state_dict`: Model weights
- `y_mean`, `y_std`: Normalization parameters
- `n_static`, `n_time`: Feature dimensions
- `hidden_dim`, `heads`, `dropout`: Model architecture parameters

### Data File (CSV)
The test data CSV must contain the following columns:
- `dt`: Timestamp (datetime format)
- `id_station`: Sensor/station identifier
- `temperature`: Temperature readings
- `latitude`, `longitude`: Sensor locations
- `lat_norm`, `lon_norm`: Normalized coordinates (optional, will be computed if missing)
- Static feature columns (e.g., LST_Terra, LST_Aqua, etc.)

## Output

The script generates:

1. **Plots for Masked Sensors** (`forecast_masked_sensors_run*.png`)
   - Shows true vs. predicted temperatures for the 3 masked sensors
   - Highlights the initial input period and forecast period
   - Displays MAE for the forecast period

2. **Plots for Unmasked Sensors** (`forecast_unmasked_sensors_run*.png`)
   - Shows true vs. predicted temperatures for 3 randomly selected unmasked sensors
   - Same format as masked sensor plots

3. **Summary File** (`forecast_summary.txt`)
   - Average metrics across all runs
   - Separate metrics for masked and unmasked sensors
   - Individual run results

## Metrics

The script reports the following metrics for both masked and unmasked sensors:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and true values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences

Metrics are computed only for the forecast period (after the initial input steps).

## Example

Assuming you have:
- A trained model at: `/path/to/Output_LOOCV_Publication_temperature/S-THC_21317948_21326445/model_state.pth`
- Test data at: `/path/to/filtered_2023_aug_hourly.csv`

Run:

```bash
python forecast_mode.py \
    --model /path/to/Output_LOOCV_Publication_temperature/S-THC_21317948_21326445/model_state.pth \
    --data /path/to/filtered_2023_aug_hourly.csv \
    --output ./forecast_results \
    --initial-steps 6 \
    --total-steps 24 \
    --n-runs 10
```

This will:
1. Use the first 6 time steps as input
2. Predict the next 18 time steps (24 total)
3. Run 10 experiments with different random sensor maskings
4. Save results to `./forecast_results/`

## Interpretation of Results

### Good Performance Indicators:
- **Unmasked sensors**: MAE < 1.0°C indicates the model can forecast well on visible sensors
- **Masked sensors**: MAE < 2.0°C indicates good reconstruction capability for hidden sensors
- Small standard deviation across runs indicates robust performance

### What the Plots Show:
- **Initial period** (gray shaded): Model sees real data for unmasked sensors
- **Forecast period** (after red line): Model uses only its own predictions
- **Masked sensors**: Always show zero input (masked), but model should predict reasonable values
- **Unmasked sensors**: Show realistic predictions that track the true temperature trends

## Technical Details

### Autoregressive Rollout Process:
1. Initialize with first `n` time steps (e.g., 6 hours)
2. For unmasked sensors: use real temperature values (normalized)
3. For masked sensors: always use 0.0 (masked)
4. Predict temperatures for all sensors at next time step
5. Use predicted values (for all sensors) as input for next prediction
6. Repeat until reaching desired forecast length

### Graph Construction:
The script builds a simple k-nearest neighbor graph (k=6) based on sensor indices. For production use, consider:
- Loading the actual graph structure from training
- Using spatial distances for edge construction
- Including mesh nodes if used during training

## Troubleshooting

**Issue**: "No consecutive sequence of length >= N found"
- **Solution**: Your test data doesn't have enough consecutive hourly measurements. Reduce `--total-steps`.

**Issue**: Model predictions are constant or unrealistic
- **Solution**: Check that the data normalization parameters (y_mean, y_std) in the model match your data distribution.

**Issue**: High errors on masked sensors
- **Solution**: This is expected if the model wasn't trained with sufficient masking. The model needs to learn to infer masked sensors from spatial patterns.

## Citation

If you use this forecast mode testing in your research, please cite:
[Your paper citation here]
