# Ta-GAT: Temporal-spatial Graph Attention Network

A Graph Neural Network (GNN) model for temperature prediction using Graph Attention Networks (GAT) over irregular spatial meshes.

## Overview

This repository implements a dual-stream Graph Attention Network for predicting temperature across spatial sensor networks. The model combines:
- **Geographic features**: Spatial coordinates and relationships
- **Semantic features**: Static environmental features (LST, NDVI, DEM, etc.)
- **Temporal encoding**: Cyclical time features (hour, day-of-year)
- **Graph structure**: Irregular mesh with sensor-to-sensor and sensor-to-mesh edges

## New: Forecast Mode Testing 🎯

Test your trained model's ability to predict future temperatures and reconstruct missing sensors!

### Quick Start with Forecast Mode

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test installation (creates synthetic data)
python test_forecast_mode.py

# 3. Run with your trained model
python forecast_mode.py \
    --model path/to/model_state.pth \
    --data path/to/test_data.csv \
    --output results/
```

**What it does:**
- Randomly masks 3 sensors (treats as missing)
- Uses first 5-6 time steps as input
- Predicts future steps autoregressively (feeding predictions back)
- Evaluates separately for masked vs unmasked sensors
- Generates plots and error metrics (MAE, RMSE)

**Documentation:**
- 📖 [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- 📚 [Forecast Mode README](FORECAST_MODE_README.md) - Complete documentation
- 📝 [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical details
- 💡 [Example Script](example_forecast.py) - Usage example
- 🧪 [Test Script](test_forecast_mode.py) - Validation with synthetic data

## Repository Structure

```
Ta-GAT/
├── Ta-GAT.ipynb              # Main training notebook
├── forecast_mode.py          # Forecast mode testing script
├── example_forecast.py       # Usage example
├── test_forecast_mode.py     # Test with synthetic data
├── requirements.txt          # Python dependencies
├── QUICKSTART.md            # 5-minute setup guide
├── FORECAST_MODE_README.md  # Complete forecast mode docs
├── IMPLEMENTATION_SUMMARY.md # Technical details
└── Output_LOOCV_*/          # Training outputs (models, results)
```

## Features

### Training (Ta-GAT.ipynb)
- **Leave-One-Out Cross-Validation (LOOCV)**: Validates on each station
- **Sensor masking**: Randomly masks 20% of sensors during training
- **Irregular mesh**: Uses Delaunay triangulation for spatial coverage
- **Hybrid graph**: Combines sensor-sensor, sensor-mesh, and mesh-mesh edges
- **Early stopping**: Prevents overfitting
- **Ensemble reconstruction**: Pools predictions from multiple trained models

### Forecast Mode (forecast_mode.py)
- **Autoregressive forecasting**: Predicts future without seeing true values
- **Sensor reconstruction**: Tests ability to infer hidden sensor readings
- **Multiple runs**: Averages over different random maskings
- **Visualization**: Time series plots with clear indicators
- **Separate metrics**: Reports masked vs unmasked performance independently

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, rasterio, scipy

See [requirements.txt](requirements.txt) for complete list.

## Installation

```bash
# Clone the repository
git clone https://github.com/edwinol96/Ta-GAT.git
cd Ta-GAT

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

1. Open `Ta-GAT.ipynb` in Jupyter
2. Update configuration paths (data, static features, output directory)
3. Run all cells
4. Trained models saved to `Output_LOOCV_*/*/model_state.pth`

### Testing in Forecast Mode

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

**Basic usage:**
```bash
python forecast_mode.py \
    --model Output_LOOCV_Irregular_MeshTa/S-THC_21317948_21326445/model_state.pth \
    --data filtered_2023_aug_hourly.csv \
    --output forecast_results/
```

**With custom parameters:**
```bash
python forecast_mode.py \
    --model path/to/model.pth \
    --data path/to/data.csv \
    --output results/ \
    --initial-steps 6 \
    --total-steps 24 \
    --n-runs 10 \
    --seed 42
```

## Model Architecture

**DualStreamGAT** consists of:
1. **Geographic Stream**: Processes spatial coordinates
   - Input: 2D (lat, lon)
   - Layer: GATv2Conv with 3 attention heads
   
2. **Semantic Stream**: Processes features + temperature
   - Input: Static features + temporal encoding + temperature
   - Layer: GATv2Conv with 3 attention heads
   
3. **Fusion Layer**: Combines both streams
   - LayerNorm → ELU activation → Linear (to 1D output)

## Data Format

### Training Data
CSV with columns:
- `dt`: Timestamp
- `id_station`: Station/sensor ID
- `temperature`: Temperature reading
- `latitude`, `longitude`: Geographic coordinates
- Static feature columns (from TIF files)

### Forecast Testing Data
Same format as training, but requires:
- Consecutive hourly timestamps (at least 15-20)
- Multiple sensors (at least 5-10)
- Normalized coordinates (`lat_norm`, `lon_norm`)

## Results Interpretation

### Training Metrics
- **RMSE/MAE**: Lower is better (typically < 1.0°C is good)
- **Laplacian loss**: Measures spatial smoothness
- **Per-station metrics**: Check for spatial patterns

### Forecast Metrics
- **Unmasked sensors** (visible):
  - MAE < 1.0°C: Excellent forecast
  - MAE 1-2°C: Good forecast
  
- **Masked sensors** (hidden):
  - MAE < 2.0°C: Excellent reconstruction
  - MAE 2-3°C: Good reconstruction

## Citation

If you use this code in your research, please cite:

```
[Citation to be added]
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[License information to be added]

## Contact

For questions or issues, please open an issue on GitHub or contact [contact information].

## Acknowledgments

This work builds on:
- PyTorch Geometric for graph neural network implementations
- Graph Attention Networks (GAT) architecture
- Spatial-temporal modeling for urban heat island research

---

**Key Documentation Files:**
- [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- [FORECAST_MODE_README.md](FORECAST_MODE_README.md) - Complete forecast mode documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details and design decisions
