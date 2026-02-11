# Quick Start Guide - Forecast Mode Testing

## What is Forecast Mode?

Forecast mode tests your trained GNN model's ability to:
1. **Predict future temperatures** without seeing future data (autoregressive forecasting)
2. **Reconstruct missing sensors** that were hidden from the model

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python test_forecast_mode.py
```

This creates synthetic data and runs a quick test. If it completes successfully, you're ready to go!

### Step 3: Run with Your Data

```bash
python forecast_mode.py \
    --model path/to/your/model_state.pth \
    --data path/to/your/data.csv \
    --output results/
```

## What You Need

1. **A trained model file** (`.pth` format)
   - Should be in `Output_LOOCV_*/*/model_state.pth`
   - Created when you trained the Ta-GAT model

2. **Test data** (`.csv` format)
   - Must have columns: `dt`, `id_station`, `temperature`, `latitude`, `longitude`
   - Should have at least 20 consecutive hourly timestamps

## Understanding the Output

After running, you'll get:

### 📊 Plots
- `forecast_masked_sensors_run*.png` - Shows how well the model predicts the 3 hidden sensors
- `forecast_unmasked_sensors_run*.png` - Shows how well the model forecasts the visible sensors

### 📄 Summary
- `forecast_summary.txt` - Contains average MAE and RMSE metrics

### What the Numbers Mean

**Unmasked Sensors** (sensors the model can "see"):
- **MAE < 1.0°C**: Excellent forecast accuracy ✅
- **MAE 1.0-2.0°C**: Good forecast accuracy ⚠️
- **MAE > 2.0°C**: Poor forecast accuracy ❌

**Masked Sensors** (sensors hidden from the model):
- **MAE < 2.0°C**: Excellent reconstruction ✅
- **MAE 2.0-3.0°C**: Good reconstruction ⚠️
- **MAE > 3.0°C**: Poor reconstruction ❌

## Common Issues & Solutions

### ❌ "No consecutive sequence found"
**Problem**: Your data doesn't have enough consecutive hourly measurements  
**Solution**: Use `--total-steps 15` (or lower) to require fewer consecutive points

### ❌ "Module not found"
**Problem**: Dependencies not installed  
**Solution**: Run `pip install -r requirements.txt`

### ❌ "Model file not found"
**Problem**: Wrong path to model file  
**Solution**: Check that the path is correct and file exists

### ❌ High errors on all sensors
**Problem**: Model/data mismatch or normalization issues  
**Solution**: Verify the model was trained on similar data

## Advanced Usage

### Custom Parameters

```bash
python forecast_mode.py \
    --model model.pth \
    --data data.csv \
    --output results/ \
    --initial-steps 6 \      # Use first 6 hours as input
    --total-steps 24 \       # Predict 24 hours total
    --n-runs 10 \            # Run 10 different experiments
    --seed 42                # For reproducibility
```

### Using the Example Script

1. Edit `example_forecast.py` to set your paths
2. Run: `python example_forecast.py`

## What Makes This Different?

**Normal Model Testing** (what you've been doing):
- Uses true temperature values at each timestep
- Tests if model can interpolate/fill in missing values
- One-step-ahead prediction with perfect context

**Forecast Mode** (what this does):
- Uses true values ONLY for first few steps
- Then feeds predictions back as input (autoregressive)
- Tests if model can predict future without seeing it
- Tests if model can reconstruct fully hidden sensors

## Visual Guide

```
Time:        1  2  3  4  5  6  7  8  9  10 11 12 13 ...
            [---Initial Input---] [----Forecast Period----]
                                  ^
                                  Model on its own from here!

Unmasked:   ✓  ✓  ✓  ✓  ✓  ✓  ?  ?  ?  ?  ?  ?  ?  ...
Masked:     X  X  X  X  X  X  ?  ?  ?  ?  ?  ?  ?  ...

Legend:
  ✓ = Model sees true value
  X = Masked (always zero input)
  ? = Model must predict
```

## Next Steps

1. ✅ Run test with synthetic data: `python test_forecast_mode.py`
2. ✅ Run with your trained model: `python forecast_mode.py --model ... --data ...`
3. ✅ Check the plots in the output directory
4. ✅ Review metrics in `forecast_summary.txt`
5. ✅ Experiment with different parameters

## Need More Help?

- **Full Documentation**: See `FORECAST_MODE_README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Example Code**: See `example_forecast.py`

## Quick Checklist

Before running with real data, make sure you have:

- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Verified installation (`python test_forecast_mode.py`)
- [ ] Located your trained model file (`.pth`)
- [ ] Prepared your test data (`.csv` with required columns)
- [ ] Created an output directory for results

Ready? Run:
```bash
python forecast_mode.py --model YOUR_MODEL.pth --data YOUR_DATA.csv --output results/
```

Good luck! 🚀
