#!/usr/bin/env python3
"""
Example script demonstrating how to use the forecast mode testing.

This script shows:
1. How to prepare your data
2. How to run forecast mode with a trained model
3. How to interpret the results

Note: Update the paths below to match your setup.
"""

import os
import sys
from pathlib import Path

# Example paths - UPDATE THESE FOR YOUR SETUP
MODEL_PATH = "Output_LOOCV_Irregular_MeshTa/S-THC 21317948_21326445/model_state.pth"
DATA_PATH = "filtered_2023_aug_hourly.csv"
OUTPUT_DIR = "forecast_results"

# Configuration
NUM_INITIAL_STEPS = 6  # Use first 6 time steps as input
TOTAL_STEPS = 20       # Predict 20 time steps total (14 forecast steps)
N_RUNS = 5             # Run 5 times with different random maskings
SEED = 42              # For reproducibility


def check_files_exist():
    """Check if required files exist"""
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("\nTo get a trained model:")
        print("1. Run the Ta-GAT.ipynb notebook to train a model")
        print("2. The model will be saved in Output_LOOCV_*/ directories")
        print("3. Update MODEL_PATH in this script")
        return False
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("\nTo get test data:")
        print("1. Ensure you have the sensor CSV file")
        print("2. Update DATA_PATH in this script")
        return False
    
    return True


def run_forecast_demo():
    """Run the forecast mode demo"""
    print("=" * 80)
    print("FORECAST MODE DEMO")
    print("=" * 80)
    print()
    print("This demo will:")
    print(f"  1. Load model from: {MODEL_PATH}")
    print(f"  2. Load test data from: {DATA_PATH}")
    print(f"  3. Run {N_RUNS} forecast experiments")
    print(f"  4. Use first {NUM_INITIAL_STEPS} steps as input")
    print(f"  5. Predict {TOTAL_STEPS - NUM_INITIAL_STEPS} future steps")
    print(f"  6. Save results to: {OUTPUT_DIR}")
    print()
    
    # Import the forecast mode module
    try:
        from forecast_mode import run_forecast_experiment
    except ImportError as e:
        print(f"ERROR: Cannot import forecast_mode module: {e}")
        print("\nMake sure forecast_mode.py is in the same directory")
        return
    
    # Check files
    if not check_files_exist():
        return
    
    # Run the experiment
    print("Starting forecast experiments...")
    print("-" * 80)
    
    try:
        run_forecast_experiment(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            num_initial_steps=NUM_INITIAL_STEPS,
            total_steps=TOTAL_STEPS,
            n_runs=N_RUNS,
            seed=SEED
        )
        
        print()
        print("=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Check the output directory for:")
        print(f"  - Plots: {OUTPUT_DIR}/forecast_*.png")
        print(f"  - Summary: {OUTPUT_DIR}/forecast_summary.txt")
        print()
        
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR DURING FORECAST EXPERIMENT")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()


def print_usage():
    """Print usage instructions"""
    print()
    print("USAGE INSTRUCTIONS:")
    print("=" * 80)
    print()
    print("Option 1: Run this demo script (after updating paths)")
    print("  $ python example_forecast.py")
    print()
    print("Option 2: Use forecast_mode.py directly")
    print("  $ python forecast_mode.py \\")
    print("      --model path/to/model.pth \\")
    print("      --data path/to/data.csv \\")
    print("      --output results/")
    print()
    print("Option 3: With custom parameters")
    print("  $ python forecast_mode.py \\")
    print("      --model path/to/model.pth \\")
    print("      --data path/to/data.csv \\")
    print("      --output results/ \\")
    print("      --initial-steps 6 \\")
    print("      --total-steps 24 \\")
    print("      --n-runs 10 \\")
    print("      --seed 42")
    print()
    print("For more details, see FORECAST_MODE_README.md")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        # Check if paths are still default (need to be updated)
        if MODEL_PATH.startswith("Output_") and not os.path.exists(MODEL_PATH):
            print()
            print("⚠️  ATTENTION: Please update the paths in this script first!")
            print()
            print("Edit example_forecast.py and set:")
            print("  - MODEL_PATH: Path to your trained model (.pth file)")
            print("  - DATA_PATH: Path to your test data (.csv file)")
            print()
            print("Then run: python example_forecast.py")
            print()
            print_usage()
        else:
            run_forecast_demo()
