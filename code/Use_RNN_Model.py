import os
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Purpose: RNN Model Prediction and Visualization
# What it does:
#   - Loads all NeuroSense RNN models from the models folder
#   - Loads REAL EEG data from featuresets (preserving timestep for visualization)
#   - Makes predictions using the RNN model (PyTorch)
#   - Creates timestep-aware visualizations using REAL temporal information

def load_rnn_models():
    # Find all NeuroSense RNN model files in the models directory
    all_model_files = glob.glob('models/NeuroSense_RNN_model_*.pth')
    if not all_model_files:
        raise FileNotFoundError("No NeuroSense_RNN_model_*.pth files found in the models directory")
    all_model_files = sorted(all_model_files)
    print(f"Found {len(all_model_files)} NeuroSense RNN model file(s):")
    for model in all_model_files:
        print(f"  - {model}")
    return all_model_files

def load_real_data():
    try:
        real_data = pd.read_csv("featuresets/local datasets_2025-07-15_17-28.csv")
        print(f"Loaded real EEG dataset: {real_data.shape}")
        if "Timestep" in real_data.columns:
            timestep = real_data["Timestep"].values
            print(f"Found real timestep data: {len(timestep)} points from {timestep.min():.1f} to {timestep.max():.1f}")
        else:
            print("No timestep column found, using sample indices")
            timestep = np.arange(len(real_data))
        feature_columns = real_data.drop(columns=["Label", "Timestep"], errors='ignore')
        print(f"Prepared {feature_columns.shape[1]} features for prediction")
        print(f"Feature names: {list(feature_columns.columns)[:3]}... (showing first 3)")
        return feature_columns, timestep
    except FileNotFoundError:
        print("Could not find the original featureset file")
        featureset_files = glob.glob("featuresets/*.csv")
        if featureset_files:
            selected_file = featureset_files[0]
            print(f"\nUsing: {selected_file}")
            real_data = pd.read_csv(selected_file)
            timestep = real_data.get("Timestep", np.arange(len(real_data))).values
            feature_columns = real_data.drop(columns=["Label", "Timestep"], errors='ignore')
            return feature_columns, timestep
        else:
            raise FileNotFoundError("No featureset files found. Cannot proceed without real EEG data.")

def main():
    model_files = load_rnn_models()
    real_data, timestep = load_real_data()
    print(f"Loaded {len(model_files)} RNN model(s) and real EEG data with shape {real_data.shape}")

if __name__ == "__main__":
    main()
