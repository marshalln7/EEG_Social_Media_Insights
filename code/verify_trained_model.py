#!/usr/bin/env python3
"""
EEG Model Verification Script
Run with: python3 verify_trained_model.py
"""
from Use_Trained_Model import EEGModelPredictor

print("Starting EEG Model Prediction...")

# Create predictor
predictor = EEGModelPredictor(
    models_dir="models",
    featuresets_dir="local featuresets",
    model_types=['concentration', 'mendeley']
)

# Load and predict
print(" Loading models...")
model_info = predictor.load_models()
print(f" Loaded {model_info['total_loaded']} models")

print("Loading dataset...")
# Specify the exact dataset file you want to use
data_info = predictor.load_data(data_file="local datasets_2025-07-17_09-50.csv")
print(f" Loaded data: {data_info['data_shape']} from {data_info['file_used']}")

print(" Making predictions...")
predictions = predictor.predict()

print(" Generating visualization graphs...")
predictor.visualize_predictions()
