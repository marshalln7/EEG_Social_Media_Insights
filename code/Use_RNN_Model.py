import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import glob
import torch

# Import model framework, interpretation functions
from RNN_Creator import TestRNN, normalice, classify

# Purpose: RNN Model Prediction and Visualization
# What it does:
#   - Loads all NeuroSense RNN models from the models folder
#   - Loads REAL EEG data from featuresets (preserving timestep for visualization)
#   - Makes predictions using the RNN model (PyTorch)
#   - Creates timestep-aware visualizations using REAL temporal information

# Find and create a list of the names of all RNN models that follow naming scheme (Only NeuroSense for now)
def load_rnn_models():
    # Find all NeuroSense RNN model files in the models directory
    all_model_files = glob.glob('../models/*_RNN_model_*.pth')
    if not all_model_files:
        raise FileNotFoundError("No *_RNN_model_*.pth files found in the models directory")
    all_model_files = sorted(all_model_files)
    print(f"Found {len(all_model_files)} RNN model file(s):")
    for model in all_model_files:
        print(f"  - {model}")
    return all_model_files

# Load and return designated dataset within code, returns all explanatory variables (NO LABEL) and the `Timestep` column
# If file does not exist, uses the first file within the specified file
## filename = "featuresets/local datasets_2025-07-15_17-28.csv"
def load_real_data(filename):
    try:
        real_data = pd.read_csv(filename)
        real_data.sort_values(by = 'Timestep', inplace = True)
        print(f"Loaded real EEG dataset: {real_data.shape}")
        if "Timestep" in real_data.columns:
            timestep = real_data["Timestep"].values
            print(f"Found real timestep data: {len(timestep)} points from {timestep.min():.1f} to {timestep.max():.1f}")
        else:
            print("No timestep column found, using sample indices")
            timestep = np.arange(len(real_data))
        feature_columns = real_data.drop(columns=["Label", "Timestep"], axis = 1)
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
            feature_columns = real_data.drop(columns=["Label", "Timestep"], axis = 1)
            return feature_columns, timestep
        else:
            raise FileNotFoundError("No featureset files found. Cannot proceed without real EEG data.")
        
# Returns the predicted values of the model, both probabilistic (pseudo-logistic) regression and classification
# Returns these values in an interpretable format
def RNN_preds(filename, model):
    features, timestep = load_real_data(filename)

    # Convert `features` into torch tensor to feed into neural network
    input = torch.from_numpy(np.array(features, dtype = np.float32))

    # Load in model according to TestRNN structure (imported from RNN_Creator.py) and feed in data
    loaded_model = torch.load(model, weights_only = False)
    model_output = loaded_model(input.view(features.shape[0], -1, features.shape[1]))

    # Return regression and classification output
    reg_output = normalice(model_output, proportion = True)
    clf_output = classify(reg_output)

    # Format orientation of the regression output (list of lists, n rows with k columns for k classifications -> list of lists, k rows with n columns)
    reg_output = np.array(reg_output).T.tolist()

    return reg_output, clf_output, timestep
    

def RNN_viz(reg_output, clf_output, timestep, class_names):
     
    sns.set_palette("husl")

    if len(reg_output) != len(class_names):
        return print('Regression length must equal the number of classes in `class_names`')
    
    color_list = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    for emotion, color, label in zip(reg_output, color_list[:len(reg_output)], class_names):
        plt.plot(timestep, emotion, color = color, label = label, linewidth=3, alpha=0.9)

    plt.title('EEG Concentration Models - Statistical Distribution Analysis', fontweight='bold', fontsize=12)
    plt.xlabel("Time (seconds from EEG recording start)")
    plt.ylabel('Probability of Given Emotion')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def main():
    model_files = load_rnn_models()
    real_data, timestep = load_real_data()
    print(f"Loaded {len(model_files)} RNN model(s) and real EEG data with shape {real_data.shape}")

if __name__ == "__main__":
    main()
