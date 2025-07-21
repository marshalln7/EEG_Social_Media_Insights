import os
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Purpose: Model Prediction and Visualization
# What it does:
    # Loads all trained models from the models folder
    # Loads REAL EEG data from featuresets (preserving timestep for visualization)
    # Makes predictions using trained models (excluding timestep from prediction input)
    # Creates timestep-aware visualizations using REAL temporal information
    # Provides model comparison and summary statistics

# Load all trained models
def load_all_models():
    # Find all .pkl model files in the models directory
    all_model_files = glob.glob('models/*.pkl')
    
    if not all_model_files:
        raise FileNotFoundError("No .pkl model files found in the models directory")
    
    # Sort models by filename (which may include timestamp)
    all_model_files = sorted(all_model_files)
    
    # Separate regression and classification models based on filename patterns
    reg_models = []
    clf_models = []
    other_models = []
    
    for model_path in all_model_files:
        filename = os.path.basename(model_path).lower()
        if 'reg' in filename or 'regression' in filename:
            reg_models.append(model_path)
        elif 'clf' in filename or 'classification' in filename or 'classifier' in filename:
            clf_models.append(model_path)
        else:
            # If we can't determine the type from filename, treat as general model
            other_models.append(model_path)
    
    print(f"Found {len(all_model_files)} total model files:")
    print(f"  - {len(reg_models)} regression models")
    print(f"  - {len(clf_models)} classification models")
    print(f"  - {len(other_models)} other/unknown type models")
    
    print("\nRegression models:")
    for model in reg_models:
        print(f"  - {model}")
    
    print("\nClassification models:")
    for model in clf_models:
        print(f"  - {model}")
    
    if other_models:
        print("\nOther models (will be treated as general models):")
        for model in other_models:
            print(f"  - {model}")
    
    # Load all regression models
    reg_loaded_models = []
    for model_path in reg_models:
        try:
            print(f"Loading regression model: {model_path}")
            reg_loaded_models.append(joblib.load(model_path))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    # Load all classification models
    clf_loaded_models = []
    for model_path in clf_models:
        try:
            print(f"Loading classification model: {model_path}")
            clf_loaded_models.append(joblib.load(model_path))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    # Load other models
    other_loaded_models = []
    for model_path in other_models:
        try:
            print(f"Loading general model: {model_path}")
            other_loaded_models.append(joblib.load(model_path))
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    
    return reg_loaded_models, clf_loaded_models, other_loaded_models, reg_models, clf_models, other_models

# Load real EEG data for prediction (preserving timestep for visualization)
def load_real_data():
    # Load real data for prediction
    try:
        real_data = pd.read_csv("featuresets/local datasets_2025-07-15_17-28.csv")
        print(f"Loaded real EEG dataset: {real_data.shape}")
        
        # Extract timestep for visualization (this is the REAL timestep from EEG data)
        if "Timestep" in real_data.columns:
            timestep = real_data["Timestep"].values
            print(f"Found real timestep data: {len(timestep)} points from {timestep.min():.1f} to {timestep.max():.1f}")
        else:
            print("No timestep column found, using sample indices")
            timestep = np.arange(len(real_data))
        
        # Remove Label and Timestep columns for prediction (models weren't trained on these)
        feature_columns = real_data.drop(columns=["Label", "Timestep"], errors='ignore')
        
        print(f"Prepared {feature_columns.shape[1]} features for prediction")
        print(f"Feature names: {list(feature_columns.columns)[:3]}... (showing first 3)")
        
        return feature_columns, timestep
        
    except FileNotFoundError:
        print("Could not find the original featureset file")
        print("Available options:")
        featureset_files = glob.glob("featuresets/*.csv")
        if featureset_files:
            print("   Found these featureset files:")
            for i, file in enumerate(featureset_files):
                print(f"   {i+1}. {file}")
            
            # Use the first available file
            selected_file = featureset_files[0]
            print(f"\nUsing: {selected_file}")
            
            real_data = pd.read_csv(selected_file)
            timestep = real_data.get("Timestep", np.arange(len(real_data))).values
            feature_columns = real_data.drop(columns=["Label", "Timestep"], errors='ignore')
            
            return feature_columns, timestep
        else:
            raise FileNotFoundError("No featureset files found. Cannot proceed without real EEG data.")

# Make predictions and visualize
def predict_and_visualize():
    # Load all trained models
    reg_models, clf_models, other_models, reg_model_names, clf_model_names, other_model_names = load_all_models()
    
    # Load real data
    real_data, timestep = load_real_data()
    
    # Make predictions with all models
    all_reg_predictions = []
    all_clf_predictions = []
    all_other_predictions = []
    
    # Regression models
    if reg_models:
        print("\nMaking predictions with regression models:")
        for i, (model, model_name) in enumerate(zip(reg_models, reg_model_names)):
            try:
                predictions = model.predict(real_data)
                all_reg_predictions.append(predictions)
                print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
            except Exception as e:
                print(f"  Error predicting with {model_name}: {e}")
    
    # Classification models
    if clf_models:
        print("\nMaking predictions with classification models:")
        for i, (model, model_name) in enumerate(zip(clf_models, clf_model_names)):
            try:
                predictions = model.predict(real_data)
                all_clf_predictions.append(predictions)
                print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
            except Exception as e:
                print(f"  Error predicting with {model_name}: {e}")
    
    # Other/general models
    if other_models:
        print("\nMaking predictions with other/general models:")
        for i, (model, model_name) in enumerate(zip(other_models, other_model_names)):
            try:
                predictions = model.predict(real_data)
                all_other_predictions.append(predictions)
                print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
            except Exception as e:
                print(f"  Error predicting with {model_name}: {e}")
    
    # Create simple, effective visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    create_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                reg_model_names, clf_model_names, other_model_names, timestep)
    
    # Ask if user wants to save
    save_option = input("\n💾 Would you like to save these visualizations? (y/n): ").lower().strip()
    if save_option in ['y', 'yes']:
        save_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                  reg_model_names, clf_model_names, other_model_names, timestep)
    
    return all_clf_predictions, all_reg_predictions, all_other_predictions

def create_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions, 
                               reg_model_names, clf_model_names, other_model_names, timestep=None):
    """Create simple, effective visualizations for model predictions"""
    
    # Set a clean style
    plt.style.use('default')
    
    # 1. Overview plot - all models in one view
    create_overview_plot(all_reg_predictions, all_clf_predictions, all_other_predictions,
                        reg_model_names, clf_model_names, other_model_names, timestep)
    
    # 2. Summary statistics plot
    create_summary_statistics_plot(all_reg_predictions, all_clf_predictions, 
                                 reg_model_names, clf_model_names, timestep)
    
    # 3. Individual model plots (only if there are multiple models)
    if len(all_reg_predictions) > 1:
        create_comparison_plot(all_reg_predictions, reg_model_names, "Regression Models", timestep)
    
    if len(all_clf_predictions) > 1:
        create_comparison_plot(all_clf_predictions, clf_model_names, "Classification Models", timestep)

def create_overview_plot(all_reg_predictions, all_clf_predictions, all_other_predictions,
                        reg_model_names, clf_model_names, other_model_names, timestep=None):
    """Create a single overview plot with all essential information - MAIN FUNCTION"""
    
    total_models = len(all_reg_predictions) + len(all_clf_predictions) + len(all_other_predictions)
    if total_models == 0:
        print("No models to visualize")
        return
    
    # Create subplots based on what we have
    fig_height = 6
    if all_reg_predictions and all_clf_predictions:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    elif all_reg_predictions or all_clf_predictions or all_other_predictions:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
    else:
        return
    
    # Plot regression models
    if all_reg_predictions:
        # Use timestep if available, otherwise fall back to sample indices
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_reg_predictions[0]))
            x_label = 'Sample Index'
        
        # Downsample for cleaner visualization if we have too many points
        if len(x_axis) > 500:
            step = len(x_axis) // 300  # Show approximately 300 points
            x_axis_sampled = x_axis[::step]
        else:
            step = 1
            x_axis_sampled = x_axis
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_reg_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_reg_predictions, reg_model_names, colors)):
            predictions_sampled = predictions[::step]
            label = f"Reg Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=color, 
                    linewidth=1.5, alpha=0.8, marker='o', markersize=1)
        
        ax1.set_title('Regression Model Predictions (Downsampled for Clarity)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Predicted Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        if len(all_reg_predictions) > 1:
            mean_pred = np.mean([np.mean(pred) for pred in all_reg_predictions])
            ax1.text(0.02, 0.98, f'Avg Mean: {mean_pred:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
    
    # Plot classification models
    if all_clf_predictions and ax2 is not None:
        # Use timestep if available, otherwise fall back to sample indices
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_clf_predictions[0]))
            x_label = 'Sample Index'
        
        # Downsample for cleaner visualization if we have too many points
        if len(x_axis) > 500:
            step = len(x_axis) // 300  # Show approximately 300 points
            x_axis_sampled = x_axis[::step]
        else:
            step = 1
            x_axis_sampled = x_axis
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            predictions_sampled = predictions[::step]
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax2.plot(x_axis_sampled, predictions_sampled, label=label, color=color, 
                    linewidth=1.5, marker='o', markersize=2, alpha=0.8)
        
        ax2.set_title('Classification Model Predictions (Downsampled for Clarity)', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Predicted Class')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add class distribution info
        if len(all_clf_predictions) > 0:
            unique_classes = len(np.unique(all_clf_predictions[0]))
            ax2.text(0.02, 0.98, f'Classes: {unique_classes}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    verticalalignment='top')
    elif all_clf_predictions and ax2 is None:
        # If we only have classification models, use ax1
        # Use timestep if available, otherwise fall back to sample indices
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_clf_predictions[0]))
            x_label = 'Sample Index'
        
        # Downsample for cleaner visualization if we have too many points
        if len(x_axis) > 500:
            step = len(x_axis) // 300  # Show approximately 300 points
            x_axis_sampled = x_axis[::step]
        else:
            step = 1
            x_axis_sampled = x_axis
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            predictions_sampled = predictions[::step]
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=color, 
                    linewidth=1.5, marker='o', markersize=2, alpha=0.8)
        
        ax1.set_title('Classification Model Predictions (Downsampled for Clarity)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Predicted Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
    plt.suptitle(f'EEG Model Predictions Overview ({total_models} models)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_summary_statistics_plot(all_reg_predictions, all_clf_predictions, 
                                 reg_model_names, clf_model_names, timestep=None):
    """Create summary statistics plots for better insight into model behavior"""
    
    if not all_reg_predictions and not all_clf_predictions:
        print("No predictions to summarize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Prediction ranges over time (binned)
    if all_reg_predictions and timestep is not None:
        ax = axes[0, 0]
        
        # Create time bins for aggregation
        time_bins = np.linspace(timestep.min(), timestep.max(), 20)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            # Bin the predictions by time
            binned_means = []
            binned_stds = []
            
            for j in range(len(time_bins) - 1):
                mask = (timestep >= time_bins[j]) & (timestep < time_bins[j + 1])
                if np.any(mask):
                    binned_means.append(np.mean(predictions[mask]))
                    binned_stds.append(np.std(predictions[mask]))
                else:
                    binned_means.append(np.nan)
                    binned_stds.append(np.nan)
            
            binned_means = np.array(binned_means)
            binned_stds = np.array(binned_stds)
            
            label = f"Model {i+1}"
            ax.plot(bin_centers, binned_means, label=label, linewidth=2, marker='o')
            ax.fill_between(bin_centers, binned_means - binned_stds, binned_means + binned_stds, 
                           alpha=0.3)
        
        ax.set_title('Regression Predictions: Mean ± Std Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Prediction Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No regression data\nor timestep available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Regression Statistics (N/A)')
    
    # 2. Classification distribution over time
    if all_clf_predictions and timestep is not None:
        ax = axes[0, 1]
        
        # Create time bins
        time_bins = np.linspace(timestep.min(), timestep.max(), 20)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            # Calculate class distribution in each time bin
            unique_classes = np.unique(predictions)
            class_proportions = []
            
            for j in range(len(time_bins) - 1):
                mask = (timestep >= time_bins[j]) & (timestep < time_bins[j + 1])
                if np.any(mask):
                    bin_predictions = predictions[mask]
                    # Calculate proportion of most common class in this bin
                    most_common_class_prop = np.max(np.bincount(bin_predictions.astype(int))) / len(bin_predictions)
                    class_proportions.append(most_common_class_prop)
                else:
                    class_proportions.append(np.nan)
            
            label = f"Model {i+1}"
            ax.plot(bin_centers, class_proportions, label=label, linewidth=2, marker='s')
        
        ax.set_title('Classification: Dominant Class Proportion Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Proportion of Dominant Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No classification data\nor timestep available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Classification Statistics (N/A)')
    
    # 3. Overall prediction distribution histograms
    if all_reg_predictions:
        ax = axes[1, 0]
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            ax.hist(predictions, bins=30, alpha=0.6, label=f'Model {i+1}', density=True)
        ax.set_title('Regression Prediction Distributions', fontweight='bold')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No regression data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Regression Distribution (N/A)')
    
    # 4. Classification accuracy/consistency over time
    if all_clf_predictions and len(all_clf_predictions) > 1:
        ax = axes[1, 1]
        
        # Calculate agreement between models over time
        time_bins = np.linspace(timestep.min(), timestep.max(), 20)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        agreement_rates = []
        for j in range(len(time_bins) - 1):
            mask = (timestep >= time_bins[j]) & (timestep < time_bins[j + 1])
            if np.any(mask):
                # Get predictions from all models for this time bin
                bin_predictions = [pred[mask] for pred in all_clf_predictions]
                if len(bin_predictions[0]) > 0:
                    # Calculate how often all models agree
                    agreements = 0
                    total = len(bin_predictions[0])
                    for k in range(total):
                        model_preds = [pred[k] for pred in bin_predictions]
                        if len(set(model_preds)) == 1:  # All models agree
                            agreements += 1
                    agreement_rates.append(agreements / total if total > 0 else 0)
                else:
                    agreement_rates.append(np.nan)
            else:
                agreement_rates.append(np.nan)
        
        ax.plot(bin_centers, agreement_rates, linewidth=2, marker='d', color='red')
        ax.set_title('Model Agreement Rate Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Agreement Rate')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need multiple classification\nmodels for agreement analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Model Agreement (N/A)')
    
    plt.suptitle('EEG Model Prediction Summary Statistics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_comparison_plot(predictions_list, model_names, title, timestep=None):
    """Create a simple comparison plot for multiple models of the same type"""
    
    if len(predictions_list) < 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time series comparison
    # Use timestep if available, otherwise fall back to sample indices
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Timestep'
    else:
        x_axis = np.arange(len(predictions_list[0]))
        x_label = 'Sample Index'
    
    # Downsample for cleaner visualization if we have too many points
    if len(x_axis) > 500:
        step = len(x_axis) // 300  # Show approximately 300 points
        x_axis_sampled = x_axis[::step]
    else:
        step = 1
        x_axis_sampled = x_axis
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_list)))
    
    for i, (predictions, model_name, color) in enumerate(zip(predictions_list, model_names, colors)):
        predictions_sampled = predictions[::step]
        short_name = os.path.basename(model_name).replace('.pkl', '')
        ax1.plot(x_axis_sampled, predictions_sampled, label=f'Model {i+1}', 
                color=color, linewidth=1.5, alpha=0.8, marker='o', markersize=1)
    
    ax1.set_title(f'{title} - Time Series (Downsampled)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot for distribution comparison
    ax2.boxplot(predictions_list, tick_labels=[f'Model {i+1}' for i in range(len(predictions_list))])
    ax2.set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Predicted Value')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def save_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions,
                              reg_model_names, clf_model_names, other_model_names, timestep=None):
    """Save simple visualizations to files"""
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    saved_files = []
    
    print(f"Saving visualizations to {viz_dir}/...")
    
    # Save overview plot
    fig = create_overview_plot_for_saving(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                         reg_model_names, clf_model_names, other_model_names, timestep)
    if fig is not None:
        filename = f"{viz_dir}/model_overview_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(filename)
    
    # Save summary statistics plot
    fig = create_summary_statistics_plot_for_saving(all_reg_predictions, all_clf_predictions,
                                                   reg_model_names, clf_model_names, timestep)
    if fig is not None:
        filename = f"{viz_dir}/summary_statistics_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(filename)
    
    # Save comparison plots
    if len(all_reg_predictions) > 1:
        fig = create_comparison_plot_for_saving(all_reg_predictions, reg_model_names, "Regression Models", timestep)
        if fig is not None:
            filename = f"{viz_dir}/regression_comparison_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
    
    if len(all_clf_predictions) > 1:
        fig = create_comparison_plot_for_saving(all_clf_predictions, clf_model_names, "Classification Models", timestep)
        if fig is not None:
            filename = f"{viz_dir}/classification_comparison_{timestamp}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
    
    print(f"\n Successfully saved {len(saved_files)} visualization(s):")
    for file in saved_files:
        print(f"   - {file}")
    print()

# Legacy function for backwards compatibility (removed complex visualizations)
def create_comprehensive_visualizations(*args, **kwargs):
    """Redirects to simple visualizations"""
    return create_simple_visualizations(*args, **kwargs)

def create_overview_plot_for_saving(all_reg_predictions, all_clf_predictions, all_other_predictions,
                                   reg_model_names, clf_model_names, other_model_names, timestep=None):
    """Create overview plot for saving (returns figure instead of showing)"""
    
    total_models = len(all_reg_predictions) + len(all_clf_predictions) + len(all_other_predictions)
    if total_models == 0:
        return None
    
    # Create subplots based on what we have
    if all_reg_predictions and all_clf_predictions:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    elif all_reg_predictions or all_clf_predictions or all_other_predictions:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
    else:
        return None
    
    # Plot regression models
    if all_reg_predictions:
        # Use timestep if available, otherwise fall back to sample indices
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_reg_predictions[0]))
            x_label = 'Sample Index'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_reg_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_reg_predictions, reg_model_names, colors)):
            label = f"Reg Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(x_axis, predictions, label=label, color=color, linewidth=2, alpha=0.8)
        
        ax1.set_title('Regression Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Predicted Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        if len(all_reg_predictions) > 1:
            mean_pred = np.mean([np.mean(pred) for pred in all_reg_predictions])
            ax1.text(0.02, 0.98, f'Avg Mean: {mean_pred:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
    
    # Plot classification models
    if all_clf_predictions and ax2 is not None:
        # Use timestep if available, otherwise fall back to sample indices
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_clf_predictions[0]))
            x_label = 'Sample Index'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax2.plot(x_axis, predictions, label=label, color=color, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax2.set_title('Classification Model Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('Predicted Class')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add class distribution info
        if len(all_clf_predictions) > 0:
            unique_classes = len(np.unique(all_clf_predictions[0]))
            ax2.text(0.02, 0.98, f'Classes: {unique_classes}', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    verticalalignment='top')
    elif all_clf_predictions and ax2 is None:
        # If we only have classification models, use ax1
        # Use timestep if available, otherwise fall back to sample indices
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_clf_predictions[0]))
            x_label = 'Sample Index'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(x_axis, predictions, label=label, color=color, 
                    linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax1.set_title('Classification Model Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Predicted Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
    fig.suptitle(f'EEG Model Predictions Overview ({total_models} models)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

def create_comparison_plot_for_saving(predictions_list, model_names, title, timestep=None):
    """Create comparison plot for saving (returns figure instead of showing)"""
    
    if len(predictions_list) < 2:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time series comparison
    # Use timestep if available, otherwise fall back to sample indices
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Timestep'
    else:
        x_axis = np.arange(len(predictions_list[0]))
        x_label = 'Sample Index'
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_list)))
    
    for i, (predictions, model_name, color) in enumerate(zip(predictions_list, model_names, colors)):
        short_name = os.path.basename(model_name).replace('.pkl', '')
        ax1.plot(x_axis, predictions, label=f'Model {i+1}', 
                color=color, linewidth=2, alpha=0.8)
    
    ax1.set_title(f'{title} - Time Series', fontsize=14, fontweight='bold')
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Predicted Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot for distribution comparison
    ax2.boxplot(predictions_list, tick_labels=[f'Model {i+1}' for i in range(len(predictions_list))])
    ax2.set_title(f'{title} - Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Predicted Value')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'{title} Comparison', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

def create_summary_statistics_plot_for_saving(all_reg_predictions, all_clf_predictions, 
                                            reg_model_names, clf_model_names, timestep=None):
    """Create summary statistics plot for saving (returns figure instead of showing)"""
    
    if not all_reg_predictions and not all_clf_predictions:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Prediction ranges over time (binned)
    if all_reg_predictions and timestep is not None:
        ax = axes[0, 0]
        
        # Create time bins for aggregation
        time_bins = np.linspace(timestep.min(), timestep.max(), 20)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            # Bin the predictions by time
            binned_means = []
            binned_stds = []
            
            for j in range(len(time_bins) - 1):
                mask = (timestep >= time_bins[j]) & (timestep < time_bins[j + 1])
                if np.any(mask):
                    binned_means.append(np.mean(predictions[mask]))
                    binned_stds.append(np.std(predictions[mask]))
                else:
                    binned_means.append(np.nan)
                    binned_stds.append(np.nan)
            
            binned_means = np.array(binned_means)
            binned_stds = np.array(binned_stds)
            
            label = f"Model {i+1}"
            ax.plot(bin_centers, binned_means, label=label, linewidth=2, marker='o')
            ax.fill_between(bin_centers, binned_means - binned_stds, binned_means + binned_stds, 
                           alpha=0.3)
        
        ax.set_title('Regression Predictions: Mean ± Std Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Prediction Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No regression data\nor timestep available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
        axes[0, 0].set_title('Regression Statistics (N/A)')
    
    # 2. Classification distribution over time
    if all_clf_predictions and timestep is not None:
        ax = axes[0, 1]
        
        # Create time bins
        time_bins = np.linspace(timestep.min(), timestep.max(), 20)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            # Calculate class distribution in each time bin
            unique_classes = np.unique(predictions)
            class_proportions = []
            
            for j in range(len(time_bins) - 1):
                mask = (timestep >= time_bins[j]) & (timestep < time_bins[j + 1])
                if np.any(mask):
                    bin_predictions = predictions[mask]
                    # Calculate proportion of most common class in this bin
                    most_common_class_prop = np.max(np.bincount(bin_predictions.astype(int))) / len(bin_predictions)
                    class_proportions.append(most_common_class_prop)
                else:
                    class_proportions.append(np.nan)
            
            label = f"Model {i+1}"
            ax.plot(bin_centers, class_proportions, label=label, linewidth=2, marker='s')
        
        ax.set_title('Classification: Dominant Class Proportion Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Proportion of Dominant Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No classification data\nor timestep available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
        axes[0, 1].set_title('Classification Statistics (N/A)')
    
    # 3. Overall prediction distribution histograms
    if all_reg_predictions:
        ax = axes[1, 0]
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            ax.hist(predictions, bins=30, alpha=0.6, label=f'Model {i+1}', density=True)
        ax.set_title('Regression Prediction Distributions', fontweight='bold')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No regression data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Regression Distribution (N/A)')
    
    # 4. Classification accuracy/consistency over time
    if all_clf_predictions and len(all_clf_predictions) > 1:
        ax = axes[1, 1]
        
        # Calculate agreement between models over time
        time_bins = np.linspace(timestep.min(), timestep.max(), 20)
        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        
        agreement_rates = []
        for j in range(len(time_bins) - 1):
            mask = (timestep >= time_bins[j]) & (timestep < time_bins[j + 1])
            if np.any(mask):
                # Get predictions from all models for this time bin
                bin_predictions = [pred[mask] for pred in all_clf_predictions]
                if len(bin_predictions[0]) > 0:
                    # Calculate how often all models agree
                    agreements = 0
                    total = len(bin_predictions[0])
                    for k in range(total):
                        model_preds = [pred[k] for pred in bin_predictions]
                        if len(set(model_preds)) == 1:  # All models agree
                            agreements += 1
                    agreement_rates.append(agreements / total if total > 0 else 0)
                else:
                    agreement_rates.append(np.nan)
            else:
                agreement_rates.append(np.nan)
        
        ax.plot(bin_centers, agreement_rates, linewidth=2, marker='d', color='red')
        ax.set_title('Model Agreement Rate Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Agreement Rate')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'Need multiple classification\nmodels for agreement analysis', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Model Agreement (N/A)')
    
    fig.suptitle('EEG Model Prediction Summary Statistics', fontsize=16, fontweight='bold')
    fig.tight_layout()
    return fig

def check_saved_visualizations():
    """Check the size and status of recently saved visualizations"""
    import glob
    
    viz_files = glob.glob("visualizations/*.png")
    if not viz_files:
        print("No visualization files found in visualizations/ directory")
        return
    
    # Sort by modification time, most recent first
    viz_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\nRecent visualization files:")
    print("=" * 50)
    
    for i, file in enumerate(viz_files[:5]):  # Show last 5 files
        size = os.path.getsize(file)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
        
        status = "Valid" if size > 1000 else "Blank/Invalid"
        size_mb = size / (1024 * 1024)  # Convert to MB
        
        filename = os.path.basename(file)
        print(f"{i+1}. {filename}")
        print(f"   Size: {size_mb:.2f} MB ({size:,} bytes)")
        print(f"   Modified: {mod_time}")
        print(f"   Status: {status}")
        print()

if __name__ == "__main__":
    clf_predictions, reg_predictions, other_predictions = predict_and_visualize()
