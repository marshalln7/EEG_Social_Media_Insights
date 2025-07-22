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

class EEGModelPredictor:
    """
    A reusable class for loading EEG models and making predictions
    
    This class provides a clean interface for:
    - Loading trained models from specified directories
    - Loading EEG feature data for prediction
    - Making predictions with multiple models
    - Visualizing and saving results
    - Exporting predictions to files
    """
    
    def __init__(self, models_dir="models", featuresets_dir="featuresets", 
                 model_types=None, verbose=True):
        """
        Initialize the EEG Model Predictor
        
        Args:
            models_dir (str): Directory containing model files
            featuresets_dir (str): Directory containing feature data
            model_types (list): List of model types to include (e.g., ['concentration', 'mendeley'])
            verbose (bool): Whether to print status messages
        """
        self.models_dir = models_dir
        self.featuresets_dir = featuresets_dir
        self.model_types = model_types or ['concentration', 'mendeley']
        self.verbose = verbose
        
        # Internal state
        self.reg_models = []
        self.clf_models = []
        self.other_models = []
        self.reg_model_names = []
        self.clf_model_names = []
        self.other_model_names = []
        self.feature_data = None
        self.timestep = None
        self.reg_predictions = []
        self.clf_predictions = []
        self.other_predictions = []
        
    def load_models(self, model_filter=None):
        """
        Load models from the models directory
        
        Args:
            model_filter (list): Additional filter for model types (optional)
            
        Returns:
            dict: Summary of loaded models
        """
        model_types_to_use = model_filter if model_filter else self.model_types
        
        # Find all .pkl model files in the models directory
        all_model_files = glob.glob(f'{self.models_dir}/*.pkl')
        
        if not all_model_files:
            raise FileNotFoundError(f"No .pkl model files found in the {self.models_dir} directory")
        
        # Sort models by filename (which may include timestamp)
        all_model_files = sorted(all_model_files)
        
        # Reset internal state
        self.reg_models = []
        self.clf_models = []
        self.other_models = []
        self.reg_model_names = []
        self.clf_model_names = []
        self.other_model_names = []
        
        # Separate models based on filename patterns
        for model_path in all_model_files:
            filename = os.path.basename(model_path).lower()
            
            # Filter by model types
            if not any(model_type in filename for model_type in model_types_to_use):
                continue
                
            if 'reg' in filename or 'regression' in filename:
                self.reg_model_names.append(model_path)
            elif 'clf' in filename or 'classification' in filename or 'classifier' in filename:
                self.clf_model_names.append(model_path)
            else:
                self.other_model_names.append(model_path)
        
        if self.verbose:
            print(f"Found {len(all_model_files)} total model files:")
            print(f"  - {len(self.reg_model_names)} regression models")
            print(f"  - {len(self.clf_model_names)} classification models")
            print(f"  - {len(self.other_model_names)} other/unknown type models")
        
        # Load regression models
        for model_path in self.reg_model_names:
            try:
                if self.verbose:
                    print(f"Loading regression model: {model_path}")
                self.reg_models.append(joblib.load(model_path))
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {model_path}: {e}")
        
        # Load classification models
        for model_path in self.clf_model_names:
            try:
                if self.verbose:
                    print(f"Loading classification model: {model_path}")
                self.clf_models.append(joblib.load(model_path))
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {model_path}: {e}")
        
        # Load other models
        for model_path in self.other_model_names:
            try:
                if self.verbose:
                    print(f"Loading general model: {model_path}")
                self.other_models.append(joblib.load(model_path))
            except Exception as e:
                if self.verbose:
                    print(f"Error loading {model_path}: {e}")
        
        return {
            'regression_models': len(self.reg_models),
            'classification_models': len(self.clf_models),
            'other_models': len(self.other_models),
            'total_loaded': len(self.reg_models) + len(self.clf_models) + len(self.other_models)
        }
    
    def load_data(self, data_file=None, auto_select=True):
        """
        Load EEG data for prediction
        
        Args:
            data_file (str): Specific data file to load (optional)
            auto_select (bool): Whether to auto-select first available file if specified file not found
            
        Returns:
            dict: Summary of loaded data
        """
        # Try to load specific file first
        if data_file:
            try:
                full_path = data_file if os.path.isabs(data_file) else f"{self.featuresets_dir}/{data_file}"
                real_data = pd.read_csv(full_path)
                if self.verbose:
                    print(f"Loaded specified EEG dataset: {real_data.shape}")
                    
                # Extract timestep for visualization
                if "Timestep" in real_data.columns:
                    self.timestep = real_data["Timestep"].values
                    if self.verbose:
                        print(f"Found real timestep data: {len(self.timestep)} points from {self.timestep.min():.1f} to {self.timestep.max():.1f}")
                else:
                    if self.verbose:
                        print("No timestep column found, using sample indices")
                    self.timestep = np.arange(len(real_data))
                
                # Remove Label and Timestep columns for prediction
                self.feature_data = real_data.drop(columns=["Label", "Timestep"], errors='ignore')
                
                if self.verbose:
                    print(f"Prepared {self.feature_data.shape[1]} features for prediction")
                
                return {
                    'data_shape': self.feature_data.shape,
                    'timestep_length': len(self.timestep),
                    'file_used': full_path
                }
                
            except FileNotFoundError:
                if not auto_select:
                    raise FileNotFoundError(f"Could not find specified data file: {data_file}")
                if self.verbose:
                    print(f"Could not find specified file: {data_file}")
        
        # Auto-select from available files
        try:
            # Try default file first
            default_file = f"{self.featuresets_dir}/local datasets_2025-07-15_17-28.csv"
            real_data = pd.read_csv(default_file)
            if self.verbose:
                print(f"Loaded default EEG dataset: {real_data.shape}")
            file_used = default_file
        except FileNotFoundError:
            if self.verbose:
                print("Could not find the default featureset file")
                print("Searching for available options...")
            
            featureset_files = glob.glob(f"{self.featuresets_dir}/*.csv")
            if featureset_files:
                if self.verbose:
                    print("Found these featureset files:")
                    for i, file in enumerate(featureset_files):
                        print(f"   {i+1}. {file}")
                
                # Use the first available file
                selected_file = featureset_files[0]
                if self.verbose:
                    print(f"Using: {selected_file}")
                
                real_data = pd.read_csv(selected_file)
                file_used = selected_file
            else:
                raise FileNotFoundError(f"No featureset files found in {self.featuresets_dir}. Cannot proceed without real EEG data.")
        
        # Extract timestep for visualization
        if "Timestep" in real_data.columns:
            self.timestep = real_data["Timestep"].values
            if self.verbose:
                print(f"Found real timestep data: {len(self.timestep)} points from {self.timestep.min():.1f} to {self.timestep.max():.1f}")
        else:
            if self.verbose:
                print("No timestep column found, using sample indices")
            self.timestep = np.arange(len(real_data))
        
        # Remove Label and Timestep columns for prediction
        self.feature_data = real_data.drop(columns=["Label", "Timestep"], errors='ignore')
        
        if self.verbose:
            print(f"Prepared {self.feature_data.shape[1]} features for prediction")
            print(f"Feature names: {list(self.feature_data.columns)[:3]}... (showing first 3)")
        
        return {
            'data_shape': self.feature_data.shape,
            'timestep_length': len(self.timestep),
            'file_used': file_used
        }
    
    def predict(self, data=None, models=None):
        """
        Make predictions using loaded models
        
        Args:
            data (pd.DataFrame): Data to predict on (uses loaded data if None)
            models (dict): Specific models to use (uses loaded models if None)
            
        Returns:
            dict: Predictions organized by model type
        """
        # Use provided data or loaded data
        prediction_data = data if data is not None else self.feature_data
        if prediction_data is None:
            raise ValueError("No data available for prediction. Please load data first.")
        
        # Use provided models or loaded models
        if models is None:
            reg_models = self.reg_models
            clf_models = self.clf_models
            other_models = self.other_models
            reg_model_names = self.reg_model_names
            clf_model_names = self.clf_model_names
            other_model_names = self.other_model_names
        else:
            reg_models = models.get('regression', [])
            clf_models = models.get('classification', [])
            other_models = models.get('other', [])
            reg_model_names = models.get('regression_names', [])
            clf_model_names = models.get('classification_names', [])
            other_model_names = models.get('other_names', [])
        
        # Reset predictions
        self.reg_predictions = []
        self.clf_predictions = []
        self.other_predictions = []
        
        # Regression models
        if reg_models:
            if self.verbose:
                print("\nMaking predictions with regression models:")
            for i, (model, model_name) in enumerate(zip(reg_models, reg_model_names)):
                try:
                    predictions = model.predict(prediction_data)
                    self.reg_predictions.append(predictions)
                    if self.verbose:
                        print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Error predicting with {model_name}: {e}")
        
        # Classification models
        if clf_models:
            if self.verbose:
                print("\nMaking predictions with classification models:")
            for i, (model, model_name) in enumerate(zip(clf_models, clf_model_names)):
                try:
                    predictions = model.predict(prediction_data)
                    self.clf_predictions.append(predictions)
                    if self.verbose:
                        print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Error predicting with {model_name}: {e}")
        
        # Other models
        if other_models:
            if self.verbose:
                print("\nMaking predictions with other/general models:")
            for i, (model, model_name) in enumerate(zip(other_models, other_model_names)):
                try:
                    predictions = model.predict(prediction_data)
                    self.other_predictions.append(predictions)
                    if self.verbose:
                        print(f"  Model {i+1} ({model_name}): predictions shape {predictions.shape}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Error predicting with {model_name}: {e}")
        
        return {
            'regression': self.reg_predictions,
            'classification': self.clf_predictions,
            'other': self.other_predictions
        }
    
    def get_predictions_summary(self):
        """
        Get summary statistics of predictions
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_models': len(self.reg_predictions) + len(self.clf_predictions) + len(self.other_predictions),
            'regression_models': len(self.reg_predictions),
            'classification_models': len(self.clf_predictions),
            'other_models': len(self.other_predictions)
        }
        
        if self.reg_predictions:
            summary['regression_stats'] = {
                'mean_predictions': [np.mean(pred) for pred in self.reg_predictions],
                'std_predictions': [np.std(pred) for pred in self.reg_predictions],
                'min_predictions': [np.min(pred) for pred in self.reg_predictions],
                'max_predictions': [np.max(pred) for pred in self.reg_predictions]
            }
        
        if self.clf_predictions:
            summary['classification_stats'] = {
                'unique_classes': [np.unique(pred) for pred in self.clf_predictions],
                'class_distributions': [{int(cls): int(np.sum(pred == cls)) for cls in np.unique(pred)} 
                                       for pred in self.clf_predictions]
            }
        
        return summary
    
    def visualize_predictions(self, show_plots=True, save_plots=False, output_dir="visualizations"):
        """
        Create visualizations of predictions
        
        Args:
            show_plots (bool): Whether to display plots
            save_plots (bool): Whether to save plots to files
            output_dir (str): Directory to save plots
            
        Returns:
            list: List of saved file paths (if save_plots=True)
        """
        if not any([self.reg_predictions, self.clf_predictions, self.other_predictions]):
            if self.verbose:
                print("No predictions available for visualization. Please run predict() first.")
            return []
        
        if self.verbose and show_plots:
            print("\n" + "="*60)
            print("GENERATING VISUALIZATIONS")
            print("="*60)
        
        saved_files = []
        
        if show_plots:
            # Create and show visualizations
            create_simple_visualizations(self.reg_predictions, self.clf_predictions, self.other_predictions,
                                        self.reg_model_names, self.clf_model_names, self.other_model_names, 
                                        self.timestep)
        
        if save_plots:
            # Save visualizations
            saved_files = save_simple_visualizations(self.reg_predictions, self.clf_predictions, self.other_predictions,
                                                   self.reg_model_names, self.clf_model_names, self.other_model_names, 
                                                   self.timestep, output_dir)
        
        return saved_files
    
    def save_predictions_to_csv(self, output_file=None, output_dir="results"):
        """
        Save predictions to CSV file
        
        Args:
            output_file (str): Specific output filename (optional)
            output_dir (str): Directory to save results
            
        Returns:
            str: Path to saved file
        """
        if not any([self.reg_predictions, self.clf_predictions, self.other_predictions]):
            raise ValueError("No predictions available to save. Please run predict() first.")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_file = f"predictions_{timestamp}.csv"
        
        full_path = os.path.join(output_dir, output_file)
        
        # Prepare data for saving
        results_df = pd.DataFrame()
        
        # Add timestep if available
        if self.timestep is not None:
            results_df['Timestep'] = self.timestep
        else:
            results_df['Index'] = range(len(self.feature_data)) if self.feature_data is not None else range(max(len(pred) for pred in self.reg_predictions + self.clf_predictions + self.other_predictions))
        
        # Add regression predictions
        for i, (pred, model_name) in enumerate(zip(self.reg_predictions, self.reg_model_names)):
            col_name = f"Reg_Model_{i+1}_{os.path.basename(model_name).replace('.pkl', '')}"
            results_df[col_name] = pred
        
        # Add classification predictions
        for i, (pred, model_name) in enumerate(zip(self.clf_predictions, self.clf_model_names)):
            col_name = f"Clf_Model_{i+1}_{os.path.basename(model_name).replace('.pkl', '')}"
            results_df[col_name] = pred
        
        # Add other predictions
        for i, (pred, model_name) in enumerate(zip(self.other_predictions, self.other_model_names)):
            col_name = f"Other_Model_{i+1}_{os.path.basename(model_name).replace('.pkl', '')}"
            results_df[col_name] = pred
        
        # Save to CSV
        results_df.to_csv(full_path, index=False)
        
        if self.verbose:
            print(f"Predictions saved to: {full_path}")
        
        return full_path

# Legacy function wrappers for backward compatibility

def load_all_models():
    """Legacy function - creates predictor instance and loads models"""
    predictor = EEGModelPredictor()
    predictor.load_models()
    return (predictor.reg_models, predictor.clf_models, predictor.other_models,
            predictor.reg_model_names, predictor.clf_model_names, predictor.other_model_names)

def load_real_data():
    """Legacy function - creates predictor instance and loads data"""
    predictor = EEGModelPredictor()
    predictor.load_data()
    return predictor.feature_data, predictor.timestep

def predict_and_visualize():
    """Legacy function - maintains original behavior for backward compatibility"""
    predictor = EEGModelPredictor()
    
    # Load models and data
    model_summary = predictor.load_models()
    data_summary = predictor.load_data()
    
    # Make predictions
    predictor.predict()
    
    # Create visualizations
    predictor.visualize_predictions(show_plots=True, save_plots=False)
    
    # Ask if user wants to save (maintaining original interactive behavior)
    save_option = input("\n💾 Would you like to save these visualizations? (y/n): ").lower().strip()
    if save_option in ['y', 'yes']:
        predictor.visualize_predictions(show_plots=False, save_plots=True)
    
    return predictor.clf_predictions, predictor.reg_predictions, predictor.other_predictions

# Convenience functions for simple usage

def quick_predict(models_dir="models", data_file=None, visualize=True, save_plots=False, 
                  model_types=None, verbose=True):
    """
    Quick one-line prediction function
    
    Args:
        models_dir (str): Directory containing model files
        data_file (str): Specific data file to use (auto-selects if None)
        visualize (bool): Whether to show visualizations
        save_plots (bool): Whether to save plots
        model_types (list): Model types to include
        verbose (bool): Whether to print status messages
        
    Returns:
        dict: Prediction results and summary
    """
    predictor = EEGModelPredictor(models_dir=models_dir, model_types=model_types, verbose=verbose)
    
    # Load models and data
    model_summary = predictor.load_models()
    data_summary = predictor.load_data(data_file=data_file)
    
    # Make predictions
    predictions = predictor.predict()
    
    # Visualizations
    if visualize or save_plots:
        predictor.visualize_predictions(show_plots=visualize, save_plots=save_plots)
    
    # Get summary
    summary = predictor.get_predictions_summary()
    
    return {
        'predictions': predictions,
        'summary': summary,
        'model_info': model_summary,
        'data_info': data_summary,
        'predictor': predictor  # Return predictor for advanced usage
    }

def batch_predict(models_list, data_files_list, output_dir="results", visualize_each=False):
    """
    Batch prediction for multiple model/data combinations
    
    Args:
        models_list (list): List of model directories or specific model files
        data_files_list (list): List of data files to predict on
        output_dir (str): Directory to save results
        visualize_each (bool): Whether to create visualizations for each combination
        
    Returns:
        list: List of results for each combination
    """
    results = []
    
    for i, models in enumerate(models_list):
        for j, data_file in enumerate(data_files_list):
            try:
                print(f"\nProcessing combination {i+1}-{j+1}: {models} with {data_file}")
                
                # Handle both directory and file inputs for models
                if os.path.isdir(models):
                    models_dir = models
                    predictor = EEGModelPredictor(models_dir=models_dir, verbose=True)
                else:
                    # If specific model file provided, handle differently
                    predictor = EEGModelPredictor(verbose=True)
                    # Custom loading would need to be implemented here
                
                predictor.load_models()
                predictor.load_data(data_file=data_file)
                predictions = predictor.predict()
                
                # Save predictions
                output_file = f"batch_predictions_{i+1}_{j+1}_{datetime.now().strftime('%H-%M-%S')}.csv"
                saved_path = predictor.save_predictions_to_csv(output_file=output_file, output_dir=output_dir)
                
                if visualize_each:
                    viz_files = predictor.visualize_predictions(show_plots=False, save_plots=True, 
                                                               output_dir=f"{output_dir}/visualizations")
                
                results.append({
                    'models': models,
                    'data_file': data_file,
                    'predictions': predictions,
                    'summary': predictor.get_predictions_summary(),
                    'saved_path': saved_path
                })
                
            except Exception as e:
                print(f"Error processing combination {i+1}-{j+1}: {e}")
                results.append({
                    'models': models,
                    'data_file': data_file,
                    'error': str(e)
                })
    
    return results

# Visualization helper functions (used by both class and legacy functions)

def create_simple_visualizations(all_reg_predictions, all_clf_predictions, all_other_predictions, 
                               reg_model_names, clf_model_names, other_model_names, timestep=None):
    """Create simple, effective visualizations for model predictions"""
    
    # Set a clean style
    plt.style.use('default')
    
    # 1. Overview plot - all models in one view
    create_overview_plot(all_reg_predictions, all_clf_predictions, all_other_predictions,
                        reg_model_names, clf_model_names, other_model_names, timestep)
    
    # 2. Individual model plots (only if there are multiple models)
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
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    elif all_reg_predictions or all_clf_predictions or all_other_predictions:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
        ax2 = None
        ax3 = None
        ax4 = None
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
        
        # Use distinct colors for better differentiation
        colors = plt.cm.Set1(np.linspace(0, 1, len(all_reg_predictions)))
        
        # Calculate common Y-axis scale for all regression models
        all_reg_values = np.concatenate(all_reg_predictions)
        y_min, y_max = np.min(all_reg_values), np.max(all_reg_values)
        y_margin = (y_max - y_min) * 0.1  # Add 10% margin
        
        for i, (predictions, model_name, color) in enumerate(zip(all_reg_predictions, reg_model_names, colors)):
            predictions_sampled = predictions[::step]
            label = f"Reg Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=color, 
                    linewidth=2, alpha=0.9, marker='o', markersize=1)
        
        ax1.set_title('Regression Model Predictions (Downsampled for Clarity)', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Predicted Value')
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)  # Set common Y-axis scale
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add summary statistics
        if len(all_reg_predictions) > 1:
            mean_pred = np.mean([np.mean(pred) for pred in all_reg_predictions])
            ax1.text(0.02, 0.98, f'Avg Mean: {mean_pred:.3f}', transform=ax1.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top')
    
    # Plot classification models (similar structure)
    if all_clf_predictions and ax2 is None:
        # If we only have classification models, show time series with timestep
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
        
        # Plot classification time series
        colors = plt.cm.Set1(np.linspace(0, 1, len(all_clf_predictions)))
        
        for i, (predictions, model_name, color) in enumerate(zip(all_clf_predictions, clf_model_names, colors)):
            predictions_sampled = predictions[::step]
            label = f"Clf Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=color, 
                    linewidth=2, alpha=0.9, marker='o', markersize=2)
        
        ax1.set_title('Classification Model Predictions Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Predicted Class')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
    plt.suptitle(f'EEG Model Predictions Overview ({total_models} models)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_comparison_plot(predictions_list, model_names, title, timestep=None):
    """Create a simple comparison plot for multiple models of the same type"""
    
    if len(predictions_list) < 2:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Time series comparison
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Timestep'
    else:
        x_axis = np.arange(len(predictions_list[0]))
        x_label = 'Sample Index'
    
    # Downsample for cleaner visualization if we have too many points
    if len(x_axis) > 500:
        step = len(x_axis) // 300
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
                              reg_model_names, clf_model_names, other_model_names, timestep=None, 
                              output_dir="visualizations"):
    """Save simple visualizations to files"""
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    saved_files = []
    
    print(f"Saving visualizations to {output_dir}/...")
    
    # Save overview plot (simplified version without complex subplots for saving)
    total_models = len(all_reg_predictions) + len(all_clf_predictions) + len(all_other_predictions)
    if total_models > 0:
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        # Plot all predictions in one view
        all_predictions = all_reg_predictions + all_clf_predictions + all_other_predictions
        all_names = reg_model_names + clf_model_names + other_model_names
        all_types = ['Reg']*len(all_reg_predictions) + ['Clf']*len(all_clf_predictions) + ['Other']*len(all_other_predictions)
        
        if timestep is not None and len(timestep) > 0:
            x_axis = timestep
            x_label = 'Timestep'
        else:
            x_axis = np.arange(len(all_predictions[0]))
            x_label = 'Sample Index'
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(all_predictions)))
        
        for i, (predictions, model_name, model_type, color) in enumerate(zip(all_predictions, all_names, all_types, colors)):
            label = f"{model_type} Model {i+1}: {os.path.basename(model_name).replace('.pkl', '')}"
            ax.plot(x_axis, predictions, label=label, color=color, linewidth=2, alpha=0.9)
        
        ax.set_title('EEG Model Predictions Overview', fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Predicted Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"{output_dir}/model_overview_{timestamp}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved_files.append(filename)
    
    print(f"\nSuccessfully saved {len(saved_files)} visualization(s):")
    for file in saved_files:
        print(f"   - {file}")
    
    return saved_files

def check_saved_visualizations():
    """Check the size and status of recently saved visualizations"""
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
    # Maintain original behavior for backward compatibility
    clf_predictions, reg_predictions, other_predictions = predict_and_visualize()
