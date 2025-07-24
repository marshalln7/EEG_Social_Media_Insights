import os
import pandas as pd
import joblib
from matplotlib import pyplot as plt
import numpy as np
import glob
from datetime import datetime
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from scipy import stats

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
    
    def __init__(self, models_dir="models", featuresets_dir="local featuresets", 
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
        Load models from the models directory - selecting only the latest model for each type
        
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
        
        # Reset internal state
        self.reg_models = []
        self.clf_models = []
        self.other_models = []
        self.reg_model_names = []
        self.clf_model_names = []
        self.other_model_names = []
        
        # Dictionary to store latest models by category and type
        latest_models = {
            'reg': {},
            'clf': {},
            'other': {}
        }
        
        # Separate models based on filename patterns and find latest for each type
        for model_path in all_model_files:
            filename = os.path.basename(model_path).lower()
            
            # Filter by model types
            if not any(model_type in filename for model_type in model_types_to_use):
                continue
            
            # Extract model category (concentration, mendeley, etc.)
            model_category = None
            for model_type in model_types_to_use:
                if model_type in filename:
                    model_category = model_type
                    break
            
            if not model_category:
                continue
                
            # Get file modification time for comparison
            mod_time = os.path.getmtime(model_path)
            
            if 'reg' in filename or 'regression' in filename:
                category = 'reg'
            elif 'clf' in filename or 'classification' in filename or 'classifier' in filename:
                category = 'clf'
            else:
                category = 'other'
            
            # Keep only the latest model for each category-type combination
            key = f"{model_category}_{category}"
            if key not in latest_models[category] or mod_time > latest_models[category][key]['mod_time']:
                latest_models[category][key] = {
                    'path': model_path,
                    'mod_time': mod_time
                }
        
        # Extract the latest model paths
        for category_models in latest_models['reg'].values():
            self.reg_model_names.append(category_models['path'])
        for category_models in latest_models['clf'].values():
            self.clf_model_names.append(category_models['path'])
        for category_models in latest_models['other'].values():
            self.other_model_names.append(category_models['path'])
        
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
                
                # Use the most recent file (by modification time)
                selected_file = max(featureset_files, key=os.path.getmtime)
                if self.verbose:
                    print(f"Auto-selecting latest file: {selected_file}")
                
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
    """Create model-specific visualizations - automatically detects model type and uses appropriate scales"""
    
    # Set a clean style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # Detect model types from filenames
    model_types = set()
    all_model_names = reg_model_names + clf_model_names + other_model_names
    
    for model_name in all_model_names:
        filename = os.path.basename(model_name).lower()
        if "concentration" in filename:
            model_types.add("concentration")
        elif "mendeley" in filename:
            model_types.add("mendeley")
        elif "neurosense" in filename:
            model_types.add("neurosense")
    
    # Create visualizations based on detected model types
    if "concentration" in model_types:
        # Create concentration-specific visualizations
        conc_reg_preds = [pred for pred, name in zip(all_reg_predictions, reg_model_names) 
                         if "concentration" in os.path.basename(name).lower()]
        conc_clf_preds = [pred for pred, name in zip(all_clf_predictions, clf_model_names) 
                         if "concentration" in os.path.basename(name).lower()]
        conc_reg_names = [name for name in reg_model_names 
                         if "concentration" in os.path.basename(name).lower()]
        conc_clf_names = [name for name in clf_model_names 
                         if "concentration" in os.path.basename(name).lower()]
        
        if conc_reg_preds or conc_clf_preds:
            create_concentration_overview_plot(conc_reg_preds, conc_clf_preds, 
                                             conc_reg_names, conc_clf_names, timestep)
        if conc_reg_preds:
            create_concentration_regression_plot(conc_reg_preds, conc_reg_names, timestep)
        if conc_clf_preds:
            create_concentration_classification_plot(conc_clf_preds, conc_clf_names, timestep)
    
    if "mendeley" in model_types:
        # Create mendeley-specific visualizations
        mend_reg_preds = [pred for pred, name in zip(all_reg_predictions, reg_model_names) 
                         if "mendeley" in os.path.basename(name).lower()]
        mend_clf_preds = [pred for pred, name in zip(all_clf_predictions, clf_model_names) 
                         if "mendeley" in os.path.basename(name).lower()]
        mend_reg_names = [name for name in reg_model_names 
                         if "mendeley" in os.path.basename(name).lower()]
        mend_clf_names = [name for name in clf_model_names 
                         if "mendeley" in os.path.basename(name).lower()]
        
        if mend_reg_preds or mend_clf_preds:
            create_mendeley_overview_plot(mend_reg_preds, mend_clf_preds, 
                                        mend_reg_names, mend_clf_names, timestep)
        if mend_reg_preds:
            create_mendeley_regression_plot(mend_reg_preds, mend_reg_names, timestep)
        if mend_clf_preds:
            create_mendeley_classification_plot(mend_clf_preds, mend_clf_names, timestep)

def create_improved_regression_plots(all_reg_predictions, reg_model_names, timestep=None):
    """Create improved regression plots with consistent scaling and better visualization for concentration models"""
    
    if not all_reg_predictions:
        return
    
    # Set fixed scale for concentration models (0=Relaxed, 1=Neutral, 2=Concentrated)
    y_lim = (-0.1, 2.1)  # Fixed scale with small margin
    
    # Prepare x-axis with more descriptive labeling
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        x_axis = np.arange(len(all_reg_predictions[0]))
        x_label = 'EEG Sample Index (chronological order)'
    
    # Create figure with multiple subplots
    n_models = len(all_reg_predictions)
    if n_models == 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = [axes]
    else:
        # Create a grid layout: time series on top, boxplot below
        fig = plt.figure(figsize=(20, 10))
        
        # Top row: Individual time series plots (same scale)
        gs = fig.add_gridspec(2, n_models, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Use distinct colors for better differentiation
        distinct_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        colors = distinct_colors[:n_models] if n_models <= len(distinct_colors) else plt.cm.tab10(np.linspace(0, 1, n_models))
        
        # Individual time series plots with same scale
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            ax = fig.add_subplot(gs[0, i])
            
            # Downsample for cleaner visualization if needed
            if len(x_axis) > 1000:
                step = len(x_axis) // 500
                x_sampled = x_axis[::step]
                pred_sampled = predictions[::step]
            else:
                x_sampled = x_axis
                pred_sampled = predictions
            
            ax.plot(x_sampled, pred_sampled, linewidth=3, alpha=0.9, 
                   color=colors[i])
            
            # Create more descriptive, cleaner titles for concentration models
            short_name = os.path.basename(model_name).replace(".pkl", "")
            model_type = "Attention/Focus"
            
            # Truncate very long names and add line breaks if needed
            if len(short_name) > 15:
                display_name = short_name[:12] + "..."
            else:
                display_name = short_name.replace("_", " ")
            
            ax.set_title(f'{model_type} Model {i+1}\n{display_name}', fontweight='bold', fontsize=10)
            
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel('Concentration Level', fontsize=9)
            ax.set_ylim(y_lim)  # Fixed scale for concentration
            
            # Set y-axis ticks and labels for concentration levels
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=8)
            
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Add horizontal reference lines for concentration levels
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1) 
            ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Bottom: Boxplot comparison
        ax_box = fig.add_subplot(gs[1, :])
        box_data = all_reg_predictions
        # Create shorter labels for boxplots
        box_labels = [f'M{i+1}' for i in range(n_models)]
        
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, 
                           showmeans=True, meanline=True)
        
        # Use more distinct colors for better differentiation
        colors = distinct_colors[:n_models] if n_models <= len(distinct_colors) else plt.cm.tab10(np.linspace(0, 1, n_models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax_box.set_title('EEG Concentration Models - Statistical Distribution Analysis', fontweight='bold', fontsize=12)
        ax_box.set_ylabel('Concentration Level', fontsize=10)
        ax_box.set_ylim(y_lim)  # Fixed scale for concentration
        
        # Set y-axis ticks and labels for concentration levels
        ax_box.set_yticks([0, 1, 2])
        ax_box.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=9)
        
        ax_box.tick_params(axis='x', labelsize=9)
        ax_box.grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        ax_box.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax_box.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax_box.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.suptitle(f'EEG-Based Concentration Models Analysis - {n_models} Models (Muse Headset Data)', 
                    fontsize=18, fontweight='bold', y=0.95)
    
    # Better layout adjustment
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_improved_classification_plots(all_clf_predictions, clf_model_names, timestep=None):
    """Create separate, cleaner classification plots for concentration models"""
    
    if not all_clf_predictions:
        return
    
    n_models = len(all_clf_predictions)
    
    # Prepare x-axis with descriptive labels
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        x_axis = np.arange(len(all_clf_predictions[0]))
        x_label = 'EEG Sample Index (chronological order)'
    
    # Use distinct colors for better differentiation
    distinct_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E']
    colors = distinct_colors[:n_models] if n_models <= len(distinct_colors) else plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Create 3 separate plots for better clarity
    
    # Plot 1: Individual time series plots (one per model)
    create_individual_classification_plots(all_clf_predictions, clf_model_names, x_axis, x_label, colors)
    
    # Plot 2: Comparison plots (boxplot and frequency)
    create_classification_comparison_plots(all_clf_predictions, clf_model_names, colors)
    
    # Plot 3: Class distribution analysis
    create_classification_distribution_plots(all_clf_predictions, clf_model_names, colors)

def create_individual_concentration_classification_plots(all_clf_predictions, clf_model_names, x_axis, x_label, colors):
    """Create individual time series plots for each concentration classification model"""
    
    n_models = len(all_clf_predictions)
    
    # Determine grid layout
    if n_models <= 2:
        rows, cols = 1, n_models
        fig_size = (12 * n_models, 6)
    elif n_models <= 4:
        rows, cols = 2, 2
        fig_size = (16, 10)
    else:
        rows = (n_models + 2) // 3
        cols = 3
        fig_size = (18, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        ax = axes[i]
        
        # Downsample if needed for cleaner visualization
        if len(x_axis) > 800:
            step = len(x_axis) // 400
            x_sampled = x_axis[::step]
            pred_sampled = predictions[::step]
        else:
            x_sampled = x_axis
            pred_sampled = predictions
        
        # Create clean step plot for better class visibility
        ax.step(x_sampled, pred_sampled, where='mid', color=colors[i], 
               linewidth=2.5, alpha=0.9, label=f'Model {i+1}')
        
        # Fill between steps for better visibility
        ax.fill_between(x_sampled, pred_sampled, alpha=0.3, color=colors[i], step='mid')
        
        # Create descriptive model names for concentration
        short_name = os.path.basename(model_name).replace(".pkl", "")
        model_type = "Attention/Focus"
        class_description = "Concentration Levels"
            
        if len(short_name) > 20:
            display_name = short_name[:17] + "..."
        else:
            display_name = short_name.replace("_", " ")
        
        ax.set_title(f'{model_type} Classification Model {i+1}\n{display_name}', 
                    fontweight='bold', fontsize=12, pad=15)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Concentration Level', fontsize=10)
        
        # Set fixed y-axis for concentration levels
        ax.set_ylim(-0.1, 2.1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Add horizontal reference lines for concentration levels
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add concentration level info
        unique_classes = np.unique(predictions)
        class_text = f'Concentration Classes: {", ".join(map(str, sorted(unique_classes)))} (0=Relaxed, 1=Neutral, 2=Concentrated)'
            
        ax.text(0.02, 0.98, class_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Individual EEG Concentration Classification Model Time Series - {n_models} Models (Muse Headset AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_concentration_classification_comparison_plots(all_clf_predictions, clf_model_names, colors):
    """Create comparison plots for concentration classification models"""
    
    n_models = len(all_clf_predictions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Enhanced boxplot
    box_data = all_clf_predictions
    box_labels = [f'Model {i+1}' for i in range(n_models)]
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.6)
    
    # Apply distinct colors to boxplots
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style other boxplot elements
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax1.set_title('EEG Concentration Classification Models - Statistical Distribution Analysis', 
                 fontweight='bold', fontsize=14)
    ax1.set_ylabel('Concentration Level', fontsize=11)
    
    # Set fixed y-axis for concentration levels
    ax1.set_ylim(-0.1, 2.1)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=9)
    
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_facecolor('#fafafa')
    
    # Add horizontal reference lines
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Right: Enhanced class frequency comparison
    # Always show all concentration levels (0, 1, 2) even if frequency is 0
    all_classes = [0, 1, 2]  # Fixed concentration levels
    
    x_pos = np.arange(len(all_classes))
    width = 0.8 / n_models
    
    # Find max frequency for label positioning
    max_freq = 0
    for predictions in all_clf_predictions:
        for class_val in all_classes:
            freq = np.sum(predictions == class_val)
            max_freq = max(max_freq, freq)
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        frequencies = [np.sum(predictions == class_val) for class_val in all_classes]
        offset = (i - n_models/2 + 0.5) * width
        
        model_label = f'Concentration Model {i+1}'
            
        bars = ax2.bar(x_pos + offset, frequencies, width, 
                      label=model_label, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # Add frequency labels on bars (show all, including 0)
        for bar, freq in zip(bars, frequencies):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_freq*0.01,
                    f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Concentration Level Categories', fontsize=11)
    ax2.set_ylabel('Frequency Count (Number of Predictions)', fontsize=11)
    ax2.set_title('EEG Concentration Classification Models - Class Frequency Distribution', 
                 fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos)
    
    # Create concentration level labels (always show all 3)
    class_labels = ["Relaxed (0)", "Neutral (1)", "Concentrated (2)"]
    ax2.set_xticklabels(class_labels, fontsize=10)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    plt.suptitle('EEG-Based Concentration Classification Models - Statistical Comparison (Muse Headset Data)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_concentration_classification_distribution_plots(all_clf_predictions, clf_model_names, colors):
    """Create distribution analysis plots for concentration classification models"""
    
    n_models = len(all_clf_predictions)
    
    # Get all unique classes across models
    all_classes = set()
    for predictions in all_clf_predictions:
        all_classes.update(predictions)
    all_classes = sorted(list(all_classes))
    
    # Create pie charts for each model
    if n_models <= 3:
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    else:
        rows = (n_models + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
        axes = axes.flatten()
    
    if n_models == 1:
        axes = [axes]
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        ax = axes[i]
        
        # Calculate class frequencies
        class_counts = {}
        for class_val in all_classes:
            count = np.sum(predictions == class_val)
            if count > 0:
                class_counts[class_val] = count
        
        if class_counts:
            # Create concentration level labels
            labels = []
            for c in class_counts.keys():
                if c == 0:
                    labels.append("Relaxed (0)")
                elif c == 1:
                    labels.append("Neutral (1)")
                elif c == 2:
                    labels.append("Concentrated (2)")
                else:
                    labels.append(f"Level {c}")
            
            sizes = list(class_counts.values())
            
            # Use distinct colors - green for relaxed, orange for neutral, red for concentrated
            pie_colors = []
            for c in class_counts.keys():
                if c == 0:
                    pie_colors.append('#2ECC71')  # Green for relaxed
                elif c == 1:
                    pie_colors.append('#F39C12')  # Orange for neutral
                elif c == 2:
                    pie_colors.append('#E74C3C')  # Red for concentrated
                else:
                    pie_colors.append(colors[i % len(colors)])
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', 
                                             startangle=90, textprops={'fontsize': 10})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        
        # Create descriptive model titles
        short_name = os.path.basename(model_name).replace(".pkl", "")
        model_type = "Attention/Focus"
            
        if len(short_name) > 15:
            display_name = short_name[:12] + "..."
        else:
            display_name = short_name.replace("_", " ")
        
        ax.set_title(f'{model_type} Model {i+1}\n{display_name}\nConcentration Distribution (%)', 
                    fontweight='bold', fontsize=11)
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('EEG Concentration Classification Models - Level Distribution Analysis (Muse EEG AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_individual_classification_plots(all_clf_predictions, clf_model_names, x_axis, x_label, colors):
    """Create individual time series plots for each classification model"""
    
    n_models = len(all_clf_predictions)
    
    # Determine grid layout
    if n_models <= 2:
        rows, cols = 1, n_models
        fig_size = (12 * n_models, 6)
    elif n_models <= 4:
        rows, cols = 2, 2
        fig_size = (16, 10)
    else:
        rows = (n_models + 2) // 3
        cols = 3
        fig_size = (18, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        ax = axes[i]
        
        # Downsample if needed for cleaner visualization
        if len(x_axis) > 800:
            step = len(x_axis) // 400
            x_sampled = x_axis[::step]
            pred_sampled = predictions[::step]
        else:
            x_sampled = x_axis
            pred_sampled = predictions
        
        # Create clean step plot for better class visibility
        ax.step(x_sampled, pred_sampled, where='mid', color=colors[i], 
               linewidth=2.5, alpha=0.9, label=f'Model {i+1}')
        
        # Fill between steps for better visibility
        ax.fill_between(x_sampled, pred_sampled, alpha=0.3, color=colors[i], step='mid')
        
        # Create descriptive model names and types
        short_name = os.path.basename(model_name).replace(".pkl", "")
        if "concentration" in short_name.lower():
            model_type = "Attention/Focus"
            class_description = "Focus Levels"
        elif "mendeley" in short_name.lower():
            model_type = "Emotional State"
            class_description = "Emotion Categories"
        elif "neurosense" in short_name.lower():
            model_type = "Music Emotion"
            class_description = "Arousal/Valence"
        else:
            model_type = "EEG Analysis"
            class_description = "Predicted Classes"
            
        if len(short_name) > 20:
            display_name = short_name[:17] + "..."
        else:
            display_name = short_name.replace("_", " ")
        
        ax.set_title(f'{model_type} Classification Model {i+1}\n{display_name}', 
                    fontweight='bold', fontsize=12, pad=15)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('EEG-Predicted Class Category', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Add more informative class info
        unique_classes = np.unique(predictions)
        if "concentration" in model_name.lower():
            class_text = f'Focus Classes: {", ".join(map(str, sorted(unique_classes)))} (Higher=More Focused)'
        elif "emotion" in model_name.lower() or "mendeley" in model_name.lower():
            class_mapping = {1: "Anger", 2: "Fear", 3: "Happiness", 4: "Sadness"}
            class_labels = [class_mapping.get(c, f"Class {c}") for c in sorted(unique_classes)]
            class_text = f'Emotions: {", ".join(class_labels)}'
        elif "neurosense" in model_name.lower():
            class_text = f'Arousal/Valence Levels: {", ".join(map(str, sorted(unique_classes)))} (Higher=More Intense)'
        else:
            class_text = f'Classes: {", ".join(map(str, sorted(unique_classes)))}'
            
        ax.text(0.02, 0.98, class_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Individual EEG Classification Model Time Series - {n_models} Models (Muse Headset AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_classification_comparison_plots(all_clf_predictions, clf_model_names, colors):
    """Create comparison plots for classification models"""
    
    n_models = len(all_clf_predictions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Enhanced boxplot
    box_data = all_clf_predictions
    box_labels = [f'Model {i+1}' for i in range(n_models)]
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.6)
    
    # Apply distinct colors to boxplots
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style other boxplot elements
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax1.set_title('EEG Classification Models - Statistical Distribution Analysis', 
                 fontweight='bold', fontsize=14)
    ax1.set_ylabel('EEG-Predicted Class Values', fontsize=11)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_facecolor('#fafafa')
    
    # Right: Enhanced class frequency comparison with better labels
    all_classes = set()
    for predictions in all_clf_predictions:
        all_classes.update(predictions)
    all_classes = sorted(list(all_classes))
    
    x_pos = np.arange(len(all_classes))
    width = 0.8 / n_models
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        frequencies = [np.sum(predictions == class_val) for class_val in all_classes]
        offset = (i - n_models/2 + 0.5) * width
        
        # Create more descriptive model labels
        if "concentration" in model_name.lower():
            model_label = f'Focus Model {i+1}'
        elif "emotion" in model_name.lower() or "mendeley" in model_name.lower():
            model_label = f'Emotion Model {i+1}'
        elif "neurosense" in model_name.lower():
            model_label = f'Music Model {i+1}'
        else:
            model_label = f'Model {i+1}'
            
        bars = ax2.bar(x_pos + offset, frequencies, width, 
                      label=model_label, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # Add frequency labels on bars
        for bar, freq in zip(bars, frequencies):
            if freq > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(frequencies)*0.01,
                        f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('EEG-Predicted Class Categories', fontsize=11)
    ax2.set_ylabel('Frequency Count (Number of Predictions)', fontsize=11)
    ax2.set_title('EEG Classification Models - Class Frequency Distribution', 
                 fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos)
    
    # Create more meaningful class labels
    class_labels = []
    for c in all_classes:
        if any("emotion" in name.lower() or "mendeley" in name.lower() for name in clf_model_names):
            emotion_map = {1: "Anger", 2: "Fear", 3: "Happy", 4: "Sad"}
            class_labels.append(emotion_map.get(c, f"Class {c}"))
        elif any("concentration" in name.lower() for name in clf_model_names):
            class_labels.append(f"Focus Lv.{c}")
        else:
            class_labels.append(f"Class {c}")
    
    ax2.set_xticklabels(class_labels, fontsize=10)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    plt.suptitle('EEG-Based Classification Models - Statistical Comparison (Muse Headset Data)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_classification_distribution_plots(all_clf_predictions, clf_model_names, colors):
    """Create distribution analysis plots for classification models"""
    
    n_models = len(all_clf_predictions)
    
    # Get all unique classes across models
    all_classes = set()
    for predictions in all_clf_predictions:
        all_classes.update(predictions)
    all_classes = sorted(list(all_classes))
    
    # Create pie charts for each model
    if n_models <= 3:
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    else:
        rows = (n_models + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
        axes = axes.flatten()
    
    if n_models == 1:
        axes = [axes]
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        ax = axes[i]
        
        # Calculate class frequencies
        class_counts = {}
        for class_val in all_classes:
            count = np.sum(predictions == class_val)
            if count > 0:
                class_counts[class_val] = count
        
        if class_counts:
            # Create meaningful labels based on model type
            model_file = os.path.basename(model_name).lower()
            labels = []
            for c in class_counts.keys():
                if "emotion" in model_file or "mendeley" in model_file:
                    emotion_map = {1: "Anger", 2: "Fear", 3: "Happiness", 4: "Sadness"}
                    labels.append(emotion_map.get(c, f"Emotion {c}"))
                elif "concentration" in model_file:
                    labels.append(f"Focus Level {c}")
                elif "neurosense" in model_file:
                    labels.append(f"Arousal/Val. {c}")
                else:
                    labels.append(f"Class {c}")
            
            sizes = list(class_counts.values())
            
            # Use distinct colors
            pie_colors = [colors[j % len(colors)] for j in range(len(labels))]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', 
                                             startangle=90, textprops={'fontsize': 10})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        
        # Create descriptive model titles
        short_name = os.path.basename(model_name).replace(".pkl", "")
        if "concentration" in short_name.lower():
            model_type = "Attention/Focus"
        elif "mendeley" in short_name.lower():
            model_type = "Emotional State" 
        elif "neurosense" in short_name.lower():
            model_type = "Music Emotion"
        else:
            model_type = "EEG Analysis"
            
        if len(short_name) > 15:
            display_name = short_name[:12] + "..."
        else:
            display_name = short_name.replace("_", " ")
        
        ax.set_title(f'{model_type} Model {i+1}\n{display_name}\nClass Distribution (%)', 
                    fontweight='bold', fontsize=11)
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('EEG Classification Models - Class Distribution Analysis (Muse EEG AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_concentration_overview_plot(all_reg_predictions, all_clf_predictions, reg_model_names, clf_model_names, timestep=None):
    """Create the main concentration overview plot matching your image"""
    
    if not all_reg_predictions and not all_clf_predictions:
        print("No models to visualize")
        return
    
    # Create the 2x2 subplot layout as shown in your image
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Prepare x-axis with better labeling
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        # Use the longest prediction array for x-axis
        if all_reg_predictions and all_clf_predictions:
            max_len = max(len(all_reg_predictions[0]), len(all_clf_predictions[0]))
        elif all_reg_predictions:
            max_len = len(all_reg_predictions[0])
        else:
            max_len = len(all_clf_predictions[0])
        x_axis = np.arange(max_len)
        x_label = 'Time (seconds from EEG recording start)'
    
    # TOP LEFT: Plot regression models
    if all_reg_predictions:
        # Downsample if needed
        if len(x_axis) > 500:
            step = len(x_axis) // 300
            x_axis_sampled = x_axis[::step]
        else:
            step = 1
            x_axis_sampled = x_axis
        
        # Use distinct colors
        distinct_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        colors = distinct_colors[:len(all_reg_predictions)]
        
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            predictions_sampled = predictions[::step]
            short_name = os.path.basename(model_name).replace('.pkl', '')
            if len(short_name) > 12:
                short_name = short_name[:9] + "..."
            label = f"R1: {short_name}"
            ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=colors[i % len(colors)], 
                    linewidth=3, alpha=0.9, marker='o', markersize=2)
        
        ax1.set_title('EEG Concentration Regression Models - Temporal Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=10)
        ax1.set_ylabel('Concentration Level', fontsize=10)
        ax1.set_ylim(-0.1, 2.1)  # Fixed concentration scale
        
        # Set y-axis ticks and labels
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=9)
        
        # Add reference lines
        ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
    else:
        ax1.text(0.5, 0.5, 'No Regression Models', ha='center', va='center', transform=ax1.transAxes, fontsize=16)
        ax1.set_title('EEG Concentration Regression Models - Temporal Predictions', fontsize=14, fontweight='bold')
    
    # TOP RIGHT: Plot classification models
    if all_clf_predictions:
        # Use same x-axis setup
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            predictions_sampled = predictions[::step]
            short_name = os.path.basename(model_name).replace('.pkl', '')
            if len(short_name) > 12:
                short_name = short_name[:9] + "..."
            label = f"C1: {short_name}"
            ax2.step(x_axis_sampled, predictions_sampled, where='mid', label=label, color='#E74C3C', 
                    linewidth=3, alpha=0.9)
        
        ax2.set_title('EEG Concentration Classification Models - Temporal Class Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=10)
        ax2.set_ylabel('Concentration Level', fontsize=10)
        ax2.set_ylim(-0.1, 2.1)
        
        # Set y-axis ticks and labels
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=9)
        
        # Add reference lines
        ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.set_facecolor('#f8f9fa')
    else:
        ax2.text(0.5, 0.5, 'No Classification Models', ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        ax2.set_title('EEG Concentration Classification Models - Temporal Class Predictions', fontsize=14, fontweight='bold')
    
    # BOTTOM LEFT: Classification frequency bar chart
    if all_clf_predictions:
        # Always show all concentration levels (0, 1, 2) even if frequency is 0
        all_classes = [0, 1, 2]  # Fixed concentration levels
        
        x_pos = np.arange(len(all_classes))
        width = 0.8 / len(all_clf_predictions)
        
        # Find max frequency for label positioning
        max_freq = 0
        for predictions in all_clf_predictions:
            for class_val in all_classes:
                freq = np.sum(predictions == class_val)
                max_freq = max(max_freq, freq)
        
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            frequencies = [np.sum(predictions == class_val) for class_val in all_classes]
            offset = (i - len(all_clf_predictions)/2 + 0.5) * width
            
            bars = ax3.bar(x_pos + offset, frequencies, width, 
                          label=f'C{i+1}', color='#E74C3C', alpha=0.8,
                          edgecolor='black', linewidth=1)
            
            # Add frequency labels on bars (show all, including 0)
            for bar, freq in zip(bars, frequencies):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_freq*0.01,
                        f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Predicted Classes', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Classification Models - Class Distribution', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        
        # Create concentration level labels (always show all 3)
        class_labels = ["Relaxed", "Neutral", "Concentrated"]
        ax3.set_xticklabels(class_labels, fontsize=10)
        if len(all_clf_predictions) > 1:
            ax3.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax3.grid(True, alpha=0.4, linestyle='--')
        ax3.set_facecolor('#fafafa')
    else:
        ax3.text(0.5, 0.5, 'No Classification Models', ha='center', va='center', transform=ax3.transAxes, fontsize=16)
        ax3.set_title('Classification Models - Class Distribution', fontsize=14, fontweight='bold')
    
    # BOTTOM RIGHT: Classification boxplot
    if all_clf_predictions:
        box_data = all_clf_predictions
        box_labels = [f'C{i+1}' for i in range(len(all_clf_predictions))]
        
        bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True, 
                       showmeans=True, meanline=True)
        
        # Color the boxplots
        for patch in bp['boxes']:
            patch.set_facecolor('#E74C3C')
            patch.set_alpha(0.7)
        
        ax4.set_title('Concentration Classification Models - Distribution Comparison', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Concentration Level', fontsize=10)
        ax4.set_ylim(-0.1, 2.1)
        
        # Set y-axis ticks and labels
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=9)
        
        # Add reference lines
        ax4.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax4.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax4.tick_params(axis='x', labelsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor('#fafafa')
    else:
        ax4.text(0.5, 0.5, 'No Classification Models', ha='center', va='center', transform=ax4.transAxes, fontsize=16)
        ax4.set_title('Concentration Classification Models - Distribution Comparison', fontsize=14, fontweight='bold')
    
    total_models = len(all_reg_predictions) + len(all_clf_predictions)
    plt.suptitle(f'EEG Concentration Model Predictions Overview - {total_models} Models (Muse Headset: AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_concentration_regression_plot(all_reg_predictions, reg_model_names, timestep=None):
    """Create a dedicated plot for concentration regression models"""
    
    if not all_reg_predictions:
        return
    
    # Prepare x-axis
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        x_axis = np.arange(len(all_reg_predictions[0]))
        x_label = 'EEG Sample Index (chronological order)'
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Downsample if needed
    if len(x_axis) > 500:
        step = len(x_axis) // 300
        x_axis_sampled = x_axis[::step]
    else:
        step = 1
        x_axis_sampled = x_axis
    
    # Use distinct colors
    distinct_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    colors = distinct_colors[:len(all_reg_predictions)]
    
    # TOP: Time series plot
    for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
        predictions_sampled = predictions[::step]
        short_name = os.path.basename(model_name).replace('.pkl', '')
        if len(short_name) > 15:
            short_name = short_name[:12] + "..."
        label = f"Model {i+1}: {short_name}"
        ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=colors[i % len(colors)], 
                linewidth=3, alpha=0.9, marker='o', markersize=1.5)
    
    ax1.set_title('EEG Concentration Regression Models - Continuous Predictions Over Time', 
                 fontsize=16, fontweight='bold')
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel('Concentration Level (Continuous Scale)', fontsize=12)
    ax1.set_ylim(-0.1, 2.1)  # Fixed concentration scale
    
    # Set y-axis ticks and labels
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=10)
    
    # Add reference lines
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.6, linewidth=2, label='Relaxed Threshold')
    ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.6, linewidth=2, label='Neutral Threshold')
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Concentrated Threshold')
    
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # BOTTOM: Statistical distribution boxplot
    box_data = all_reg_predictions
    box_labels = [f'Model {i+1}' for i in range(len(all_reg_predictions))]
    
    bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.7)
    
    # Color the boxplots with the same colors as the lines
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style other boxplot elements
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax2.set_title('Regression Models - Statistical Distribution Analysis', 
                 fontsize=16, fontweight='bold')
    ax2.set_ylabel('Concentration Level Distribution', fontsize=12)
    ax2.set_ylim(-0.1, 2.1)
    
    # Set y-axis ticks and labels
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=10)
    
    # Add reference lines
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.tick_params(axis='x', labelsize=11)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    plt.suptitle(f'EEG Concentration Regression Models Analysis - {len(all_reg_predictions)} Models (Muse Headset)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_concentration_classification_plot(all_clf_predictions, clf_model_names, timestep=None):
    """Create a dedicated plot for concentration classification models"""
    
    if not all_clf_predictions:
        return
    
    # Prepare x-axis
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        x_axis = np.arange(len(all_clf_predictions[0]))
        x_label = 'EEG Sample Index (chronological order)'
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Downsample if needed
    if len(x_axis) > 500:
        step = len(x_axis) // 300
        x_axis_sampled = x_axis[::step]
    else:
        step = 1
        x_axis_sampled = x_axis
    
    # Use distinct colors
    distinct_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
    colors = distinct_colors[:len(all_clf_predictions)]
    
    # TOP LEFT: Time series step plot
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        predictions_sampled = predictions[::step]
        short_name = os.path.basename(model_name).replace('.pkl', '')
        if len(short_name) > 15:
            short_name = short_name[:12] + "..."
        label = f"Model {i+1}: {short_name}"
        ax1.step(x_axis_sampled, predictions_sampled, where='mid', label=label, 
                color=colors[i % len(colors)], linewidth=3, alpha=0.9)
        ax1.fill_between(x_axis_sampled, predictions_sampled, alpha=0.3, 
                        color=colors[i % len(colors)], step='mid')
    
    ax1.set_title('EEG Concentration Classification Models - Discrete Class Predictions Over Time', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Concentration Level (Discrete Classes)', fontsize=11)
    ax1.set_ylim(-0.1, 2.1)
    
    # Set y-axis ticks and labels
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=10)
    
    # Add reference lines
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_facecolor('#f8f9fa')
    
    # TOP RIGHT: Class frequency distribution (THIS IS THE BOTTOM LEFT PLOT FROM OVERVIEW)
    # Always show all concentration levels (0, 1, 2) even if frequency is 0
    all_classes = [0, 1, 2]  # Fixed concentration levels
    
    x_pos = np.arange(len(all_classes))
    width = 0.8 / len(all_clf_predictions)
    
    # Find max frequency for label positioning
    max_freq = 0
    for predictions in all_clf_predictions:
        for class_val in all_classes:
            freq = np.sum(predictions == class_val)
            max_freq = max(max_freq, freq)
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        frequencies = [np.sum(predictions == class_val) for class_val in all_classes]
        offset = (i - len(all_clf_predictions)/2 + 0.5) * width
        
        bars = ax2.bar(x_pos + offset, frequencies, width, 
                      label=f'Model {i+1}', color=colors[i % len(colors)], alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # Add frequency labels on bars (show all, including 0)
        for bar, freq in zip(bars, frequencies):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_freq*0.01,
                    f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Predicted Concentration Classes', fontsize=11)
    ax2.set_ylabel('Frequency Count (Number of Predictions)', fontsize=11)
    ax2.set_title('Classification Models - Class Distribution (Frequency Bars)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    
    # Create concentration level labels (always show all 3)
    class_labels = ["Relaxed", "Neutral", "Concentrated"]
    ax2.set_xticklabels(class_labels, fontsize=10)
    if len(all_clf_predictions) > 1:
        ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    # BOTTOM LEFT: Statistical distribution boxplot (THIS IS THE BOTTOM RIGHT PLOT FROM OVERVIEW)
    box_data = all_clf_predictions
    box_labels = [f'Model {i+1}' for i in range(len(all_clf_predictions))]
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.7)
    
    # Color the boxplots
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style other boxplot elements
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax3.set_title('Concentration Classification Models - Distribution Comparison (Boxplot)', 
                 fontsize=14, fontweight='bold')
    ax3.set_ylabel('Concentration Level Distribution', fontsize=11)
    ax3.set_ylim(-0.1, 2.1)
    
    # Set y-axis ticks and labels
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Relaxed (0)', 'Neutral (1)', 'Concentrated (2)'], fontsize=10)
    
    # Add reference lines
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=1, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    ax3.tick_params(axis='x', labelsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#fafafa')
    
    # BOTTOM RIGHT: Class proportion pie chart
    if len(all_clf_predictions) == 1:
        # Single model pie chart
        predictions = all_clf_predictions[0]
        class_counts = {}
        for class_val in all_classes:
            count = np.sum(predictions == class_val)
            if count > 0:
                class_counts[class_val] = count
        
        if class_counts:
            labels = []
            for c in class_counts.keys():
                if c == 0:
                    labels.append("Relaxed (0)")
                elif c == 1:
                    labels.append("Neutral (1)")
                elif c == 2:
                    labels.append("Concentrated (2)")
                else:
                    labels.append(f"Level {c}")
            
            sizes = list(class_counts.values())
            
            # Use concentration-specific colors
            pie_colors = []
            for c in class_counts.keys():
                if c == 0:
                    pie_colors.append('#2ECC71')  # Green for relaxed
                elif c == 1:
                    pie_colors.append('#F39C12')  # Orange for neutral
                elif c == 2:
                    pie_colors.append('#E74C3C')  # Red for concentrated
                else:
                    pie_colors.append(colors[0])
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=pie_colors, 
                                             autopct='%1.1f%%', startangle=90, 
                                             textprops={'fontsize': 11})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        
        ax4.set_title('Classification Model - Overall Class Distribution', 
                     fontsize=14, fontweight='bold')
    else:
        # Multiple models - show comparison heatmap or grouped bars
        ax4.text(0.5, 0.5, f'Multiple Models\n({len(all_clf_predictions)} total)\nSee frequency bars above', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.set_title('Multiple Classification Models Summary', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'EEG Concentration Classification Models Analysis - {len(all_clf_predictions)} Models (Muse Headset)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Mendeley-specific visualization functions

def create_mendeley_overview_plot(all_reg_predictions, all_clf_predictions, reg_model_names, clf_model_names, timestep=None):
    """Create the main Mendeley overview plot similar to concentration but with Mendeley-specific scales"""
    
    if not all_reg_predictions and not all_clf_predictions:
        print("No Mendeley models to visualize")
        return
    
    # Create the 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Prepare x-axis with better labeling
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        # Use the longest prediction array for x-axis
        if all_reg_predictions and all_clf_predictions:
            max_len = max(len(all_reg_predictions[0]), len(all_clf_predictions[0]))
        elif all_reg_predictions:
            max_len = len(all_reg_predictions[0])
        else:
            max_len = len(all_clf_predictions[0])
        x_axis = np.arange(max_len)
        x_label = 'Time (seconds from EEG recording start)'
    
    # Downsample if needed
    if len(x_axis) > 500:
        step = len(x_axis) // 250
        x_axis_sampled = x_axis[::step]
    else:
        step = 1
        x_axis_sampled = x_axis
    
    # TOP LEFT: Plot regression models
    if all_reg_predictions:
        # Use distinct colors
        distinct_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        colors = distinct_colors[:len(all_reg_predictions)]
        
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            predictions_sampled = predictions[::step]
            short_name = os.path.basename(model_name).replace('.pkl', '')
            if len(short_name) > 12:
                short_name = short_name[:9] + "..."
            label = f"R{i+1}: {short_name}"
            ax1.plot(x_axis_sampled, predictions_sampled, label=label, color=colors[i % len(colors)], 
                    linewidth=3, alpha=0.9)
        
        ax1.set_title('EEG Mendeley Regression Models - Temporal Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=10)
        ax1.set_ylabel('Mendeley Activity Type', fontsize=10)
        ax1.set_ylim(0.9, 3.1)  # Fixed scale for Mendeley
        
        # Set y-axis ticks and labels for Mendeley types
        ax1.set_yticks([1, 2, 3])
        ax1.set_yticklabels(['Social Media (1)', 'Cognitive Test (2)', 'Combined (3)'], fontsize=9)
        
        # Add reference lines for Mendeley types
        ax1.axhline(y=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axhline(y=3, color='purple', linestyle='--', alpha=0.5, linewidth=1)
        
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax1.grid(True, alpha=0.4, linestyle='--')
        ax1.set_facecolor('#f8f9fa')
    else:
        ax1.text(0.5, 0.5, 'No Regression Models', ha='center', va='center', transform=ax1.transAxes, fontsize=16)
        ax1.set_title('EEG Mendeley Regression Models - Temporal Predictions', fontsize=14, fontweight='bold')
    
    # TOP RIGHT: Plot classification models
    if all_clf_predictions:
        # Use same x-axis setup
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            predictions_sampled = predictions[::step]
            short_name = os.path.basename(model_name).replace('.pkl', '')
            if len(short_name) > 12:
                short_name = short_name[:9] + "..."
            label = f"C{i+1}: {short_name}"
            ax2.step(x_axis_sampled, predictions_sampled, where='mid', label=label, color='#E74C3C', 
                    linewidth=3, alpha=0.9)
        
        ax2.set_title('EEG Mendeley Classification Models - Temporal Class Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=10)
        ax2.set_ylabel('Mendeley Activity Type', fontsize=10)
        ax2.set_ylim(0.9, 3.1)  # Fixed scale for Mendeley
        
        # Set y-axis ticks and labels for Mendeley types
        ax2.set_yticks([1, 2, 3])
        ax2.set_yticklabels(['Social Media (1)', 'Cognitive Test (2)', 'Combined (3)'], fontsize=9)
        
        # Add reference lines for Mendeley types
        ax2.axhline(y=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=3, color='purple', linestyle='--', alpha=0.5, linewidth=1)
        
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.set_facecolor('#f8f9fa')
    else:
        ax2.text(0.5, 0.5, 'No Classification Models', ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        ax2.set_title('EEG Mendeley Classification Models - Temporal Class Predictions', fontsize=14, fontweight='bold')
    
    # BOTTOM LEFT: Classification frequency bar chart
    if all_clf_predictions:
        # Always show all Mendeley types (1, 2, 3) even if frequency is 0
        all_classes = [1, 2, 3]  # Fixed Mendeley types
        
        x_pos = np.arange(len(all_classes))
        width = 0.8 / len(all_clf_predictions)
        
        # Find max frequency for label positioning
        max_freq = 0
        for predictions in all_clf_predictions:
            for class_val in all_classes:
                freq = np.sum(predictions == class_val)
                max_freq = max(max_freq, freq)
        
        for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
            frequencies = [np.sum(predictions == class_val) for class_val in all_classes]
            offset = (i - len(all_clf_predictions)/2 + 0.5) * width
            
            short_name = os.path.basename(model_name).replace('.pkl', '')
            if len(short_name) > 8:
                short_name = short_name[:5] + "..."
            
            bars = ax3.bar(x_pos + offset, frequencies, width, 
                          label=f'C{i+1}: {short_name}', alpha=0.8,
                          edgecolor='black', linewidth=1)
            
            # Add frequency labels on bars
            for bar, freq in zip(bars, frequencies):
                if freq > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_freq*0.01,
                            f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Mendeley Activity Type Categories', fontsize=11)
        ax3.set_ylabel('Frequency Count (Number of Predictions)', fontsize=11)
        ax3.set_title('EEG Mendeley Classification Models - Activity Type Frequency Distribution', 
                     fontweight='bold', fontsize=14)
        ax3.set_xticks(x_pos)
        
        # Create Mendeley type labels
        class_labels = ["Social Media (1)", "Cognitive Test (2)", "Combined (3)"]
        ax3.set_xticklabels(class_labels, fontsize=10)
        ax3.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax3.grid(True, alpha=0.4, linestyle='--')
        ax3.set_facecolor('#fafafa')
    else:
        ax3.text(0.5, 0.5, 'No Classification Models', ha='center', va='center', transform=ax3.transAxes, fontsize=16)
        ax3.set_title('EEG Mendeley Classification - Activity Type Frequency', fontsize=14, fontweight='bold')
    
    # BOTTOM RIGHT: Summary statistics or model comparison
    if len(all_clf_predictions) == 1:
        # Single model - show class distribution pie chart
        predictions = all_clf_predictions[0]
        class_counts = {}
        for class_val in [1, 2, 3]:
            count = np.sum(predictions == class_val)
            if count > 0:
                class_counts[class_val] = count
        
        if class_counts:
            labels = []
            for c in class_counts.keys():
                if c == 1:
                    labels.append("Social Media (1)")
                elif c == 2:
                    labels.append("Cognitive Test (2)")
                elif c == 3:
                    labels.append("Combined (3)")
            
            sizes = list(class_counts.values())
            colors = ['#3498DB', '#2ECC71', '#9B59B6']  # Blue, Green, Purple
            
            ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Mendeley Activity Type Distribution', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No Valid Classifications', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
    else:
        # Multiple models - show comparison summary
        ax4.text(0.5, 0.5, f'Multiple Mendeley Models\n({len(all_clf_predictions)} total)\nSee frequency bars above', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14, fontweight='bold')
        ax4.set_title('Multiple Mendeley Models Summary', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'EEG Mendeley Activity Type Models Analysis - {len(all_clf_predictions)} Models (Muse Headset)', 
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def create_mendeley_regression_plot(all_reg_predictions, reg_model_names, timestep=None):
    """Create improved regression plots with Mendeley-specific scaling and visualization"""
    
    if not all_reg_predictions:
        return
    
    # Set fixed scale for Mendeley models (1=Social Media, 2=Cognitive Test, 3=Combined)
    y_lim = (0.9, 3.1)  # Fixed scale with small margin
    
    # Prepare x-axis with more descriptive labeling
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        x_axis = np.arange(len(all_reg_predictions[0]))
        x_label = 'EEG Sample Index (chronological order)'
    
    # Create figure with multiple subplots
    n_models = len(all_reg_predictions)
    if n_models == 1:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = [axes]
    else:
        # Create a grid layout: time series on top, boxplot below
        fig = plt.figure(figsize=(20, 10))
        
        # Top row: Individual time series plots (same scale)
        gs = fig.add_gridspec(2, n_models, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Use distinct colors for better differentiation
        distinct_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#E67E22', '#E74C3C', '#1ABC9C', '#F39C12', '#34495E']
        colors = distinct_colors[:n_models] if n_models <= len(distinct_colors) else plt.cm.tab10(np.linspace(0, 1, n_models))
        
        # Individual time series plots with same scale
        for i, (predictions, model_name) in enumerate(zip(all_reg_predictions, reg_model_names)):
            ax = fig.add_subplot(gs[0, i])
            
            # Downsample for cleaner visualization if needed
            if len(x_axis) > 1000:
                step = len(x_axis) // 500
                x_sampled = x_axis[::step]
                pred_sampled = predictions[::step]
            else:
                x_sampled = x_axis
                pred_sampled = predictions
            
            ax.plot(x_sampled, pred_sampled, linewidth=3, alpha=0.9, 
                   color=colors[i])
            
            # Create more descriptive, cleaner titles for Mendeley models
            short_name = os.path.basename(model_name).replace(".pkl", "")
            model_type = "Social Media Analysis"
            
            # Truncate very long names and add line breaks if needed
            if len(short_name) > 15:
                display_name = short_name[:12] + "..."
            else:
                display_name = short_name.replace("_", " ")
            
            ax.set_title(f'{model_type} Model {i+1}\n{display_name}', fontweight='bold', fontsize=10)
            
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel('Activity Type', fontsize=9)
            ax.set_ylim(y_lim)  # Fixed scale for Mendeley
            
            # Set y-axis ticks and labels for Mendeley types
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(['Social Media (1)', 'Cognitive Test (2)', 'Combined (3)'], fontsize=8)
            
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.set_facecolor('#f8f9fa')
            
            # Add horizontal reference lines for Mendeley types
            ax.axhline(y=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1) 
            ax.axhline(y=3, color='purple', linestyle='--', alpha=0.5, linewidth=1)
        
        # Bottom: Boxplot comparison
        ax_box = fig.add_subplot(gs[1, :])
        box_data = all_reg_predictions
        # Create shorter labels for boxplots
        box_labels = [f'M{i+1}' for i in range(n_models)]
        
        bp = ax_box.boxplot(box_data, labels=box_labels, patch_artist=True, 
                           showmeans=True, meanline=True)
        
        # Use more distinct colors for better differentiation
        colors = distinct_colors[:n_models] if n_models <= len(distinct_colors) else plt.cm.tab10(np.linspace(0, 1, n_models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        ax_box.set_title('EEG Mendeley Models - Statistical Distribution Analysis', fontweight='bold', fontsize=12)
        ax_box.set_ylabel('Activity Type', fontsize=10)
        ax_box.set_ylim(y_lim)  # Fixed scale for Mendeley
        
        # Set y-axis ticks and labels for Mendeley types
        ax_box.set_yticks([1, 2, 3])
        ax_box.set_yticklabels(['Social Media (1)', 'Cognitive Test (2)', 'Combined (3)'], fontsize=9)
        
        ax_box.tick_params(axis='x', labelsize=9)
        ax_box.grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        ax_box.axhline(y=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax_box.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax_box.axhline(y=3, color='purple', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.suptitle(f'EEG-Based Mendeley Activity Models Analysis - {n_models} Models (Muse Headset Data)', 
                    fontsize=18, fontweight='bold', y=0.95)
    
    # Better layout adjustment
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_mendeley_classification_plot(all_clf_predictions, clf_model_names, timestep=None):
    """Create separate, cleaner classification plots for Mendeley models"""
    
    if not all_clf_predictions:
        return
    
    n_models = len(all_clf_predictions)
    
    # Prepare x-axis with descriptive labels
    if timestep is not None and len(timestep) > 0:
        x_axis = timestep
        x_label = 'Time (seconds from EEG recording start)'
    else:
        x_axis = np.arange(len(all_clf_predictions[0]))
        x_label = 'EEG Sample Index (chronological order)'
    
    # Use distinct colors for better differentiation
    distinct_colors = ['#3498DB', '#2ECC71', '#9B59B6', '#E67E22', '#E74C3C', '#1ABC9C', '#F39C12', '#34495E']
    colors = distinct_colors[:n_models] if n_models <= len(distinct_colors) else plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Create 3 separate plots for better clarity
    
    # Plot 1: Individual time series plots (one per model)
    create_individual_mendeley_classification_plots(all_clf_predictions, clf_model_names, x_axis, x_label, colors)
    
    # Plot 2: Comparison plots (boxplot and frequency)
    create_mendeley_classification_comparison_plots(all_clf_predictions, clf_model_names, colors)
    
    # Plot 3: Class distribution analysis
    create_mendeley_classification_distribution_plots(all_clf_predictions, clf_model_names, colors)

def create_individual_mendeley_classification_plots(all_clf_predictions, clf_model_names, x_axis, x_label, colors):
    """Create individual time series plots for each Mendeley classification model"""
    
    n_models = len(all_clf_predictions)
    
    # Determine grid layout
    if n_models <= 2:
        rows, cols = 1, n_models
        fig_size = (12 * n_models, 6)
    elif n_models <= 4:
        rows, cols = 2, 2
        fig_size = (16, 10)
    else:
        rows = (n_models + 2) // 3
        cols = 3
        fig_size = (18, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=fig_size)
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__iter__') else [axes]
    else:
        axes = axes.flatten()
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        ax = axes[i]
        
        # Downsample if needed for cleaner visualization
        if len(x_axis) > 800:
            step = len(x_axis) // 400
            x_sampled = x_axis[::step]
            pred_sampled = predictions[::step]
        else:
            x_sampled = x_axis
            pred_sampled = predictions
        
        # Create clean step plot for better class visibility
        ax.step(x_sampled, pred_sampled, where='mid', color=colors[i], 
               linewidth=2.5, alpha=0.9, label=f'Model {i+1}')
        
        # Fill between steps for better visibility
        ax.fill_between(x_sampled, pred_sampled, alpha=0.3, color=colors[i], step='mid')
        
        # Create descriptive model names for Mendeley
        short_name = os.path.basename(model_name).replace(".pkl", "")
        model_type = "Social Media Analysis"
        class_description = "Activity Types"
            
        if len(short_name) > 20:
            display_name = short_name[:17] + "..."
        else:
            display_name = short_name.replace("_", " ")
        
        ax.set_title(f'{model_type} Classification Model {i+1}\n{display_name}', 
                    fontweight='bold', fontsize=12, pad=15)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel('Activity Type', fontsize=10)
        
        # Set fixed y-axis for Mendeley types
        ax.set_ylim(0.9, 3.1)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Social Media (1)', 'Cognitive Test (2)', 'Combined (3)'], fontsize=9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Add horizontal reference lines for Mendeley types
        ax.axhline(y=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=3, color='purple', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add Mendeley type info
        unique_classes = np.unique(predictions)
        class_text = f'Activity Types: {", ".join(map(str, sorted(unique_classes)))} (1=Social Media, 2=Cognitive Test, 3=Combined)'
            
        ax.text(0.02, 0.98, class_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'Individual EEG Mendeley Activity Classification Model Time Series - {n_models} Models (Muse Headset AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_mendeley_classification_comparison_plots(all_clf_predictions, clf_model_names, colors):
    """Create comparison plots for Mendeley classification models"""
    
    n_models = len(all_clf_predictions)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Enhanced boxplot
    box_data = all_clf_predictions
    box_labels = [f'Model {i+1}' for i in range(n_models)]
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    showmeans=True, meanline=True, widths=0.6)
    
    # Apply distinct colors to boxplots
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    # Style other boxplot elements
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax1.set_title('EEG Mendeley Activity Classification Models - Statistical Distribution Analysis', 
                 fontweight='bold', fontsize=14)
    ax1.set_ylabel('Activity Type', fontsize=11)
    
    # Set fixed y-axis for Mendeley types
    ax1.set_ylim(0.9, 3.1)
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Social Media (1)', 'Cognitive Test (2)', 'Combined (3)'], fontsize=9)
    
    ax1.tick_params(axis='x', labelsize=10)
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.set_facecolor('#fafafa')
    
    # Add horizontal reference lines
    ax1.axhline(y=1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=3, color='purple', linestyle='--', alpha=0.5, linewidth=1)
    
    # Right: Enhanced class frequency comparison
    # Always show all Mendeley types (1, 2, 3) even if frequency is 0
    all_classes = [1, 2, 3]  # Fixed Mendeley types
    
    x_pos = np.arange(len(all_classes))
    width = 0.8 / n_models
    
    # Find max frequency for label positioning
    max_freq = 0
    for predictions in all_clf_predictions:
        for class_val in all_classes:
            freq = np.sum(predictions == class_val)
            max_freq = max(max_freq, freq)
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        frequencies = [np.sum(predictions == class_val) for class_val in all_classes]
        offset = (i - n_models/2 + 0.5) * width
        
        model_label = f'Mendeley Model {i+1}'
            
        bars = ax2.bar(x_pos + offset, frequencies, width, 
                      label=model_label, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # Add frequency labels on bars (show all, including 0)
        for bar, freq in zip(bars, frequencies):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_freq*0.01,
                    f'{freq}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Activity Type Categories', fontsize=11)
    ax2.set_ylabel('Frequency Count (Number of Predictions)', fontsize=11)
    ax2.set_title('EEG Mendeley Activity Classification Models - Class Frequency Distribution', 
                 fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos)
    
    # Create Mendeley type labels (always show all 3)
    class_labels = ["Social Media (1)", "Cognitive Test (2)", "Combined (3)"]
    ax2.set_xticklabels(class_labels, fontsize=10)
    ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.set_facecolor('#fafafa')
    
    plt.suptitle('EEG-Based Mendeley Activity Classification Models - Statistical Comparison (Muse Headset Data)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.show()

def create_mendeley_classification_distribution_plots(all_clf_predictions, clf_model_names, colors):
    """Create distribution analysis plots for Mendeley classification models"""
    
    n_models = len(all_clf_predictions)
    
    # Get all unique classes across models
    all_classes = set()
    for predictions in all_clf_predictions:
        all_classes.update(predictions)
    all_classes = sorted(list(all_classes))
    
    # Create pie charts for each model
    if n_models <= 3:
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    else:
        rows = (n_models + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(18, 6 * rows))
        axes = axes.flatten()
    
    if n_models == 1:
        axes = [axes]
    
    for i, (predictions, model_name) in enumerate(zip(all_clf_predictions, clf_model_names)):
        ax = axes[i]
        
        # Calculate class frequencies
        class_counts = {}
        for class_val in all_classes:
            count = np.sum(predictions == class_val)
            if count > 0:
                class_counts[class_val] = count
        
        if class_counts:
            # Create Mendeley type labels
            labels = []
            for c in class_counts.keys():
                if c == 1:
                    labels.append("Social Media (1)")
                elif c == 2:
                    labels.append("Cognitive Test (2)")
                elif c == 3:
                    labels.append("Combined (3)")
                else:
                    labels.append(f"Type {c}")
            
            sizes = list(class_counts.values())
            
            # Use distinct colors - blue for social media, green for cognitive, purple for combined
            pie_colors = []
            for c in class_counts.keys():
                if c == 1:
                    pie_colors.append('#3498DB')  # Blue for social media
                elif c == 2:
                    pie_colors.append('#2ECC71')  # Green for cognitive test
                elif c == 3:
                    pie_colors.append('#9B59B6')  # Purple for combined
                else:
                    pie_colors.append(colors[i % len(colors)])
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', 
                                             startangle=90, textprops={'fontsize': 10})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        
        # Create descriptive model titles
        short_name = os.path.basename(model_name).replace(".pkl", "")
        model_type = "Social Media Analysis"
            
        if len(short_name) > 15:
            display_name = short_name[:12] + "..."
        else:
            display_name = short_name.replace("_", " ")
        
        ax.set_title(f'{model_type} Model {i+1}\n{display_name}\nActivity Distribution (%)', 
                    fontweight='bold', fontsize=11)
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('EEG Mendeley Activity Classification Models - Type Distribution Analysis (Muse EEG AF7,TP9,TP10,AF8)', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
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
    
    return saved_files

if __name__ == "__main__":
    # Maintain original behavior for backward compatibility
    predictor = EEGModelPredictor()
    model_summary = predictor.load_models()
    data_summary = predictor.load_data()
    predictor.predict()
    predictor.visualize_predictions(show_plots=True, save_plots=False)
