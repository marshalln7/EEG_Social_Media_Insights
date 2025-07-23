from Cleaning_Loops_Notebook import single_file_cleaning
from Feature_Extraction import gen_training_matrix_ourdata
from Use_Trained_Model import EEGModelPredictor
import datetime



# Execute the cleaning loop on our data

filename = input("Enter the filename to clean: ")
subject_id = input("Enter the subject ID: ")
record_id = input("Enter the record ID (which procedure): ")
single_file_cleaning(filename, subject_id, record_id)




#Execute feature extraction on our data
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

target_dir = "local datasets" 
directory_path = "cleaned datasets/" + target_dir

output_file = f"featuresets/{target_dir}_{date}.csv"
    
gen_training_matrix_ourdata(directory_path, output_file, cols_to_ignore = -1)



"""
EEG Model Verification Script
Run with: python3 verify_trained_model.py
"""

print("Starting EEG Model Prediction...")

# Create predictor
predictor = EEGModelPredictor(
    models_dir="models",
    featuresets_dir="local featuresets",
    model_types=['concentration']
)

# Load and predict
print(" Loading models...")
model_info = predictor.load_models()
print(f" Loaded {model_info['total_loaded']} models")

print("Loading dataset...")
# Specify the exact dataset file you want to use
data_info = predictor.load_data(data_file= output_file)
print(f" Loaded data: {data_info['data_shape']} from {data_info['file_used']}")

print(" Making predictions...")
predictions = predictor.predict()

print(" Generating visualization graphs...")
predictor.visualize_predictions() 