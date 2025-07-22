from Cleaning_Loops_Notebook import single_file_cleaning
from Feature_Extraction import gen_training_matrix_mendeley, gen_training_matrix_neurosense, gen_training_matrix_ogdata, gen_training_matrix_ourdata
import Use_Trained_Model
import datetime



# Execute the cleaning loop on our data

filename = input("Enter the filename to clean: ")
subject_id = input("Enter the subject ID: ")
record_id = input("Enter the record ID (which procedure): ")
single_file_cleaning(filename, subject_id, record_id)




#Execute feature extraction on our data
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

target_dir = "local datasets" 
directory_path = "../cleaned datasets/" + target_dir

output_file = f"../featuresets/{target_dir}_{date}.csv"
    
gen_training_matrix_ourdata(directory_path, output_file, cols_to_ignore = -1)



    