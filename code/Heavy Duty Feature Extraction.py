import os, sys
import numpy as np
import datetime
import warnings
from EEG_feature_extraction import generate_feature_vectors_from_samples

warnings.filterwarnings(
    "ignore",
    message="logm result may be inaccurate*",
    category=RuntimeWarning
)

#This is the generating matrix code for the original github data
def gen_training_matrix_og(directory_path, output_file, cols_to_ignore):
	# Initialise return matrix
	FINAL_MATRIX = None
	
	for x in os.listdir(directory_path):

		# Ignore non-CSV files
		if not x.lower().endswith('.csv'):
			continue
		
		# For safety we'll ignore files containing the substring "test". 
		# [Test files should not be in the dataset directory in the first place]
		if 'test' in x.lower():
			continue
		try:
			name, state, _ = x[:-4].split('-')
		except:
			print ('Wrong file name', x)
			sys.exit(-1)
		if state.lower() == 'concentrating':
			state = 2.
		elif state.lower() == 'neutral':
			state = 1.
		elif state.lower() == 'relaxed':
			state = 0.
		else:
			print ('Wrong file name', x)
			sys.exit(-1)
			
		print ('Using file', x)
		full_file_path = directory_path  +   '/'   + x
		vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
														        nsamples = 150, 
																period = 1.,
																state = state,
														        remove_redundant = True,
																cols_to_ignore = cols_to_ignore)
		
		print('resulting vector shape for the file', vectors.shape)
		
		
		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

	print('FINAL_MATRIX', FINAL_MATRIX.shape)
	
	# Shuffle rows
	# np.random.shuffle(FINAL_MATRIX)
	
	# Save to file
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header), 
			comments = '')

	return None


# This is the generating matrix code for the Emotion data
def gen_training_matrix_emotion(directory_path, output_file, cols_to_ignore):
	# Initialise return matrix
	FINAL_MATRIX = None
	
	for x in os.listdir(directory_path):

		# Ignore non-CSV files
		if not x.lower().endswith('.csv'):
			continue
		
		# For safety we'll ignore files containing the substring "test". 
		# [Test files should not be in the dataset directory in the first place]
		if 'test' in x.lower():
			continue
		try:
			name, _ ,state, _ = x[:-4].split('-')
		except:
			print ('Wrong file name', x)
			sys.exit(-1)
		if state.lower() == '1': #Anger
			state = 1.
		elif state.lower() == '2': #Fear
			state = 2.
		elif state.lower() == '3': #Happiness
			state = 3.
		elif state.lower() == '4': #Sadness
			state = 4.
		else:
			print ('Wrong file name', x)
			sys.exit(-1)
			
		print ('Using file', x)
		full_file_path = directory_path  +   '/'   + x
		vectors, header = generate_feature_vectors_from_samples(file_path = full_file_path, 
														        nsamples = 150, 
																period = 1.,
																state = state,
														        remove_redundant = True,
																cols_to_ignore = cols_to_ignore)
		
		print ('resulting vector shape for the file', vectors.shape)
		
		
		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack( [ FINAL_MATRIX, vectors ] )

	print ('FINAL_MATRIX', FINAL_MATRIX.shape)
	
	# Shuffle rows
	# np.random.shuffle(FINAL_MATRIX)
	
	# Save to file
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header), 
			comments = '')

	return None


if __name__ == '__main__':


	date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	target_dir = "Emotion cleaned" # Change this one to change which data we're processing
	directory_path = "cleaned datasets/" + target_dir

	output_file = f"featuresets/{target_dir}_{date}.csv"
    
    # Change the function depending on which dataset we're using
    # Change the cols_to_ignore list if we're making features from raw data or aggregate data
	gen_training_matrix_emotion(directory_path, output_file, cols_to_ignore = [5,6,7,8,9])