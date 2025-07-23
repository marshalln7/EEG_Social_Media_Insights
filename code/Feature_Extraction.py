#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys
import numpy as np
import datetime
import warnings
import re
import pandas as pd
from EEG_feature_extraction import generate_feature_vectors_from_samples


# In[3]:


# Gets rid of pesky warnings about the logm not bein accurate, could be worth going back and looking at these later
warnings.filterwarnings(
    "ignore",
    message="logm result may be inaccurate*",
    category=RuntimeWarning
)


# In[ ]:


# Generate the feature extraction for the NeuroSense data
def gen_training_matrix_neurosense(directory_path, output_file, cols_to_ignore):
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
			_,name,_,_,_,state, _ = x[:-4].split('-')
		except:
			print ('Wrong file name', x)
			sys.exit(-1)
		if state.lower() == 'happy':
			state = 3.
		elif state.lower() == 'relaxed':
			state = 2.
		elif state.lower() == 'sad':
			state = 1.
		elif state.lower() == 'angry':
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

		# jerry rig some new columns to add on to the featureset	
		valence_arousal = pd.read_csv(full_file_path)[["Valence", "Arousal"]].to_numpy()[:vectors.shape[0]]
		# print(valence_arousal)
		vectors = np.hstack([vectors, valence_arousal])
		header = np.hstack([header, "Valence", "Arousal"])

		print('resulting vector shape for the file', vectors.shape)
		
		
		if FINAL_MATRIX is None:
			FINAL_MATRIX = vectors
		else:
			FINAL_MATRIX = np.vstack([FINAL_MATRIX, vectors])

	print('FINAL_MATRIX', FINAL_MATRIX.shape)
	
	# Shuffle rows
	# np.random.shuffle(FINAL_MATRIX)
	
	# Save to file
	np.savetxt(output_file, FINAL_MATRIX, delimiter = ',',
			header = ','.join(header), 
			comments = '')

	return None


if __name__ == '__main__':
	
	
	# if len(sys.argv) < 3: # For use in calling the function from the command line
	# 	print ('arg1: input dir\narg2: output file')
	# 	sys.exit(-1)

	date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	target_dir = "NeuroSense" # Change this one to change which data we're processing
	directory_path = "../cleaned datasets/" + target_dir

	output_file = f"../featuresets/{target_dir}_{date}.csv"
    
	gen_training_matrix_neurosense(directory_path, output_file, cols_to_ignore = [5, 6])


# In[5]:


#This is the generating matrix code for the original github data
def gen_training_matrix_ogdata(directory_path, output_file, cols_to_ignore):
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


if __name__ == '__main__':
	
	
	# if len(sys.argv) < 3: # For use in calling the function from the command line
	# 	print ('arg1: input dir\narg2: output file')
	# 	sys.exit(-1)

	date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	target_dir = "original_data" # Change this one to change which data we're processing
	directory_path = "../cleaned datasets/" + target_dir

	output_file = f"../featuresets/{target_dir}_{date}.csv"
    
	gen_training_matrix_ogdata(directory_path, output_file, cols_to_ignore = -1)


# In[8]:


# This is the generating matrix code for Mendeley Data

def gen_training_matrix_mendeley(directory_path, output_file, cols_to_ignore):
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
		if state.lower() == '1':
			state = 1.
		elif state.lower() == '2':
			state = 2.
		elif state.lower() == '3':
			state = 3.
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
	
	
	# if len(sys.argv) < 3: # For use in calling the function from the command line
	# 	print ('arg1: input dir\narg2: output file')
	# 	sys.exit(-1)

	date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	target_dir = "Mendeley cleaned" # Change this one to change which data we're processing
	directory_path = "../cleaned datasets/" + target_dir

	output_file = f"../featuresets/{target_dir}_{date}.csv"
    
	gen_training_matrix_mendeley(directory_path, output_file, cols_to_ignore = [1,2,3,4])


# In[7]:


#This is the generating matrix code for our own data
def gen_training_matrix_ourdata(directory_path, output_file, cols_to_ignore):
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
			name, state, _, _ = x[:-4].split('-')
		except:
			print ('Wrong file name', x)
			sys.exit(-1)
		if state.lower() == 'Marshall':
			state = 2.
		elif state.lower() == 'label':
			state = 1.
		elif state.lower() == 'Kenzo':
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
	
	
	# if len(sys.argv) < 3: # For use in calling the function from the command line
	# 	print ('arg1: input dir\narg2: output file')
	# 	sys.exit(-1)

	date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	target_dir = "local datasets" # Change this one to change which data we're processing
	directory_path = "../cleaned datasets/" + target_dir

	output_file = f"../local featuresets/{target_dir}_{date}.csv"
    
	gen_training_matrix_ourdata(directory_path, output_file, cols_to_ignore = -1)


# In[6]:


# This is the generating matrix code for the Emotion data
def gen_training_matrix_emokey(directory_path, output_file, cols_to_ignore):
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
	
	
	# if len(sys.argv) < 3: # For use in calling the function from the command line
	# 	print ('arg1: input dir\narg2: output file')
	# 	sys.exit(-1)

	date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

	target_dir = "Emotion cleaned" # Change this one to change which data we're processing
	directory_path = "../cleaned datasets/" + target_dir

	output_file = f"../featuresets/{target_dir}_{date}.csv"
    
	gen_training_matrix_emokey(directory_path, output_file, cols_to_ignore = [1,2,3,4])

