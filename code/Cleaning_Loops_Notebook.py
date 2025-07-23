#!/usr/bin/env python
# coding: utf-8


import mne
import os
import pandas as pd
import json
import glob
# In[1]:


from EEG_cleaning import muse_clean



# The `muse_clean()` function requests 4 inputs:
# - `filepath`: The path to the folder containing the dataset in question (relative *or* absolute)
# - `filename`: The name of the file to clean (include file type extension)
# - `subject_id`: The unique custom identification of the subject for the output file name
# - `record_id`: The unique custom identification of the recorded label for the output file name
# - `new_folder` (*optional*): The name of the outputted folder within the cleaned datasets directory containing all of the cleaned datasets (default to be named "*cleaned datasets* as well")

# Single File Cleaning

# In[2]:

def single_file_cleaning(filename,subject_id,record_id):
    """
    This function is used to clean a single file, and is useful for testing the cleaning function.
    """
    filepath = 'raw datasets/Local Recordings'
  

    muse_clean(filepath, filename, subject_id, record_id, new_folder='local datasets')

# Mendeley Cleaning Loop

# In[3]:


import os
def mendeley_cleaning_loop():
    """
    This function cleans all files in the Mendeley dataset.
    """
    # Define the folder list for Mendeley dataset
    folder_list = ['raw datasets/Mendeley/' + folder for folder in os.listdir('raw datasets/Mendeley/')]

    for patient in range(1, 6):

        current_folder = folder_list[patient - 1]
        file_list = os.listdir(current_folder)

        for experiment in range(1, 4):

            current_file = file_list[experiment - 1]
            muse_clean(current_folder, current_file, patient, experiment, 'Mendeley cleaned')
            print(f"Patient {patient} Experiment {experiment} complete...")

    print("All done!")


# Emotion Cleaning Loop

# In[2]:

def emotion_cleaning_loop():
    """
    This function cleans all files in the Emotion dataset.
    """
    # Define the folder list for Emotion dataset            


    folder_list = [r'../raw datasets/EmoKey Short/muse_wearable_data/' + folder for folder in os.listdir('../raw datasets/EmoKey Short/muse_wearable_data/')]

    for patient in range(1, 45):

        current_folder = folder_list[patient - 1]
        file_list = os.listdir(current_folder)

        for experiment in range(1, 5):
        # print(current_folder)
        # print(file_list)
            current_file = file_list[experiment - 1]
            muse_clean(current_folder, current_file, patient, experiment, 'Emotion cleaned')
            print(f"Patient {patient} Experiment {experiment} complete...")


# General Cleaning Loop

# In[ ]:


def cleaning_loop(folder_list: list, new_folder: str = 'cleaned datasets') -> None:

    """
    Inputs: A list of folder directories containing data, and a list of file names to be cleaned
    Outputs: A new folder (customizable name) that contains the cleaned versions of all specified files in all specified folders

    This code functions assuming it follows the following file naming scheme:
    ```subject{name}-{label}-{label_number}.csv```
    """

    import os

    file_exists = True

    for folder in folder_list:

        try:
            cwd = os.getcwd()
            os.chdir(folder)
            file_list = os.listdir()
        except FileNotFoundError:
            print(f"Folder {folder} not found, try again!")
            break

        for file in file_list:
            subject_id = file.split('-')[0][-1]
            label_id = file.split(('-'))[1] + file.split(('-'))[2]

            try:
                muse_clean(folder, file, subject_id, label_id, new_folder)
                print(f"Patient {subject_id} Experiment {label_id} complete...")
                print("All done!")
            except FileNotFoundError:
                print(f"File {file} does not exist, try again!")
                file_exists = False
                break

        if not file_exists:
            break
                
                

def neuro_sense_cleaning_loop():
    """
    This function cleans all files in the NeuroSense dataset.
    """
    # Define the folder list for NeuroSense dataset
    directory = "../raw datasets/NeuroSense/"
    subject_list = os.listdir(directory)
    subject_list = [sub for sub in subject_list if sub.startswith("sub")]

    for subject in subject_list:
        session_list = os.listdir(os.path.join(directory, subject))
        for session in session_list:
            # Load EDF file
            edf_pattern = os.path.join(directory, subject, session, "eeg", "*.edf")
            edf_files = glob.glob(edf_pattern)
            raw = mne.io.read_raw_edf(edf_files[0], preload=True)

            # Pick only the desired channels
            desired_channels = ['TP9', 'AF7', 'AF8', 'TP10']
            raw.pick_channels(desired_channels)

            # Get data and timestamps


            # Get data and timestamps
            data, times = raw.get_data(return_times=True)  # data shape: (n_channels, n_times)

            # Create a DataFrame
            df = pd.DataFrame(data.T, columns=desired_channels)  # Transpose to shape (n_times, n_channels)
            df.insert(0, "timestamp", times)  # Add timestamp column at the beginning


            # Open and read the JSON file
            json_pattern = os.path.join(directory, subject, session, "eeg", "*.json")
            json_files = glob.glob(json_pattern)
            with open(json_files[0], 'r') as file:
                json_data = json.load(file)

            print(json_data)

            # Make the information about the music being listened to into a dictionary

            # Step 1: Split into key-value pairs
            pairs = json_data["TaskDescription"].split(', ')

            # Step 2: Convert to dictionary
            task_data = {}
            for pair in pairs:
                key, value = pair.split(': ', 1)
                # Attempt to cast to float or int
                if value.replace('.', '', 1).isdigit():
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                task_data[key] = value

            # Use the data
            general_tag = task_data['Lastfm_tag']
            valence = task_data['AVG_Valence']
            arousal = task_data['AVG_Arousal']

            df["Valence"] = valence
            df["Arousal"] = arousal

            print("General Tag:", general_tag)
            print("Valence:", valence)
            print("Arousal:", arousal)

            # Save to CSV in the right folder
            if valence >= 5:
                if arousal >= 5:
                    quadrant = "Happy"
                else:
                    quadrant = "Relaxed"
            else:
                if arousal >= 5:
                    quadrant = "Angry"
                else:
                    quadrant = "Sad"

            savename = f"{subject}-{session}-{general_tag}-{quadrant}-cleaned"
            df.to_csv(f"cleaned datasets/NeuroSense/{savename}.csv", index=False)

            print("CSV file saved as eeg_data.csv")

