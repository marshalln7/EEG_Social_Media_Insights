import mne
import os
import pandas as pd
import json
import glob

directory = "/yunity/msn39/EEG_Social_Media_Insights/raw datasets/NeuroSense/"
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