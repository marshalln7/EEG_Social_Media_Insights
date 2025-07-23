# Code used in order to clean the datasets attached in the Mendeley dataset
## By Kenzo Hubert

# ----------------------------------------------------------------
# VOCABULARY:
# - Pauli Test: A personality test commonly given to employees to test their focus and learning rate, derived from the Kraepelin test
#   (https://www.algobash.com/en/types-of-psychological-tests/)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# KEY THINGS TO NOTE:
# - 5 subjects with 3 experimental conditions:
#    - rec_01: User watches posts that they like
#    - rec_02: User taking Pauli test without any distractions
#    - rec_03: User taking Pauli tests with autitory social media distractions
# - Each subject is exposed to all 3 experimental conditions
#
# - Headband columns that begin with HSI (Horse Shoe Indicator) contain data values of either 1, 2, or 4
#    - 1: Good readings from the specified sensor
#    - 2: Medium readings from the specified sensor
#    - 4: Bad readings from the specified sensor
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# CODING
""" 
print(df.head())
print(df.columns)
print(df.shape)

print(df.value_counts('Battery', dropna = False))
print(df.value_counts('HSI_TP9', dropna = False)) 
"""

## Creating time-based indices using the `TimeStamp` column



## Removing observations with no data, and observations no change compared to previous observations

### First step is to remove columns that get in the way of this process:
### The `RAW_{}` columns, `AUX_RIGHT` column, and `Element` columns all contain meaningless changing 
### values that inhibit the measuring of consistency between other relevant recorded values

### Remove all rows that do not fit the following criteria:
###    - The entries in this row are all different from the previous rows (row is unique)
###    - The majority of entries contain values, headband is on the head (row is meaningful)
###    - The entries are all taken with "good" readings from the sensors (row is accurate)


# -------------------------------------------------------------------------------------------------------------------------------
## Package imports
import os
import pandas as pd


## Create a function with the above code that takes in a file path, returning a "cleaned" dataset according to the above criteria
def muse_clean(filepath: str, filename: str, subject_id: str, record_id: str, new_folder: str = 'cleaned datasets') -> None:
    """
    Inputs: The subject number (1-5) and the record number (1-3)
    Output: A file output (.csv) that returns a "cleaned" version of the raw data file

    Through entering the requested information (subject number, record number), users are able to download a "cleaned" version of
    the dataset inputted according to the following criteria:

      - The entries in this row are all different from the previous rows (row is unique)
      - The majority of entries contain values, headband is on the head (row is meaningful)
      - The entries are all taken with "good" readings from the sensors (row is accurate) 

    raw - determines if in this particular cleaning we're using the raw measurements or
        the averaged measurements across each of the bands (Alpha, Beta, etc.) as calculated by the Muse headband   
    """
    
    ## Create a directory to insert newly cleaned files into, if not already created
    os.makedirs(f'../cleaned datasets/{new_folder}', exist_ok = True)

    ## Reading in dataset
    df = pd.read_csv(f'{filepath}/{filename}')

    ## Creating time-based indices using the `TimeStamp` column
    df['timestamp'] = pd.to_datetime(df['TimeStamp'], format='%Y-%m-%d %H:%M:%S.%f')
    df['timestamp'] = df['timestamp'].astype('int64') / 1e9 
    

    ## If it's an emotions dataset, chop out the emotionally significant moments
    if "Key_Moments" in df.columns:
        print("Processing as an Emotions dataset...")
        key_moment_timestamps = df[df["Key_Moments"] == 1].timestamp.tolist()
        print(f"Key moments found at timestamps: {key_moment_timestamps}")
        key_moment_timestamps = [df.timestamp[0]] + key_moment_timestamps # Adds the first timestamp as an emotional marker as well
        key_moment_intervals = [[timestamp, timestamp + 10] for timestamp in key_moment_timestamps] # 10 seconds after each key moment
        mask = pd.Series(False, index=df.index)
        # Gets the rows that are within the 10 second intervals
        for low, high in key_moment_intervals:
            mask |= (df["timestamp"] >= low) & (df["timestamp"] <= high) 
        df = df[mask]

    # Don't forget that this will be the index, and will be necessary for the feature extraction
    df.set_index('timestamp', inplace = True)

    ## Removing observations with no data, and observations no change compared to the previous observation

    ### Remove all rows that do not fit the following criteria:
    ###    - The entries in this row are all different from the previous row (row is unique)
    ###    - The majority of entries contain values, headband is on the head (row is meaningful)
    ###    - The entries are all taken with "good" readings from the sensors (row is accurate)
    drop_list = []
    previous_row = [i for i in df.iloc[0]]
    for i in range(1, df.shape[0]):
        current_row = [value for value in df.iloc[i]]
        if previous_row == current_row:
            drop_list.append(i)
        else:
            previous_row = current_row

    df.drop(drop_list, axis = 0, inplace = True)
    print(f"Removed {len(drop_list)} rows with no data or no change compared to previous rows.")
    print(drop_list)
    df = df[(df["HeadBandOn"] == 1) & (df["HSI_TP9"] == 1) & (df["HSI_AF7"] == 1) & (df["HSI_AF8"] == 1) & (df["HSI_TP10"] == 1)]

    ## Finalizing final column selection:
    df["Delta_Aggregate"] = df.filter(regex="^Delta_").mean(axis=1)
    df["Theta_Aggregate"] = df.filter(regex="^Theta_").mean(axis=1)
    df["Alpha_Aggregate"] = df.filter(regex="^Alpha_").mean(axis=1)
    df["Beta_Aggregate"] = df.filter(regex="^Beta_").mean(axis=1)
    df["Gamma_Aggregate"] = df.filter(regex="^Gamma_").mean(axis=1)

    agg_columns = ["Delta_Aggregate", "Theta_Aggregate", "Alpha_Aggregate", "Beta_Aggregate", "Gamma_Aggregate"]

    output_columns = [col for col in df.columns if 'RAW' in col] + agg_columns

    df = df[output_columns]

    ## Download dataset into the newly created folder (default name set to "cleaned datasets")
    df.to_csv(f"../cleaned datasets/{new_folder}/subject{subject_id}-label-{record_id}-cleaned.csv")