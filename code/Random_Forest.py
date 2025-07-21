import os
from sklearn import tree
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import datetime
from sklearn.utils import resample

# Purpose: Training only
# What it does:
    # Loads the dataset
    # Splits data into train/test
    # Trains both RandomForestClassifier and RandomForestRegressor
    # Evaluates models on test set
    # Saves the trained models with timestamps
    # Prints model performance and save locations

dataset_name = "Emotions"
data = pd.read_csv("featuresets\Mendeley cleaned_2025-07-18_11-02.csv").drop(labels="Timestep", axis=1)
data.dropna(how="any", axis=0)

classes = data["Label"].unique()
print(f"Classes found: {classes}")

for x in classes:
    print(f"Class {x} has {len(data[data['Label'] == x])} samples")

print("Downsampling...")

smallest_class_size = min([len(data[data["Label"] == x]) for x in classes])
downsample_chunks = []

for x in classes:
    # Downsample each class
    df_downsampled = resample(
        data[data["Label"] == x],
        replace=False,                     # no bootstrapping
        n_samples=smallest_class_size,        # match minority size
        random_state=42
    )
    downsample_chunks.append(df_downsampled)

data = pd.concat(downsample_chunks)



X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label"]), data["Label"], test_size=0.2, random_state=42)

clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=5, max_depth=5, n_jobs=-1)
clf = clf.fit(X_train, y_train)

reg = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5, n_jobs=-1)
reg = reg.fit(X_train, y_train)

clf_predictions = clf.predict(X_test)
reg_predictions = reg.predict(X_test)

clf_score = clf.score(X_test, y_test)
reg_score = reg.score(X_test, y_test)

print(f"Classification accuracy: {clf_score:.4f}")
print(f"Regression R² score: {reg_score:.4f}")

# Save the trained model
today = datetime.datetime.now()
datetime_str = today.strftime("%Y-%m-%d_%H-%M")
joblib.dump(reg, f'models/{dataset_name}_rf_reg_model_{datetime_str}.pkl')
joblib.dump(clf, f'models/{dataset_name}_rf_clf_model_{datetime_str}.pkl')

print(f"Models saved:")
print(f"  - models/{dataset_name}_rf_reg_model_{datetime_str}.pkl")
print(f"  - models/{dataset_name}_rf_clf_model_{datetime_str}.pkl")