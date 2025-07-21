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

dataset_name = "NeuroSense"
data = pd.read_csv("featuresets/NeuroSense_2025-07-21_14-39.csv", nrows=2000).drop(labels="Timestep", axis=1)
data.dropna(how="any", axis=0)

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label", "Valence", "Arousal"]), data[["Valence", "Arousal"]], test_size=0.2, random_state=42)

reg = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5, n_jobs=-1)
reg = reg.fit(X_train, y_train)

reg_predictions = reg.predict(X_test)
print(reg_predictions)

reg_score = reg.score(X_test, y_test)

print(f"Regression R² score: {reg_score:.4f}")

faux_class_predictions = []
for line in reg_predictions:
    if line[0] >= 5:
        if line[1] >= 5:
            new = 3.
        else:
            new = 2.
    else:
        if line[1] >= 5:
            new = 0.
        else:
            new = 1.
    faux_class_predictions.append(new)

#TO DO Make these actually correspond to each other
print(faux_class_predictions)
print(data["Label"])


# Save the trained model
today = datetime.datetime.now()
datetime_str = today.strftime("%Y-%m-%d_%H-%M")
joblib.dump(reg, f'models/{dataset_name}_rf_reg_model_{datetime_str}.pkl')

print(f"Models saved:")
print(f"  - models/{dataset_name}_rf_reg_model_{datetime_str}.pkl")