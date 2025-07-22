import os
from sklearn import tree
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matplotlib import pyplot as plt
import datetime
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Purpose: Training only
# What it does:
    # Loads the dataset
    # Splits data into train/test
    # Trains both RandomForestClassifier and RandomForestRegressor
    # Evaluates models on test set
    # Saves the trained models with timestamps
    # Prints model performance and save locations

dataset_name = "NeuroSense"
data = pd.read_csv("featuresets/NeuroSense_2025-07-22_06-57.csv", nrows=500).drop(labels="Timestep", axis=1)
data.dropna(how="any", axis=0)

plt.scatter(data["Valence"], data["Arousal"], alpha=0.7, c=data["Label"])
plt.xlabel("Actual Valence")
plt.ylabel("Actual Arousal")
plt.title("Actual Valence vs Arousal")
plt.grid(True)
plt.savefig("Actual_valence_arousal.png", dpi=300, bbox_inches='tight')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label", "Valence", "Arousal"]), data[["Valence", "Arousal"]], test_size=0.2, random_state=42)

reg = RandomForestRegressor(criterion='absolute_error', min_samples_leaf=5, max_depth=5, n_jobs=-1)
reg = reg.fit(X_train, y_train)

reg_predictions = reg.predict(X_test)
pred_df = pd.DataFrame(reg_predictions)
pred_df["Index"] = X_test.index
pred_df.set_index("Index", inplace=True)
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

true_quadrants = list(data.loc[X_test.index, "Label"])

faux_class_predictions = [int(p) for p in faux_class_predictions]
true_quadrants = [int(t) for t in true_quadrants]

accuracy = accuracy_score(true_quadrants, faux_class_predictions)

print(faux_class_predictions)
print(true_quadrants)
print(accuracy)


plt.scatter(reg_predictions[:, 0], reg_predictions[:, 1], alpha=0.7, c=true_quadrants)
plt.xlabel("Predicted Valence")
plt.ylabel("Predicted Arousal")
plt.title("Predicted Valence vs Arousal")
plt.grid(True)
plt.savefig("predicted_valence_arousal.png", dpi=300, bbox_inches='tight')
plt.show()


# Save the trained model
today = datetime.datetime.now()
datetime_str = today.strftime("%Y-%m-%d_%H-%M")
joblib.dump(reg, f'models/{dataset_name}_rf_reg_model_{datetime_str}.pkl')

print(f"Models saved:")
print(f"  - models/{dataset_name}_rf_reg_model_{datetime_str}.pkl")