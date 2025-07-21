import os
from sklearn import tree
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import numpy as np
# Purpose: Training only
# What it does:
    # Loads the dataset
    # Splits data into train/test
    # Trains both RandomForestClassifier and RandomForestRegressor
    # Evaluates models on test set
    # Saves the trained models with timestamps
    # Prints model performance and save locations
start = time.time()
dataset_name = "Mendeley"
data = pd.read_csv("featuresets/Mendeley cleaned_2025-07-11_13-52.csv", nrows= 500).drop(labels="Timestep", axis=1)

X_train, X_test, y_train, y_test = train_test_split(data.drop(axis=1, labels=["Label"]), data["Label"], test_size=0.2, random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2,3,5],
    'min_samples_leaf': [1, 3, 5],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

param_grid2 = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2,3,5],
    'min_samples_leaf': [1, 3, 5],
    'bootstrap': [True, False],
    'criterion': ['squared_error', 'absolute_error', 'poisson']
}
#Best Parameters: {'bootstrap': False, 'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid=param_grid, cv=cv, verbose = 4, n_jobs=-1)
# grid_search.fit(X_train, y_train)
grid_search2 = GridSearchCV(RandomForestRegressor(random_state = 42), param_grid=param_grid2, cv=cv, verbose = 4, n_jobs=-1)
grid_search2.fit(X_train, y_train)

# clf = RandomForestClassifier(**grid_search.best_params_, random_state = 42, verbose = 4, n_jobs=-1)
# clf = clf.fit(X_train, y_train)

reg = RandomForestRegressor(**grid_search2.best_params_, random_state=42, verbose = 4, n_jobs=-1)
reg = reg.fit(X_train, y_train)

# clf_predictions = clf.predict(X_test)
reg_predictions = reg.predict(X_test)

#clf_score = clf.score(X_test, y_test)
reg_score = reg.score(X_test, y_test)









results = pd.DataFrame(grid_search2.cv_results_)

# Sort by best score
top_n = results.sort_values(by='mean_test_score', ascending=False).head(10)

# Make param combo more readable
top_n['param_combo'] = top_n['params'].apply(
    lambda d: '\n'.join([f"{k}={v}" for k, v in d.items()])
)

# Plot
plt.figure(figsize=(14, 10))
sns.barplot(
    data=top_n, 
    x='mean_test_score', 
    y='param_combo', 
    palette='viridis'
)

# Improve aesthetics
plt.xlabel('Mean CV Score', fontsize=12)
plt.ylabel('Hyperparameter Combination', fontsize=12)
plt.title('Top 10 GridSearchCV Regressor Results', fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=9)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.rcParams['font.family'] = 'monospace'

# Save at high resolution
plt.savefig("gridsearchcvregressor_top10.png", dpi=300, bbox_inches='tight')
plt.show()



# Create a parallel coordinates plot for the GridSearchCV results
# Select top rows (optional to reduce clutter)
top = results.sort_values(by='mean_test_score', ascending=False).head(20)

# Keep only numeric columns for plotting
plot_data = top.copy()
for col in plot_data.columns:
    if col.startswith('param_'):
        plot_data[col] = plot_data[col].astype(str)

# Create parallel coordinates plot
plt.figure(figsize=(12,6))
parallel_coordinates(
    plot_data[['mean_test_score'] + [col for col in plot_data.columns if col.startswith('param_')]],
    class_column='mean_test_score',
    colormap='viridis'
)
plt.title('GridSearchCV Regressor Parallel Coordinates Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"gridsearch_parallel.png", dpi=300)
plt.show()






# Plotting each parameter's effect on the score
param_names = [key[6:] for key in results.columns if key.startswith("param_")]
score_types = ["mean_test_score"]
if "mean_train_score" in results.columns:
    score_types.append("mean_train_score")

# Loop through each parameter
for param in param_names:
    param_col = f"param_{param}"
    
    # Convert all parameter values to strings, handle None
    results[param_col] = results[param_col].apply(lambda x: 'None' if x is None else str(x))
    
    x_labels = results[param_col].unique()
    x_pos = np.arange(len(x_labels))
    
    # Compute mean and std for each x_label manually
    grouped = results.groupby(param_col).mean(numeric_only=True)
    
    plt.figure(figsize=(10, 6))
    
    for score_type in score_types:
        means = [grouped.loc[label][score_type] if label in grouped.index else np.nan for label in x_labels]
        stds = [results[results[param_col] == label][score_type.replace('mean', 'std')].mean() for label in x_labels]
        
        plt.errorbar(x_pos, means, yerr=stds, capsize=4, label=score_type.replace('_', ' ').title(), marker='o')

    plt.xticks(x_pos, x_labels, rotation=45)
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.title(f"{param} vs GridSearchCV Regressor Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"gridsearchregressor_param_{param}.png", dpi=300)
    plt.show()

# print("Best Parameters:", grid_search.best_params_)
print("Best Parameter Regressor:", grid_search2.best_params_)
# print(f"Classification accuracy: {clf_score:.4f}")
print(f"Regression R² score: {reg_score:.4f}")
#print(confusion_matrix(y_test, clf_predictions, labels=[1, 2, 3]))
# print(classification_report(y_test, clf_predictions, labels=[1, 2, 3]))

time.sleep(1)
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# Save the trained model
# today = datetime.datetime.now()
# datetime_str = today.strftime("%Y-%m-%d_%H-%M")
# joblib.dump(reg, f'models/{dataset_name}_rf_reg_model_{datetime_str}.pkl')
# joblib.dump(clf, f'models/{dataset_name}_rf_clf_model_{datetime_str}.pkl')

# print(f"Models saved:")
# print(f"  - models/Mendeley_rf_reg_model_{datetime_str}.pkl")
# print(f"  - models/Mendeley_rf_clf_model_{datetime_str}.pkl")