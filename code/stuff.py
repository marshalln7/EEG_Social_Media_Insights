import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv(r"featuresets\Kaggle_Emotions_Dataset\emotions.csv")

df = df.rename(columns={"label": "Label"})

df.to_csv("emotions.csv", index=False)