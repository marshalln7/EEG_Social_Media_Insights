import pandas as pd
import numpy as np

# Load your CSV
df = pd.read_csv(r"emotions.csv")

# Create the timestep column
timestep = np.arange(0.5, 0.5 * (len(df) + 1), 0.5)

# Insert at position 0 (beginning of the DataFrame)
df.insert(0, "Timestep", timestep)
df = df.rename(columns={"label": "Label"})

df.to_csv("emotions.csv", index=False)