import pandas as pd
import numpy as np

rows = []

for unit in range(1, 4):
    for cycle in range(1, 201):
        noise = np.random.normal(0, 0.02, 5)
        degradation = cycle * 0.002
        sensors = 1 - degradation + noise

        rows.append([
            unit,
            cycle,
            *sensors
        ])

df = pd.DataFrame(
    rows,
    columns=["unit", "cycle", "s1", "s2", "s3", "s4", "s5"]
)

df.to_csv("data/train.csv", index=False)
print("train.csv generated")
