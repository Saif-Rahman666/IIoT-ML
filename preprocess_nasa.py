# preprocess_nasa.py
import pandas as pd
import numpy as np

COLS = (
    ["unit", "time"] +
    [f"os{i}" for i in range(1, 4)] +
    [f"s{i}" for i in range(1, 22)]
)

def main():
    df = pd.read_csv(
        "data/nasa/train_FD001.txt",
        sep=r"\s+",
        header=None
    )
    df.columns = COLS

    # Compute RUL
    max_cycle = df.groupby("unit")["time"].transform("max")
    df["RUL"] = max_cycle - df["time"]

    df.to_csv("data/train.csv", index=False)
    print("Saved data/train.csv with RUL")

if __name__ == "__main__":
    main()
