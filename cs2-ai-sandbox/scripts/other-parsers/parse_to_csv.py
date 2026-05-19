from pathlib import Path
import pandas as pd

file = next(Path("dataset/raw_ticks").glob("*.parquet"))

df = pd.read_parquet(file)

sample = df.head(100)
sample.to_csv("dataset/sample_ticks.csv", index=False)

print("Saved: dataset/sample_ticks.csv")