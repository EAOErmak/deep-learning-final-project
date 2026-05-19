from pathlib import Path
import pandas as pd

PARQUET_PATH = Path("dataset/raw_ticks")

def main():
    files = list(PARQUET_PATH.glob("*.parquet"))

    if not files:
        print("No parquet files found in dataset/raw_ticks/")
        return

    file = files[0]
    print(f"Reading: {file}")

    df = pd.read_parquet(file)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nSample row as dict:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    main()