from __future__ import annotations

from pathlib import Path

import pandas as pd


def find_first_parquet(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.parquet"))
    return files[0] if files else None


def export_sample_csv(source_dir: Path, output_file: Path) -> None:
    parquet_file = find_first_parquet(source_dir)
    if parquet_file is None:
        print(f"[skip] No parquet files found in {source_dir}")
        return

    df = pd.read_parquet(parquet_file)
    df.head(1).to_csv(output_file, index=False)
    print(f"[ok] {parquet_file.name} -> {output_file}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "dataset"

    export_sample_csv(dataset_dir / "clean_buy_ticks", dataset_dir / "sample_clean_buy_ticks.csv")
    export_sample_csv(dataset_dir / "clean_play_ticks", dataset_dir / "sample_clean_play_ticks.csv")
    export_sample_csv(dataset_dir / "events", dataset_dir / "sample_events.csv")
    export_sample_csv(dataset_dir / "raw_ticks", dataset_dir / "sample_raw_ticks.csv")
    export_sample_csv(dataset_dir / "round_events", dataset_dir / "sample_round_events.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
