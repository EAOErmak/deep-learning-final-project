from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

from pathlib import Path

import pandas as pd


def list_parquet_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.parquet"))


def print_shapes(title: str, files: list[Path]) -> None:
    print(f"\n== {title} ==")
    if not files:
        print("No parquet files found.")
        return

    for path in files:
        try:
            df = pd.read_parquet(path)
            print(f"{path.name}: shape={df.shape}")
        except Exception as exc:
            print(f"{path.name}: failed to read ({exc})")


def print_preview(title: str, files: list[Path]) -> None:
    print(f"\n== {title} Preview ==")
    if not files:
        print("No parquet files found.")
        return

    try:
        df = pd.read_parquet(files[0])
        print(df.head())
    except Exception as exc:
        print(f"Failed to preview {files[0].name}: {exc}")


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"

    raw_files = list_parquet_files(dataset_dir / "raw_ticks")
    play_files = list_parquet_files(dataset_dir / "clean_play_ticks")
    buy_files = list_parquet_files(dataset_dir / "clean_buy_ticks")
    round_files = list_parquet_files(dataset_dir / "round_events")

    print_shapes("raw_ticks", raw_files)
    print_shapes("clean_play_ticks", play_files)
    print_shapes("clean_buy_ticks", buy_files)
    print_shapes("round_events", round_files)

    print_preview("clean_play_ticks", play_files)
    print_preview("clean_buy_ticks", buy_files)
    print_preview("round_events", round_files)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
