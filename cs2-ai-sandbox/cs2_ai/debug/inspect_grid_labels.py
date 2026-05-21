from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cs2_ai.navigation.path_labeler import resolve_position_columns, resolve_tick_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect Dust2 grid labels added to clean_play_ticks.')
    parser.add_argument('--input', type=Path, required=True)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main() -> int:
    args = parse_args()
    df = load_table(args.input)
    x_col, y_col, z_col = resolve_position_columns(df)
    tick_column = resolve_tick_column(df)
    print(f'rows: {len(df)}')
    print(f'unique current_cell_id: {df["current_cell_id"].nunique() if "current_cell_id" in df.columns else 0}')
    print(f'unique next_cell_id: {df["next_cell_id"].nunique() if "next_cell_id" in df.columns else 0}')
    if {'cell_ix', 'cell_iy', 'cell_iz'}.issubset(df.columns):
        print(f'min/max ix: {int(df["cell_ix"].min())} / {int(df["cell_ix"].max())}')
        print(f'min/max iy: {int(df["cell_iy"].min())} / {int(df["cell_iy"].max())}')
        print(f'min/max iz: {int(df["cell_iz"].min())} / {int(df["cell_iz"].max())}')
    has_target_pct = float(df['has_next_cell_target'].mean() * 100.0) if 'has_next_cell_target' in df.columns else 0.0
    print(f'percent has_next_cell_target: {has_target_pct:.2f}%')
    if 'dwell_bucket_id' in df.columns:
        print('dwell bucket distribution:')
        print(df['dwell_bucket_id'].value_counts(normalize=True).sort_index().to_string())
    columns = [
        tick_column,
        x_col,
        y_col,
        z_col,
        'current_cell_id',
        'next_cell_id',
        'next_cell_rel_x',
        'next_cell_rel_y',
        'next_cell_distance',
        'dwell_bucket_id',
    ]
    preview_columns = [column for column in columns if column in df.columns]
    print('first 10 rows:')
    print(df[preview_columns].head(10).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
