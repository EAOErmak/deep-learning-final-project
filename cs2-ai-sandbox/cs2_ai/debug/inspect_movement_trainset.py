from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cs2_ai.preprocessing.build_movement_trainset import resolve_position_columns, resolve_tick_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect movement train/val/test parquet trainset.')
    parser.add_argument('--input-dir', type=Path, required=True)
    return parser.parse_args()


def inspect_split(name: str, path: Path) -> None:
    if not path.exists():
        print(f'{name}: missing ({path})')
        return
    df = pd.read_parquet(path)
    print(f'{name}:')
    print(f'  rows: {len(df)}')
    print(f'  groups: {df["split_group_id"].nunique() if "split_group_id" in df.columns else 0}')
    print(f'  columns: {list(df.columns)}')
    for column in ('move_forward', 'move_back', 'move_left', 'move_right', 'move_walk', 'move_crouch', 'move_jump'):
        if column in df.columns:
            print(f'  {column} mean: {float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()):.4f}')
    if 'has_next_cell_target' in df.columns:
        print(f'  has_next_cell_target mean: {float(pd.to_numeric(df["has_next_cell_target"], errors="coerce").fillna(0.0).mean()):.4f}')
    if 'next_cell_distance' in df.columns:
        distances = pd.to_numeric(df['next_cell_distance'], errors='coerce').dropna()
        if not distances.empty:
            print(f'  next_cell_distance min/mean/max: {float(distances.min()):.4f} / {float(distances.mean()):.4f} / {float(distances.max()):.4f}')
    x_col, y_col, z_col = resolve_position_columns(df)
    tick_col = resolve_tick_column(df) or 'index'
    preview_columns = [
        tick_col,
        x_col,
        y_col,
        z_col,
        'next_cell_rel_x',
        'next_cell_rel_y',
        'next_cell_rel_z',
        'next_cell_distance',
        'move_forward',
        'move_back',
        'move_left',
        'move_right',
        'move_walk',
        'move_crouch',
        'move_jump',
    ]
    preview_columns = [column for column in preview_columns if column in df.columns]
    if preview_columns:
        print('  first 5 rows:')
        print(df[preview_columns].head(5).to_string(index=False))


def main() -> int:
    args = parse_args()
    for split_name in ('train', 'val', 'test'):
        inspect_split(split_name, args.input_dir / f'{split_name}.parquet')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
