from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cs2_ai.preprocessing.build_movement_trainset import GRID_STATS_COLUMNS, resolve_position_columns, resolve_tick_column


ACTION_COLUMNS = ('move_forward', 'move_back', 'move_left', 'move_right', 'move_walk', 'move_crouch', 'move_jump')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect movement train/val/test parquet trainset.')
    parser.add_argument('--input-dir', type=Path, required=True)
    return parser.parse_args()


def read_manifest(input_dir: Path) -> dict | None:
    path = input_dir / 'manifest.json'
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def print_action_distribution(df: pd.DataFrame) -> None:
    print('  action distribution:')
    for column in ACTION_COLUMNS:
        if column in df.columns:
            value = float(pd.to_numeric(df[column], errors='coerce').fillna(0.0).mean())
            print(f'    {column}: {value:.4f}')


def print_grid_stats(df: pd.DataFrame) -> None:
    if not all(column in df.columns for column in GRID_STATS_COLUMNS):
        print('  grid stats: unavailable')
        return
    print('  grid stats:')
    has_target = float(pd.to_numeric(df['has_next_cell_target'], errors='coerce').fillna(0.0).mean())
    print(f'    has_next_cell_target mean: {has_target:.4f}')
    distances = pd.to_numeric(df['next_cell_distance'], errors='coerce').dropna()
    if not distances.empty:
        print(f'    next_cell_distance min/mean/max: {float(distances.min()):.4f} / {float(distances.mean()):.4f} / {float(distances.max()):.4f}')
    for column in ('dwell_pass_through', 'dwell_short_hold', 'dwell_medium_hold', 'dwell_long_hold'):
        value = float(pd.to_numeric(df[column], errors='coerce').fillna(0.0).mean())
        print(f'    {column}: {value:.4f}')


def inspect_split(name: str, path: Path) -> dict[str, int]:
    if not path.exists():
        print(f'{name}: missing ({path})')
        return {'rows': 0, 'rounds': 0, 'demos': 0, 'groups': 0}

    df = pd.read_parquet(path)
    rounds = int(df['split_round_id'].nunique()) if 'split_round_id' in df.columns else 0
    demos = int(df['demo_file_name'].nunique()) if 'demo_file_name' in df.columns else 0
    groups = int(df['split_group_id'].nunique()) if 'split_group_id' in df.columns else 0
    print(f'{name}:')
    print(f'  rows: {len(df)}')
    print(f'  groups: {groups}')
    print(f'  rounds: {rounds}')
    print(f'  demos: {demos}')
    print_action_distribution(df)
    print_grid_stats(df)

    x_col, y_col, z_col = resolve_position_columns(df)
    tick_col = resolve_tick_column(df) or 'index'
    preview_columns = [
        tick_col,
        x_col,
        y_col,
        z_col,
        'demo_file_name',
        'round_number',
        'split_group_id',
        'next_cell_rel_x',
        'next_cell_rel_y',
        'next_cell_rel_z',
        'next_cell_distance',
    ]
    preview_columns = [column for column in preview_columns if column in df.columns]
    if preview_columns:
        print('  first 5 rows:')
        print(df[preview_columns].head(5).to_string(index=False))
    return {'rows': int(len(df)), 'rounds': rounds, 'demos': demos, 'groups': groups}


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.input_dir)
    if manifest is not None:
        print('manifest:')
        print(f'  input_mode: {manifest.get("input_mode")}')
        print(f'  feature_mode: {manifest.get("feature_mode")}')
        print(f'  split unit: {manifest.get("split_unit")}')
        print(f'  total demos: {manifest.get("total_demos")}')
        print(f'  total rounds: {manifest.get("total_rounds")}')

    totals: dict[str, dict[str, int]] = {}
    for split_name in ('train', 'val', 'test'):
        totals[split_name] = inspect_split(split_name, args.input_dir / f'{split_name}.parquet')

    print('summary:')
    for split_name in ('train', 'val', 'test'):
        item = totals[split_name]
        print(f'  {split_name}: rows={item["rows"]} rounds={item["rounds"]} demos={item["demos"]} groups={item["groups"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
