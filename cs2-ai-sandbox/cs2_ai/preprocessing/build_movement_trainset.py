from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from cs2_ai.features.movement_features import GRID_NAVIGATION_REQUIRED_COLUMNS, MOVEMENT_FEATURE_MODE_SOLO_GRID
from cs2_ai.navigation.path_labeler import resolve_position_columns


REQUIRED_MOVE_COLUMNS = ('move_forward', 'move_back', 'move_left', 'move_right')
OPTIONAL_MOVE_COLUMNS = ('move_walk', 'move_crouch', 'move_jump')
GRID_STATS_COLUMNS = (
    'has_next_cell_target',
    'next_cell_distance',
    'dwell_pass_through',
    'dwell_short_hold',
    'dwell_medium_hold',
    'dwell_long_hold',
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y'}:
        return True
    if normalized in {'0', 'false', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def resolve_velocity_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    candidates = (
        ('velocity_X', 'velocity_Y', 'velocity_Z'),
        ('vel_x', 'vel_y', 'vel_z'),
        ('velocity_x', 'velocity_y', 'velocity_z'),
        ('self_vel_x', 'self_vel_y', 'self_vel_z'),
        ('player_vel_x', 'player_vel_y', 'player_vel_z'),
    )
    for triplet in candidates:
        if all(column in df.columns for column in triplet):
            return triplet
    raise ValueError('Velocity columns not found. Expected one of: velocity_X/Y/Z, vel_x/y/z, self_vel_x/y/z, player_vel_x/y/z.')


def resolve_yaw_column(df: pd.DataFrame) -> str:
    for column in ('yaw', 'view_yaw', 'self_yaw', 'player_yaw'):
        if column in df.columns:
            return column
    raise ValueError('Yaw column not found. Expected one of: yaw, view_yaw, self_yaw, player_yaw.')


def resolve_tick_column(df: pd.DataFrame) -> str | None:
    for column in ('tick', 'tick_id', 'server_tick'):
        if column in df.columns:
            return column
    return None


def resolve_map_column(df: pd.DataFrame) -> str | None:
    for column in ('map', 'map_name'):
        if column in df.columns:
            return column
    return None


def canonicalize_position_and_state_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str, str], tuple[str, str, str], str]:
    x_col, y_col, z_col = resolve_position_columns(df)
    vel_x_col, vel_y_col, vel_z_col = resolve_velocity_columns(df)
    yaw_col = resolve_yaw_column(df)
    canonical = df.copy()
    if (x_col, y_col, z_col) != ('X', 'Y', 'Z'):
        canonical['X'] = pd.to_numeric(canonical[x_col], errors='coerce')
        canonical['Y'] = pd.to_numeric(canonical[y_col], errors='coerce')
        canonical['Z'] = pd.to_numeric(canonical[z_col], errors='coerce')
    if (vel_x_col, vel_y_col, vel_z_col) != ('velocity_X', 'velocity_Y', 'velocity_Z'):
        canonical['velocity_X'] = pd.to_numeric(canonical[vel_x_col], errors='coerce')
        canonical['velocity_Y'] = pd.to_numeric(canonical[vel_y_col], errors='coerce')
        canonical['velocity_Z'] = pd.to_numeric(canonical[vel_z_col], errors='coerce')
    if yaw_col != 'yaw':
        canonical['yaw'] = pd.to_numeric(canonical[yaw_col], errors='coerce')
    return canonical, (x_col, y_col, z_col), (vel_x_col, vel_y_col, vel_z_col), yaw_col


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    numeric = pd.to_numeric(series, errors='coerce')
    if not numeric.isna().all():
        return numeric.fillna(0.0).clip(0.0, 1.0)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({'1', 'true', 'yes', 'y'}).astype(float)


def resolve_movement_target_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str | None]]:
    candidates = {
        'move_forward': ('move_forward', 'forward', 'FORWARD'),
        'move_back': ('move_back', 'back', 'BACK', 'move_back', 'backward', 'move_backward'),
        'move_left': ('move_left', 'left', 'LEFT'),
        'move_right': ('move_right', 'right', 'RIGHT'),
        'move_walk': ('move_walk', 'walk', 'WALK', 'is_walking'),
        'move_crouch': ('move_crouch', 'crouch', 'duck', 'ducking', 'is_ducking'),
        'move_jump': ('move_jump', 'jump', 'JUMP', 'is_jumping', 'jump_pressed'),
    }
    result = df.copy()
    resolved: dict[str, str | None] = {}
    for canonical, options in candidates.items():
        source = next((column for column in options if column in result.columns), None)
        resolved[canonical] = source
        if source is None:
            if canonical in REQUIRED_MOVE_COLUMNS:
                raise ValueError(f'Missing required movement target column for {canonical}. Checked: {options}')
            print(f'WARNING: optional movement target column missing for {canonical}; filling zeros.')
            result[canonical] = 0.0
            continue
        result[canonical] = coerce_bool_series(result[source])
    if 'FORWARD' not in result.columns:
        result['FORWARD'] = result['move_forward']
    if 'BACK' not in result.columns:
        result['BACK'] = result['move_back']
    if 'LEFT' not in result.columns:
        result['LEFT'] = result['move_left']
    if 'RIGHT' not in result.columns:
        result['RIGHT'] = result['move_right']
    if 'WALK' not in result.columns:
        result['WALK'] = result['move_walk']
    if 'is_walking' not in result.columns:
        result['is_walking'] = result['move_walk']
    if 'ducking' not in result.columns:
        result['ducking'] = result['move_crouch']
    if 'JUMP' not in result.columns:
        result['JUMP'] = result['move_jump']
    return result, resolved


def ensure_grid_columns(df: pd.DataFrame, required: bool) -> pd.DataFrame:
    missing = [column for column in GRID_NAVIGATION_REQUIRED_COLUMNS if column not in df.columns]
    if missing and required:
        raise ValueError(
            f'Missing required grid navigation columns: {missing}. '
            'Run cs2_ai.preprocessing.label_dust2_grid first.'
        )
    result = df.copy()
    for column in missing:
        print(f'WARNING: grid navigation column missing for trainset build: {column}; filling zeros.')
        result[column] = 0.0
    return result


def resolve_group_key_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    demo_col = next((column for column in ('demo_id', 'demo_name') if column in df.columns), None)
    round_col = next((column for column in ('round_id', 'total_rounds_played') if column in df.columns), None)
    player_col = next((column for column in ('player_id', 'steam_id', 'steamid') if column in df.columns), None)
    if demo_col is None:
        demo_col = 'source_file'
    if round_col is None:
        df['round_id'] = 'unknown_round'
        round_col = 'round_id'
    if player_col is None:
        df['player_id'] = 'unknown_player'
        player_col = 'player_id'
    return demo_col, round_col, player_col


def build_split_group_id(df: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str, str]]:
    result = df.copy()
    demo_col, round_col, player_col = resolve_group_key_columns(result)
    result['split_group_id'] = (
        result[demo_col].astype(str)
        + '::'
        + result[round_col].astype(str)
        + '::'
        + result[player_col].astype(str)
    )
    result['split_round_id'] = result[demo_col].astype(str) + '::' + result[round_col].astype(str)
    return result, (demo_col, round_col, player_col)


def split_group_ids(group_ids: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> tuple[set[str], set[str], set[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f'train/val/test ratios must sum to 1.0, got {total}')
    import random

    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)
    total_groups = len(shuffled)
    if total_groups == 0:
        return set(), set(), set()
    train_count = int(total_groups * train_ratio)
    val_count = int(total_groups * val_ratio)
    if total_groups >= 3:
        if train_ratio > 0 and train_count == 0:
            train_count = 1
        if val_ratio > 0 and val_count == 0:
            val_count = 1
    if train_count + val_count > total_groups:
        val_count = max(0, total_groups - train_count)
    test_count = total_groups - train_count - val_count
    if total_groups >= 3 and test_ratio > 0 and test_count == 0:
        if train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        test_count = total_groups - train_count - val_count
    train_end = train_count
    val_end = train_end + val_count
    train = set(shuffled[:train_end])
    val = set(shuffled[train_end:val_end])
    test = set(shuffled[val_end:val_end + test_count])
    return train, val, test


def print_action_distribution(df: pd.DataFrame, prefix: str) -> None:
    print(f'{prefix} action distribution:')
    for column in ('move_forward', 'move_back', 'move_left', 'move_right', 'move_walk', 'move_crouch', 'move_jump'):
        if column in df.columns:
            print(f'  {column}: {float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).mean()):.4f}')


def print_grid_stats(df: pd.DataFrame, required_grid: bool) -> None:
    if not required_grid and not all(column in df.columns for column in GRID_STATS_COLUMNS):
        return
    print('grid stats:')
    print(f'  has_next_cell_target mean: {float(pd.to_numeric(df["has_next_cell_target"], errors="coerce").fillna(0.0).mean()):.4f}')
    distances = pd.to_numeric(df['next_cell_distance'], errors='coerce').dropna()
    if not distances.empty:
        print(f'  next_cell_distance min/mean/max: {float(distances.min()):.4f} / {float(distances.mean()):.4f} / {float(distances.max()):.4f}')
    dwell_distribution = {
        column: float(pd.to_numeric(df[column], errors='coerce').fillna(0.0).mean())
        for column in ('dwell_pass_through', 'dwell_short_hold', 'dwell_medium_hold', 'dwell_long_hold')
        if column in df.columns
    }
    for column, value in dwell_distribution.items():
        print(f'  {column}: {value:.4f}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build reproducible movement train/val/test parquet splits.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--map', type=str, default='de_dust2')
    parser.add_argument('--feature-mode', choices=['legacy', 'solo_grid'], default='solo_grid')
    parser.add_argument('--input-glob', type=str, default='**/clean_play_ticks*.parquet')
    parser.add_argument('--require-grid-labels', type=parse_bool, default=True)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-group-rows', type=int, default=32)
    parser.add_argument('--max-rows', type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    parquet_files = sorted(args.input_dir.glob(args.input_glob))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found for {args.input_dir} / {args.input_glob}')

    print(f'Found parquet files: {len(parquet_files)}')
    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        frame = pd.read_parquet(path)
        frame['source_file'] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    total_rows_before = len(df)
    print(f'total rows before filtering: {total_rows_before}')

    map_col = resolve_map_column(df)
    if map_col is None:
        print('WARNING: map column not found; skipping map filter.')
    else:
        df = df.loc[df[map_col].astype(str) == args.map].copy()

    df, position_columns, velocity_columns, yaw_column = canonicalize_position_and_state_columns(df)
    df, target_column_map = resolve_movement_target_columns(df)
    require_grid = args.feature_mode == MOVEMENT_FEATURE_MODE_SOLO_GRID and bool(args.require_grid_labels)
    df = ensure_grid_columns(df, required=require_grid)

    required_columns = ['X', 'Y', 'Z', 'velocity_X', 'velocity_Y', 'velocity_Z', 'yaw', *REQUIRED_MOVE_COLUMNS]
    required_columns.extend(OPTIONAL_MOVE_COLUMNS)
    if require_grid:
        required_columns.extend(GRID_NAVIGATION_REQUIRED_COLUMNS)
    df = df.dropna(subset=[column for column in required_columns if column in df.columns]).copy()

    tick_col = resolve_tick_column(df)
    if tick_col is None:
        print('WARNING: tick column not found; preserving original row order.')
    else:
        df = df.sort_values([column for column in ('source_file', 'total_rounds_played', 'tick', 'steamid') if column in df.columns]).reset_index(drop=True)

    df, group_columns = build_split_group_id(df)
    group_sizes = df.groupby('split_group_id').size()
    valid_group_ids = set(group_sizes[group_sizes >= int(args.min_group_rows)].index.tolist())
    df = df.loc[df['split_group_id'].isin(valid_group_ids)].copy()

    total_rows_after = len(df)
    all_group_ids = sorted(pd.unique(df['split_group_id']))
    print(f'total rows after filtering: {total_rows_after}')
    print(f'number of groups: {len(all_group_ids)}')
    train_ids, val_ids, test_ids = split_group_ids(all_group_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    df['split'] = 'test'
    df.loc[df['split_group_id'].isin(train_ids), 'split'] = 'train'
    df.loc[df['split_group_id'].isin(val_ids), 'split'] = 'val'

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = df.loc[df['split'] == 'train'].copy()
    val_df = df.loc[df['split'] == 'val'].copy()
    test_df = df.loc[df['split'] == 'test'].copy()
    train_path = output_dir / 'train.parquet'
    val_path = output_dir / 'val.parquet'
    test_path = output_dir / 'test.parquet'
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'input_dir': str(args.input_dir),
        'input_glob': args.input_glob,
        'map': args.map,
        'feature_mode': args.feature_mode,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'total_rows': int(len(df)),
        'train_rows': int(len(train_df)),
        'val_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
        'total_groups': int(len(all_group_ids)),
        'train_groups': int(len(train_ids)),
        'val_groups': int(len(val_ids)),
        'test_groups': int(len(test_ids)),
        'required_grid_columns': list(GRID_NAVIGATION_REQUIRED_COLUMNS) if require_grid else [],
        'movement_target_columns': target_column_map,
        'position_columns': list(position_columns),
        'velocity_columns': list(velocity_columns),
        'yaw_column': yaw_column,
        'group_columns': list(group_columns),
        'source_files': [str(path) for path in parquet_files],
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding='utf-8')

    print(f'split rows/groups: train={len(train_df)}/{len(train_ids)} val={len(val_df)}/{len(val_ids)} test={len(test_df)}/{len(test_ids)}')
    print_action_distribution(train_df, prefix='train')
    print_grid_stats(train_df, required_grid=require_grid)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
