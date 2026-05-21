from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cs2_ai.features.movement_features import GRID_NAVIGATION_REQUIRED_COLUMNS, MOVEMENT_FEATURE_MODE_SOLO_GRID
from cs2_ai.navigation.cell_indexer import build_grid_map
from cs2_ai.navigation.path_labeler import (
    label_navigation_for_group,
    resolve_group_columns,
    resolve_position_columns,
)


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
ROUND_FILE_PATTERN = re.compile(r'^round_(?P<round_number>.+)\.parquet$', re.IGNORECASE)


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
        'move_back': ('move_back', 'back', 'BACK', 'backward', 'move_backward'),
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
            'Grid labels are missing. Run label_dust2_grid or use --auto-label-grid true. '
            f'Missing columns: {missing}'
        )
    result = df.copy()
    for column in missing:
        print(f'WARNING: grid navigation column missing for trainset build: {column}; filling zeros.')
        result[column] = 0.0
    return result


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def discover_round_parquet_files(rounds_dataset_dir: Path) -> list[Path]:
    return sorted(path for path in rounds_dataset_dir.glob('*/rounds/round_*.parquet') if path.is_file())


def extract_round_number(round_file: Path) -> int | str:
    match = ROUND_FILE_PATTERN.match(round_file.name)
    if match is None:
        print(f'WARNING: could not extract round number from {round_file.name}; using file name.')
        return round_file.name
    raw_value = match.group('round_number')
    try:
        return int(raw_value)
    except ValueError:
        print(f'WARNING: round number is not numeric in {round_file.name}; using "{raw_value}" as string.')
        return raw_value


def resolve_round_source_file(round_file: Path) -> str:
    demo_dir = round_file.parents[1]
    demo_manifest = read_optional_json(demo_dir / 'manifest.json') or {}
    source_file = demo_manifest.get('source_file_name') or demo_manifest.get('source_file')
    if source_file:
        return str(source_file)
    return demo_dir.name


def load_rounds_dataset(rounds_dataset_dir: Path) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    round_files = discover_round_parquet_files(rounds_dataset_dir)
    if not round_files:
        raise FileNotFoundError(f'No round parquet files found for {rounds_dataset_dir} / */rounds/round_*.parquet')

    top_manifest = read_optional_json(rounds_dataset_dir / 'manifest.json')
    top_summary_exists = (rounds_dataset_dir / 'rounds_summary.csv').exists()
    if top_manifest is not None:
        print(f'DEBUG: loaded rounds dataset manifest from {rounds_dataset_dir / "manifest.json"}')
    if top_summary_exists:
        print(f'DEBUG: found rounds dataset summary at {rounds_dataset_dir / "rounds_summary.csv"}')

    frames: list[pd.DataFrame] = []
    for round_file in round_files:
        frame = pd.read_parquet(round_file)
        demo_file_name = round_file.parents[1].name
        round_number = extract_round_number(round_file)
        source_file = resolve_round_source_file(round_file)
        frame = frame.copy()
        frame['demo_file_name'] = demo_file_name
        frame['round_number'] = round_number
        frame['source_file'] = source_file
        frame['round_parquet_path'] = str(round_file)
        frames.append(frame)

    metadata = {
        'rounds_dataset_dir': str(rounds_dataset_dir),
        'top_manifest_exists': top_manifest is not None,
        'top_summary_exists': top_summary_exists,
    }
    return pd.concat(frames, ignore_index=True), round_files, metadata


def load_legacy_input_dataset(input_dir: Path, input_glob: str) -> tuple[pd.DataFrame, list[Path], dict[str, Any]]:
    parquet_files = sorted(input_dir.glob(input_glob))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found for {input_dir} / {input_glob}')

    frames: list[pd.DataFrame] = []
    for path in parquet_files:
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame['source_file'] = str(path)
        if 'demo_file_name' not in frame.columns:
            frame['demo_file_name'] = path.stem
        if 'round_number' not in frame.columns:
            if 'round_id' in frame.columns:
                frame['round_number'] = frame['round_id']
            elif 'total_rounds_played' in frame.columns:
                frame['round_number'] = frame['total_rounds_played']
        frames.append(frame)

    metadata = {
        'input_dir': str(input_dir),
        'input_glob': input_glob,
    }
    return pd.concat(frames, ignore_index=True), parquet_files, metadata


def sort_columns_for_output(df: pd.DataFrame) -> list[str]:
    preferred = ['demo_file_name', 'round_number', 'source_file', 'split', 'split_group_id', 'split_round_id']
    return [column for column in preferred if column in df.columns] + [column for column in df.columns if column not in preferred]


def label_grid_navigation_dataframe(
    df: pd.DataFrame,
    *,
    map_name: str,
    lookahead_ticks: int,
    min_target_distance: float,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    grid_map = build_grid_map(map_name)
    x_col, y_col, z_col = resolve_position_columns(df)
    tick_column = resolve_tick_column(df)
    if tick_column is None:
        raise ValueError('Grid auto-labelling requires a tick column. Expected one of: tick, tick_id, server_tick.')
    group_columns = resolve_group_columns(df)
    labeled_groups: list[pd.DataFrame] = []
    if group_columns:
        for _, group in df.groupby(group_columns, sort=False, dropna=False):
            labeled_groups.append(
                label_navigation_for_group(
                    group,
                    grid_map=grid_map,
                    lookahead_ticks=lookahead_ticks,
                    min_target_distance=min_target_distance,
                    x_col=x_col,
                    y_col=y_col,
                    z_col=z_col,
                    tick_column=tick_column,
                )
            )
    else:
        labeled_groups.append(
            label_navigation_for_group(
                df,
                grid_map=grid_map,
                lookahead_ticks=lookahead_ticks,
                min_target_distance=min_target_distance,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                tick_column=tick_column,
            )
        )
    return pd.concat(labeled_groups, ignore_index=True)


def resolve_demo_column(df: pd.DataFrame) -> str:
    for column in ('demo_file_name', 'demo_id', 'demo_name', 'source_file'):
        if column in df.columns:
            return column
    raise ValueError('Could not resolve demo grouping column.')


def resolve_round_column(df: pd.DataFrame) -> str:
    for column in ('round_number', 'round_id', 'total_rounds_played'):
        if column in df.columns:
            return column
    raise ValueError('Could not resolve round grouping column.')


def build_split_group_id(df: pd.DataFrame, split_unit: str) -> tuple[pd.DataFrame, dict[str, str]]:
    result = df.copy()
    demo_col = resolve_demo_column(result)
    round_col = resolve_round_column(result)
    result['split_round_id'] = result[demo_col].astype(str) + '::' + result[round_col].astype(str)
    if split_unit == 'round':
        result['split_group_id'] = result['split_round_id']
    elif split_unit == 'demo':
        result['split_group_id'] = result[demo_col].astype(str)
    else:
        raise ValueError(f'Unsupported split unit: {split_unit}')
    return result, {
        'demo_column': demo_col,
        'round_column': round_col,
        'split_unit': split_unit,
    }


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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', type=Path)
    input_group.add_argument('--rounds-dataset-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--map', type=str, default='de_dust2')
    parser.add_argument('--feature-mode', choices=['legacy', 'solo_grid'], default='solo_grid')
    parser.add_argument('--input-glob', type=str, default='**/clean_play_ticks*.parquet')
    parser.add_argument('--split-unit', choices=['round', 'demo'], default='round')
    parser.add_argument('--require-grid-labels', type=parse_bool, default=True)
    parser.add_argument('--auto-label-grid', type=parse_bool, default=False)
    parser.add_argument('--lookahead-ticks', type=int, default=10)
    parser.add_argument('--min-target-distance', type=float, default=75.0)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-group-rows', type=int, default=32)
    parser.add_argument('--max-rows', type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds_dataset_dir is not None:
        df, source_files, input_metadata = load_rounds_dataset(args.rounds_dataset_dir)
        input_mode = 'rounds-dataset'
    else:
        df, source_files, input_metadata = load_legacy_input_dataset(args.input_dir, args.input_glob)
        input_mode = 'legacy-input-dir'

    if args.max_rows is not None:
        df = df.head(int(args.max_rows)).copy()
    total_rows_before = len(df)
    print(f'Found parquet files: {len(source_files)}')
    print(f'total rows before filtering: {total_rows_before}')

    map_col = resolve_map_column(df)
    if map_col is None:
        print('WARNING: map column not found; skipping map filter.')
    else:
        df = df.loc[df[map_col].astype(str) == args.map].copy()

    df, position_columns, velocity_columns, yaw_column = canonicalize_position_and_state_columns(df)
    df, target_column_map = resolve_movement_target_columns(df)
    require_grid = args.feature_mode == MOVEMENT_FEATURE_MODE_SOLO_GRID and bool(args.require_grid_labels)
    has_all_grid_columns = all(column in df.columns for column in GRID_NAVIGATION_REQUIRED_COLUMNS)
    if not has_all_grid_columns and bool(args.auto_label_grid):
        print('Grid labels are missing; running auto grid labelling.')
        df = label_grid_navigation_dataframe(
            df,
            map_name=args.map,
            lookahead_ticks=args.lookahead_ticks,
            min_target_distance=args.min_target_distance,
        )
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
        sort_columns = [column for column in ('demo_file_name', 'source_file', 'round_number', 'total_rounds_played', tick_col, 'steamid') if column in df.columns]
        df = df.sort_values(sort_columns).reset_index(drop=True)

    df, group_columns = build_split_group_id(df, split_unit=args.split_unit)
    group_sizes = df.groupby('split_group_id').size()
    valid_group_ids = set(group_sizes[group_sizes >= int(args.min_group_rows)].index.tolist())
    filtered_out_groups = int(len(group_sizes) - len(valid_group_ids))
    df = df.loc[df['split_group_id'].isin(valid_group_ids)].copy()

    total_rows_after = len(df)
    all_group_ids = sorted(pd.unique(df['split_group_id']))
    print(f'total rows after filtering: {total_rows_after}')
    print(f'number of groups: {len(all_group_ids)}')
    if filtered_out_groups:
        print(f'filtered groups below min-group-rows={args.min_group_rows}: {filtered_out_groups}')
    train_ids, val_ids, test_ids = split_group_ids(all_group_ids, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    df['split'] = 'test'
    df.loc[df['split_group_id'].isin(train_ids), 'split'] = 'train'
    df.loc[df['split_group_id'].isin(val_ids), 'split'] = 'val'
    df = df[sort_columns_for_output(df)].copy()

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
        'input_mode': input_mode,
        'input_dir': str(args.input_dir) if args.input_dir is not None else None,
        'rounds_dataset_dir': str(args.rounds_dataset_dir) if args.rounds_dataset_dir is not None else None,
        'input_glob': args.input_glob,
        'map': args.map,
        'feature_mode': args.feature_mode,
        'split_unit': args.split_unit,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'min_group_rows': int(args.min_group_rows),
        'auto_label_grid': bool(args.auto_label_grid),
        'require_grid_labels': bool(args.require_grid_labels),
        'lookahead_ticks': int(args.lookahead_ticks),
        'min_target_distance': float(args.min_target_distance),
        'total_rows': int(len(df)),
        'train_rows': int(len(train_df)),
        'val_rows': int(len(val_df)),
        'test_rows': int(len(test_df)),
        'total_groups': int(len(all_group_ids)),
        'train_groups': int(len(train_ids)),
        'val_groups': int(len(val_ids)),
        'test_groups': int(len(test_ids)),
        'total_rounds': int(df['split_round_id'].nunique()) if 'split_round_id' in df.columns else 0,
        'total_demos': int(df['demo_file_name'].nunique()) if 'demo_file_name' in df.columns else 0,
        'required_grid_columns': list(GRID_NAVIGATION_REQUIRED_COLUMNS) if require_grid else [],
        'movement_target_columns': target_column_map,
        'position_columns': list(position_columns),
        'velocity_columns': list(velocity_columns),
        'yaw_column': yaw_column,
        'group_columns': group_columns,
        'source_files': [str(path) for path in source_files],
        'input_metadata': input_metadata,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding='utf-8')

    print(f'split rows/groups: train={len(train_df)}/{len(train_ids)} val={len(val_df)}/{len(val_ids)} test={len(test_df)}/{len(test_ids)}')
    print_action_distribution(train_df, prefix='train')
    print_grid_stats(train_df, required_grid=require_grid)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
