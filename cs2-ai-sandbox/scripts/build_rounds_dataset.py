from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROUND_COLUMN_CANDIDATES = ('round_number', 'total_rounds_played')
REQUIRED_COLUMNS = ('tick', 'steamid')
ACTION_COLUMNS = {
    'has_forward': 'FORWARD',
    'has_back': 'BACK',
    'has_left': 'LEFT',
    'has_right': 'RIGHT',
    'has_fire': 'FIRE',
    'has_walk': 'WALK',
}
JUMP_COLUMNS = ('JUMP', 'jump', 'IN_JUMP', 'usercmd_jump')
SUMMARY_COLUMNS = [
    'source_file',
    'demo_dir',
    'round_number',
    'output_file',
    'row_count',
    'unique_tick_count',
    'player_count',
    'tick_min',
    'tick_max',
    'has_forward',
    'has_back',
    'has_left',
    'has_right',
    'has_fire',
    'has_walk',
    'has_jump_column',
]


@dataclass(slots=True)
class DemoBuildResult:
    manifest: dict[str, Any]
    summary_rows: list[dict[str, Any]]
    skipped_rounds_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build round-based dataset from clean play tick parquet files.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--input-subdir', type=str, default='clean_play_ticks')
    parser.add_argument('--output-subdir', type=str, default='rounds-dataset')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-demos', type=int, default=None)
    parser.add_argument('--min-ticks-per-round', type=int, default=16)
    return parser.parse_args()


def determine_round_column(columns: list[str] | pd.Index) -> str:
    for candidate in ROUND_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError('Cannot build rounds dataset: missing round_number/total_rounds_played column.')


def validate_required_columns(columns: list[str] | pd.Index, parquet_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        missing_text = ', '.join(missing)
        raise ValueError(f'Missing required columns in {parquet_path}: {missing_text}')


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors='coerce').fillna(0)
    return numeric != 0


def action_presence(round_df: pd.DataFrame) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for output_key, column_name in ACTION_COLUMNS.items():
        if column_name not in round_df.columns:
            flags[output_key] = False
            continue
        flags[output_key] = bool(coerce_bool_series(round_df[column_name]).any())
    flags['has_jump_column'] = any(column in round_df.columns for column in JUMP_COLUMNS)
    return flags


def build_round_record(
    round_df: pd.DataFrame,
    *,
    round_number: int,
    source_file: str,
    demo_dir_name: str,
) -> dict[str, Any]:
    sorted_df = round_df.sort_values(['tick', 'steamid']).reset_index(drop=True)
    flags = action_presence(sorted_df)
    output_file = f'rounds/round_{round_number}.parquet'
    record = {
        'source_file': source_file,
        'demo_dir': demo_dir_name,
        'round_number': int(round_number),
        'output_file': output_file,
        'row_count': int(len(sorted_df)),
        'unique_tick_count': int(sorted_df['tick'].nunique()),
        'player_count': int(sorted_df['steamid'].nunique()),
        'tick_min': int(pd.to_numeric(sorted_df['tick'], errors='coerce').min()),
        'tick_max': int(pd.to_numeric(sorted_df['tick'], errors='coerce').max()),
    }
    record.update(flags)
    return record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)


def build_demo_dataset(
    parquet_path: Path,
    *,
    source_subdir: str,
    output_root: Path,
    min_ticks_per_round: int,
    overwrite: bool,
    dry_run: bool,
) -> DemoBuildResult:
    demo_dir_name = parquet_path.stem
    demo_output_dir = output_root / demo_dir_name
    rounds_output_dir = demo_output_dir / 'rounds'

    if demo_output_dir.exists() and not overwrite:
        raise FileExistsError(f'Output demo directory already exists and --overwrite was not provided: {demo_output_dir}')
    if demo_output_dir.exists() and overwrite and not dry_run:
        shutil.rmtree(demo_output_dir)

    df = pd.read_parquet(parquet_path)
    validate_required_columns(df.columns, parquet_path)
    round_column = determine_round_column(df.columns)

    demo_name_column_value = None
    if 'demo_name' in df.columns and not df['demo_name'].dropna().empty:
        demo_name_column_value = str(df['demo_name'].dropna().iloc[0])

    rounds: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped_rounds: list[dict[str, Any]] = []

    grouped = df.groupby(round_column, sort=True)
    for round_value, round_df in grouped:
        round_number = int(round_value)
        record = build_round_record(
            round_df,
            round_number=round_number,
            source_file=parquet_path.name,
            demo_dir_name=demo_dir_name,
        )
        if record['unique_tick_count'] < min_ticks_per_round:
            skipped_rounds.append(
                {
                    'round_number': round_number,
                    'unique_tick_count': record['unique_tick_count'],
                    'reason': f'unique_tick_count < {min_ticks_per_round}',
                }
            )
            continue

        rounds.append(
            {
                'round_number': record['round_number'],
                'output_file': record['output_file'],
                'row_count': record['row_count'],
                'unique_tick_count': record['unique_tick_count'],
                'player_count': record['player_count'],
                'tick_min': record['tick_min'],
                'tick_max': record['tick_max'],
                'has_forward': record['has_forward'],
                'has_back': record['has_back'],
                'has_left': record['has_left'],
                'has_right': record['has_right'],
                'has_fire': record['has_fire'],
                'has_walk': record['has_walk'],
                'has_jump_column': record['has_jump_column'],
            }
        )
        summary_rows.append(record)

        if not dry_run:
            rounds_output_dir.mkdir(parents=True, exist_ok=True)
            round_output_path = demo_output_dir / record['output_file']
            round_df.sort_values(['tick', 'steamid']).reset_index(drop=True).to_parquet(round_output_path, index=False)

    manifest = {
        'source_file': str(parquet_path.relative_to(parquet_path.parents[1])),
        'source_file_name': parquet_path.name,
        'source_file_path': str(parquet_path),
        'source_subdir': source_subdir,
        'output_demo_dir': str(demo_output_dir),
        'demo_name_column_value': demo_name_column_value,
        'round_column': round_column,
        'total_rows': int(len(df)),
        'total_unique_ticks': int(df['tick'].nunique()),
        'round_files_count': int(len(rounds)),
        'skipped_rounds': skipped_rounds,
        'rounds': rounds,
    }

    if not dry_run:
        demo_output_dir.mkdir(parents=True, exist_ok=True)
        write_json(demo_output_dir / 'manifest.json', manifest)
        write_summary_csv(demo_output_dir / 'rounds_summary.csv', summary_rows)

    return DemoBuildResult(manifest=manifest, summary_rows=summary_rows, skipped_rounds_count=len(skipped_rounds))


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    input_dir = data_dir / args.input_subdir
    output_dir = data_dir / args.output_subdir

    if not input_dir.exists():
        raise FileNotFoundError(f'Input dataset subdirectory not found: {input_dir}')

    input_files = sorted(path for path in input_dir.glob('*.parquet') if path.is_file())
    if args.max_demos is not None:
        input_files = input_files[:args.max_demos]
    if not input_files:
        raise FileNotFoundError(f'No parquet files found in {input_dir}')

    demo_manifests: list[dict[str, Any]] = []
    global_summary_rows: list[dict[str, Any]] = []
    skipped_rounds_count = 0

    for parquet_path in input_files:
        result = build_demo_dataset(
            parquet_path,
            source_subdir=args.input_subdir,
            output_root=output_dir,
            min_ticks_per_round=args.min_ticks_per_round,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        manifest = result.manifest
        demo_manifests.append(
            {
                'source_file': manifest['source_file'],
                'demo_dir': parquet_path.stem,
                'round_files_count': manifest['round_files_count'],
                'row_count': manifest['total_rows'],
                'unique_tick_count': manifest['total_unique_ticks'],
            }
        )
        global_summary_rows.extend(result.summary_rows)
        skipped_rounds_count += result.skipped_rounds_count

    global_manifest = {
        'data_dir': str(data_dir),
        'input_subdir': args.input_subdir,
        'output_subdir': args.output_subdir,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'total_input_files': int(len(input_files)),
        'total_output_demo_dirs': int(len(demo_manifests)),
        'total_output_round_files': int(len(global_summary_rows)),
        'total_rows': int(sum(item['row_count'] for item in demo_manifests)),
        'total_unique_rounds': int(len(global_summary_rows)),
        'skipped_rounds_count': int(skipped_rounds_count),
        'demos': demo_manifests,
    }

    print(f'Input demos scanned: {len(input_files)}')
    print(f'Output demo dirs: {len(demo_manifests)}')
    print(f'Round parquet files {"to create" if args.dry_run else "created"}: {len(global_summary_rows)}')
    print(f'Skipped rounds: {skipped_rounds_count}')
    for row in global_summary_rows[:10]:
        print(f'  {row["demo_dir"]} round {row["round_number"]} -> {row["output_file"]} ({row["row_count"]} rows, {row["unique_tick_count"]} ticks)')

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'manifest.json', global_manifest)
    write_summary_csv(output_dir / 'rounds_summary.csv', global_summary_rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
