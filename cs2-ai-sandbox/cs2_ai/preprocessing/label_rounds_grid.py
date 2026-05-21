from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cs2_ai.preprocessing.build_movement_trainset import (
    discover_round_parquet_files,
    extract_round_number,
    label_grid_navigation_dataframe,
)


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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch-label round parquet files with grid navigation targets.')
    parser.add_argument('--rounds-dataset-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--map', type=str, default='de_dust2')
    parser.add_argument('--lookahead-ticks', type=int, default=10)
    parser.add_argument('--min-target-distance', type=float, default=75.0)
    return parser.parse_args()


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def summarize_round(round_file: Path, df: pd.DataFrame, demo_dir_name: str) -> dict[str, Any]:
    tick_series = pd.to_numeric(df['tick'], errors='coerce') if 'tick' in df.columns else pd.Series(dtype='float64')
    return {
        'source_file': demo_dir_name,
        'demo_dir': demo_dir_name,
        'round_number': extract_round_number(round_file),
        'output_file': str(Path('rounds') / round_file.name),
        'row_count': int(len(df)),
        'unique_tick_count': int(df['tick'].nunique()) if 'tick' in df.columns else 0,
        'player_count': int(df['steamid'].nunique()) if 'steamid' in df.columns else 0,
        'tick_min': int(tick_series.min()) if not tick_series.empty and not pd.isna(tick_series.min()) else 0,
        'tick_max': int(tick_series.max()) if not tick_series.empty and not pd.isna(tick_series.max()) else 0,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(path, index=False)


def copy_or_write_demo_manifest(
    input_demo_dir: Path,
    output_demo_dir: Path,
    summary_rows: list[dict[str, Any]],
    *,
    rounds_dataset_dir: Path,
    output_dir: Path,
) -> None:
    original = read_optional_json(input_demo_dir / 'manifest.json')
    if original is not None:
        payload = dict(original)
    else:
        payload = {}
    payload['grid_labeled'] = True
    payload['grid_labeling'] = {
        'source_rounds_dataset_dir': str(rounds_dataset_dir),
        'output_dir': str(output_dir),
        'round_files_count': int(len(summary_rows)),
    }
    write_json(output_demo_dir / 'manifest.json', payload)


def main() -> int:
    args = parse_args()
    round_files = discover_round_parquet_files(args.rounds_dataset_dir)
    if not round_files:
        raise FileNotFoundError(f'No round parquet files found for {args.rounds_dataset_dir} / */rounds/round_*.parquet')

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global_summary_rows: list[dict[str, Any]] = []
    demo_summary_map: dict[str, list[dict[str, Any]]] = {}

    for round_file in round_files:
        demo_dir_name = round_file.parents[1].name
        input_demo_dir = round_file.parents[1]
        output_demo_dir = output_dir / demo_dir_name
        output_round_path = output_demo_dir / 'rounds' / round_file.name

        frame = pd.read_parquet(round_file)
        labeled = label_grid_navigation_dataframe(
            frame,
            map_name=args.map,
            lookahead_ticks=args.lookahead_ticks,
            min_target_distance=args.min_target_distance,
        )
        output_round_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_parquet(output_round_path, index=False)

        summary_row = summarize_round(round_file, labeled, demo_dir_name)
        global_summary_rows.append(summary_row)
        demo_summary_map.setdefault(demo_dir_name, []).append(summary_row)
        print(f'Labeled {round_file} -> {output_round_path}')

        copy_or_write_demo_manifest(
            input_demo_dir,
            output_demo_dir,
            demo_summary_map[demo_dir_name],
            rounds_dataset_dir=args.rounds_dataset_dir,
            output_dir=output_dir,
        )

    for demo_dir_name, rows in demo_summary_map.items():
        write_summary_csv(output_dir / demo_dir_name / 'rounds_summary.csv', rows)

    original_manifest = read_optional_json(args.rounds_dataset_dir / 'manifest.json')
    if original_manifest is not None:
        top_manifest = dict(original_manifest)
    else:
        top_manifest = {}
    top_manifest.update(
        {
            'created_at': datetime.now(timezone.utc).isoformat(),
            'source_rounds_dataset_dir': str(args.rounds_dataset_dir),
            'output_dir': str(output_dir),
            'grid_labeled': True,
            'grid_labeling': {
                'map': args.map,
                'lookahead_ticks': int(args.lookahead_ticks),
                'min_target_distance': float(args.min_target_distance),
            },
            'total_output_demo_dirs': int(len(demo_summary_map)),
            'total_output_round_files': int(len(global_summary_rows)),
        }
    )
    write_json(output_dir / 'manifest.json', top_manifest)
    write_summary_csv(output_dir / 'rounds_summary.csv', global_summary_rows)
    print(f'Wrote labeled rounds dataset: {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
