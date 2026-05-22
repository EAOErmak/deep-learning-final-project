from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1))))
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


def resolve_workers(requested_workers: int) -> int:
    if int(requested_workers) < 1:
        raise ValueError(f'--workers must be >= 1, got {requested_workers}')
    return int(requested_workers)


def label_single_round_file(
    *,
    round_file: str,
    output_dir: str,
    map_name: str,
    lookahead_ticks: int,
    min_target_distance: float,
    show_inner_progress: bool,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    round_path = Path(round_file)
    output_root = Path(output_dir)
    demo_dir_name = round_path.parents[1].name
    output_round_path = output_root / demo_dir_name / 'rounds' / round_path.name

    frame = pd.read_parquet(round_path)
    tick_count = int(frame['tick'].nunique()) if 'tick' in frame.columns else 0
    player_count = int(frame['steamid'].nunique()) if 'steamid' in frame.columns else 0
    labeled = label_grid_navigation_dataframe(
        frame,
        map_name=map_name,
        lookahead_ticks=lookahead_ticks,
        min_target_distance=min_target_distance,
        progress_label=f'{demo_dir_name}/{round_path.name}' if show_inner_progress else None,
    )
    output_round_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(output_round_path, index=False)
    summary_row = summarize_round(round_path, labeled, demo_dir_name)
    elapsed_seconds = time.perf_counter() - started_at
    return {
        'round_file': str(round_path),
        'demo_dir_name': demo_dir_name,
        'output_round_path': str(output_round_path),
        'summary_row': summary_row,
        'elapsed_seconds': float(elapsed_seconds),
        'rows': int(len(frame)),
        'tick_count': tick_count,
        'player_count': player_count,
    }


def process_rounds_sequential(args: argparse.Namespace, round_files: list[Path]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    total_files = len(round_files)
    global_summary_rows: list[dict[str, Any]] = []
    demo_summary_map: dict[str, list[dict[str, Any]]] = {}

    for index, round_file in enumerate(round_files, start=1):
        demo_dir_name = round_file.parents[1].name
        progress_pct = (100.0 * index) / float(total_files)
        print(f'[{index}/{total_files} | {progress_pct:6.2f}%] Labelling {demo_dir_name}/{round_file.name} ...', flush=True)
        result = label_single_round_file(
            round_file=str(round_file),
            output_dir=str(args.output_dir),
            map_name=args.map,
            lookahead_ticks=args.lookahead_ticks,
            min_target_distance=args.min_target_distance,
            show_inner_progress=True,
        )
        print(
            f'[{index}/{total_files} | {progress_pct:6.2f}%] '
            f'Rows={result["rows"]} ticks={result["tick_count"]} players={result["player_count"]}',
            flush=True,
        )
        global_summary_rows.append(result['summary_row'])
        demo_summary_map.setdefault(result['demo_dir_name'], []).append(result['summary_row'])
        print(
            f'[{index}/{total_files} | {progress_pct:6.2f}%] '
            f'Wrote {result["output_round_path"]} in {result["elapsed_seconds"]:.1f}s',
            flush=True,
        )
    return global_summary_rows, demo_summary_map


def process_rounds_parallel(args: argparse.Namespace, round_files: list[Path], workers: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    total_files = len(round_files)
    completed = 0
    global_summary_rows: list[dict[str, Any]] = []
    demo_summary_map: dict[str, list[dict[str, Any]]] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                label_single_round_file,
                round_file=str(round_file),
                output_dir=str(args.output_dir),
                map_name=args.map,
                lookahead_ticks=args.lookahead_ticks,
                min_target_distance=args.min_target_distance,
                show_inner_progress=False,
            ): round_file
            for round_file in round_files
        }
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            progress_pct = (100.0 * completed) / float(total_files)
            global_summary_rows.append(result['summary_row'])
            demo_summary_map.setdefault(result['demo_dir_name'], []).append(result['summary_row'])
            print(
                f'[{completed}/{total_files} | {progress_pct:6.2f}%] '
                f'Wrote {result["output_round_path"]} '
                f'(rows={result["rows"]} ticks={result["tick_count"]} players={result["player_count"]}) '
                f'in {result["elapsed_seconds"]:.1f}s',
                flush=True,
            )
    return global_summary_rows, demo_summary_map


def main() -> int:
    args = parse_args()
    round_files = discover_round_parquet_files(args.rounds_dataset_dir)
    if not round_files:
        raise FileNotFoundError(f'No round parquet files found for {args.rounds_dataset_dir} / */rounds/round_*.parquet')

    workers = resolve_workers(args.workers)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    total_files = len(round_files)
    print(
        f'Found {total_files} round parquet files in {args.rounds_dataset_dir}. '
        f'Writing labeled dataset to {output_dir}. Workers={workers}.',
        flush=True,
    )

    if workers == 1:
        global_summary_rows, demo_summary_map = process_rounds_sequential(args, round_files)
    else:
        try:
            global_summary_rows, demo_summary_map = process_rounds_parallel(args, round_files, workers)
        except (PermissionError, OSError) as exc:
            print(
                f'WARNING: failed to start multiprocessing workers ({exc}). Falling back to workers=1.',
                flush=True,
            )
            global_summary_rows, demo_summary_map = process_rounds_sequential(args, round_files)

    for demo_dir_name, rows in demo_summary_map.items():
        rows.sort(key=lambda item: str(item['round_number']))
        write_summary_csv(output_dir / demo_dir_name / 'rounds_summary.csv', rows)
        copy_or_write_demo_manifest(
            args.rounds_dataset_dir / demo_dir_name,
            output_dir / demo_dir_name,
            rows,
            rounds_dataset_dir=args.rounds_dataset_dir,
            output_dir=output_dir,
        )

    global_summary_rows.sort(key=lambda item: (str(item['demo_dir']), str(item['round_number'])))
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
                'workers': workers,
            },
            'total_output_demo_dirs': int(len(demo_summary_map)),
            'total_output_round_files': int(len(global_summary_rows)),
        }
    )
    write_json(output_dir / 'manifest.json', top_manifest)
    write_summary_csv(output_dir / 'rounds_summary.csv', global_summary_rows)
    print(f'Wrote labeled rounds dataset: {output_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
