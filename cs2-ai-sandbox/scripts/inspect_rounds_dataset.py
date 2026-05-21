from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ACTION_COLUMN_GROUPS = {
    'FORWARD': ('FORWARD',),
    'BACK': ('BACK',),
    'LEFT': ('LEFT',),
    'RIGHT': ('RIGHT',),
    'FIRE': ('FIRE',),
    'WALK': ('WALK',),
    'JUMP': ('JUMP', 'jump', 'IN_JUMP', 'usercmd_jump'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect round-based dataset layout and basic action stats.')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--dataset-subdir', type=str, default='rounds-dataset')
    return parser.parse_args()


def list_round_files(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.glob('*/rounds/*.parquet') if path.is_file())


def load_summary_rows(dataset_root: Path) -> pd.DataFrame:
    summary_path = dataset_root / 'rounds_summary.csv'
    if summary_path.exists():
        return pd.read_csv(summary_path)

    rows: list[dict[str, object]] = []
    for round_file in list_round_files(dataset_root):
        df = pd.read_parquet(round_file)
        rows.append(
            {
                'source_file': round_file.parents[1].name,
                'demo_dir': round_file.parents[1].name,
                'round_number': int(round_file.stem.split('_')[-1]),
                'output_file': str(round_file.relative_to(round_file.parents[1])),
                'row_count': int(len(df)),
                'unique_tick_count': int(df['tick'].nunique()) if 'tick' in df.columns else 0,
            }
        )
    return pd.DataFrame(rows)


def inspect_action_ratios(round_files: list[Path]) -> None:
    totals = {key: 0 for key in ACTION_COLUMN_GROUPS}
    total_rows = 0

    for round_file in round_files:
        df = pd.read_parquet(round_file)
        total_rows += len(df)
        for output_name, candidates in ACTION_COLUMN_GROUPS.items():
            for column in candidates:
                if column not in df.columns:
                    continue
                series = df[column]
                if not pd.api.types.is_bool_dtype(series):
                    series = pd.to_numeric(series, errors='coerce').fillna(0) != 0
                else:
                    series = series.fillna(False)
                totals[output_name] += int(series.sum())
                break

    if total_rows == 0:
        print('Action positive ratios: dataset is empty')
        return

    print('Action positive ratios:')
    for output_name, positive_count in totals.items():
        ratio = float(positive_count / total_rows)
        print(f'  {output_name}: {ratio:.6f}')


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    dataset_root = args.data_dir / args.dataset_subdir
    if not dataset_root.exists():
        raise FileNotFoundError(f'Dataset subdirectory not found: {dataset_root}')

    round_files = list_round_files(dataset_root)
    demo_dirs = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    summary_df = load_summary_rows(dataset_root)

    print(f'Demo folders: {len(demo_dirs)}')
    print(f'Round parquet files: {len(round_files)}')
    print('First 10 demo_dir:')
    for demo_dir in demo_dirs[:10]:
        print(f'  {demo_dir}')

    if not summary_df.empty:
        largest = summary_df.sort_values('row_count', ascending=False).head(10)
        smallest = summary_df.sort_values('unique_tick_count', ascending=True).head(10)
        print('Top 10 largest rounds by row_count:')
        for row in largest.itertuples(index=False):
            print(f'  {row.demo_dir} round {row.round_number}: rows={row.row_count} ticks={row.unique_tick_count}')
        print('Top 10 smallest rounds by unique_tick_count:')
        for row in smallest.itertuples(index=False):
            print(f'  {row.demo_dir} round {row.round_number}: ticks={row.unique_tick_count} rows={row.row_count}')
    else:
        print('No summary rows found.')

    inspect_action_ratios(round_files)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
