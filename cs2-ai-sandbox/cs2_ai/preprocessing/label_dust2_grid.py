from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cs2_ai.navigation.cell_indexer import build_grid_map
from cs2_ai.navigation.path_labeler import (
    label_navigation_for_group,
    resolve_group_columns,
    resolve_position_columns,
    resolve_tick_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Label clean_play_ticks with Dust2 grid navigation targets.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--map', type=str, default='de_dust2')
    parser.add_argument('--lookahead-ticks', type=int, default=10)
    parser.add_argument('--min-target-distance', type=float, default=75.0)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.csv':
        df.to_csv(path, index=False)
        return
    df.to_parquet(path, index=False)


def main() -> int:
    args = parse_args()
    grid_map = build_grid_map(args.map)
    df = load_table(args.input)
    x_col, y_col, z_col = resolve_position_columns(df)
    tick_column = resolve_tick_column(df)
    group_columns = resolve_group_columns(df)

    print(f'Input rows: {len(df)}')
    print(f'Position columns: {(x_col, y_col, z_col)}')
    print(f'Tick column: {tick_column}')
    print(f'Group columns: {group_columns if group_columns else ["<single_group>"]}')

    labeled_groups: list[pd.DataFrame] = []
    if group_columns:
        for _, group in df.groupby(group_columns, sort=False, dropna=False):
            labeled_groups.append(
                label_navigation_for_group(
                    group,
                    grid_map=grid_map,
                    lookahead_ticks=args.lookahead_ticks,
                    min_target_distance=args.min_target_distance,
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
                lookahead_ticks=args.lookahead_ticks,
                min_target_distance=args.min_target_distance,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                tick_column=tick_column,
            )
        )

    labeled = pd.concat(labeled_groups, ignore_index=True)
    save_table(labeled, args.output)
    print(f'Wrote labeled grid dataset: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
