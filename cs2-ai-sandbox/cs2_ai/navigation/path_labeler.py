from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from cs2_ai.navigation.dwell import (
    DWELL_BUCKET_LONG_HOLD,
    DWELL_BUCKET_MEDIUM_HOLD,
    DWELL_BUCKET_PASS_THROUGH,
    DWELL_BUCKET_SHORT_HOLD,
    compress_cell_segments,
)
from cs2_ai.navigation.grid_map import GridMap


POSITION_COLUMN_CANDIDATES = [
    ('X', 'Y', 'Z'),
    ('x', 'y', 'z'),
    ('pos_x', 'pos_y', 'pos_z'),
    ('self_x', 'self_y', 'self_z'),
    ('player_x', 'player_y', 'player_z'),
]

GROUP_COLUMN_CANDIDATES = [
    ('demo_id', 'demo_name'),
    ('round_id', 'round_number', 'total_rounds_played'),
    ('player_id', 'steam_id', 'steamid', 'perspective_steamid'),
]

TICK_COLUMN_CANDIDATES = ('tick', 'tick_id', 'target_tick')


def resolve_position_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    columns = set(df.columns)
    for candidate in POSITION_COLUMN_CANDIDATES:
        if all(name in columns for name in candidate):
            return candidate
    raise ValueError(
        'Could not resolve position columns. Expected one of: '
        'x/y/z, X/Y/Z, pos_x/pos_y/pos_z, self_x/self_y/self_z, player_x/player_y/player_z.'
    )


def resolve_tick_column(df: pd.DataFrame) -> str:
    for name in TICK_COLUMN_CANDIDATES:
        if name in df.columns:
            return name
    raise ValueError(f'Could not resolve tick column from candidates: {TICK_COLUMN_CANDIDATES!r}')


def resolve_group_columns(df: pd.DataFrame) -> list[str]:
    resolved: list[str] = []
    for candidate_group in GROUP_COLUMN_CANDIDATES:
        for column in candidate_group:
            if column in df.columns:
                resolved.append(column)
                break
    return resolved


def distance_3d(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    return float(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2))


def choose_target_index(
    group: pd.DataFrame,
    current_idx: int,
    *,
    lookahead_ticks: int,
    min_target_distance: float,
    x_col: str,
    y_col: str,
    z_col: str,
) -> tuple[int, int]:
    if current_idx >= len(group) - 1:
        return current_idx, 0

    current_row = group.iloc[current_idx]
    current_cell_id = int(current_row['current_cell_id'])
    preferred_idx = min(current_idx + max(int(lookahead_ticks), 1), len(group) - 1)

    candidate_indices = [preferred_idx] + [idx for idx in range(current_idx + 1, len(group)) if idx != preferred_idx]
    for idx in candidate_indices:
        candidate = group.iloc[idx]
        distance = distance_3d(
            float(current_row[x_col]),
            float(current_row[y_col]),
            float(current_row[z_col]),
            float(candidate['cell_center_x']),
            float(candidate['cell_center_y']),
            float(candidate['cell_center_z']),
        )
        if int(candidate['current_cell_id']) != current_cell_id and distance >= float(min_target_distance):
            return idx, 1

    for idx in range(current_idx + 1, len(group)):
        candidate = group.iloc[idx]
        if int(candidate['current_cell_id']) != current_cell_id:
            return idx, 1

    return current_idx, 0


def label_navigation_for_group(
    df_group: pd.DataFrame,
    *,
    grid_map: GridMap,
    lookahead_ticks: int,
    min_target_distance: float,
    x_col: str,
    y_col: str,
    z_col: str,
    tick_column: str,
) -> pd.DataFrame:
    ordered = df_group.sort_values(tick_column).reset_index(drop=True).copy()

    current_cells = [grid_map.cell_from_position(row[x_col], row[y_col], row[z_col]) for _, row in ordered.iterrows()]
    ordered['current_cell_id'] = [cell.cell_id for cell in current_cells]
    ordered['cell_ix'] = [cell.ix for cell in current_cells]
    ordered['cell_iy'] = [cell.iy for cell in current_cells]
    ordered['cell_iz'] = [cell.iz for cell in current_cells]
    ordered['cell_center_x'] = [cell.center_x for cell in current_cells]
    ordered['cell_center_y'] = [cell.center_y for cell in current_cells]
    ordered['cell_center_z'] = [cell.center_z for cell in current_cells]

    ordered['next_cell_id'] = ordered['current_cell_id']
    ordered['next_cell_center_x'] = ordered['cell_center_x']
    ordered['next_cell_center_y'] = ordered['cell_center_y']
    ordered['next_cell_center_z'] = ordered['cell_center_z']
    ordered['has_next_cell_target'] = 0

    for idx in range(len(ordered)):
        target_idx, has_target = choose_target_index(
            ordered,
            idx,
            lookahead_ticks=lookahead_ticks,
            min_target_distance=min_target_distance,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
        )
        target_row = ordered.iloc[target_idx]
        ordered.loc[idx, 'next_cell_id'] = int(target_row['current_cell_id'])
        ordered.loc[idx, 'next_cell_center_x'] = float(target_row['cell_center_x'])
        ordered.loc[idx, 'next_cell_center_y'] = float(target_row['cell_center_y'])
        ordered.loc[idx, 'next_cell_center_z'] = float(target_row['cell_center_z'])
        ordered.loc[idx, 'has_next_cell_target'] = int(has_target)

    ordered['next_cell_rel_x'] = ordered['next_cell_center_x'] - ordered[x_col].astype(float)
    ordered['next_cell_rel_y'] = ordered['next_cell_center_y'] - ordered[y_col].astype(float)
    ordered['next_cell_rel_z'] = ordered['next_cell_center_z'] - ordered[z_col].astype(float)
    ordered['next_cell_distance'] = (
        (ordered['next_cell_rel_x'] ** 2 + ordered['next_cell_rel_y'] ** 2 + ordered['next_cell_rel_z'] ** 2) ** 0.5
    )

    ordered['segment_start_tick'] = 0
    ordered['segment_end_tick'] = 0
    ordered['dwell_ticks'] = 0
    ordered['dwell_bucket_id'] = 0
    ordered['dwell_pass_through'] = 0.0
    ordered['dwell_short_hold'] = 0.0
    ordered['dwell_medium_hold'] = 0.0
    ordered['dwell_long_hold'] = 0.0

    for segment in compress_cell_segments(ordered, tick_column=tick_column):
        mask = (ordered[tick_column].astype(int) >= segment.segment_start_tick) & (ordered[tick_column].astype(int) <= segment.segment_end_tick)
        ordered.loc[mask, 'segment_start_tick'] = segment.segment_start_tick
        ordered.loc[mask, 'segment_end_tick'] = segment.segment_end_tick
        ordered.loc[mask, 'dwell_ticks'] = segment.dwell_ticks
        ordered.loc[mask, 'dwell_bucket_id'] = segment.dwell_bucket_id
        ordered.loc[mask, 'dwell_pass_through'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_PASS_THROUGH else 0.0
        ordered.loc[mask, 'dwell_short_hold'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_SHORT_HOLD else 0.0
        ordered.loc[mask, 'dwell_medium_hold'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_MEDIUM_HOLD else 0.0
        ordered.loc[mask, 'dwell_long_hold'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_LONG_HOLD else 0.0

    return ordered
