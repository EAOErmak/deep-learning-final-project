from __future__ import annotations

import math
from pathlib import Path

import numpy as np
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


def _build_segment_arrays(
    cell_ids: np.ndarray,
    ticks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(cell_ids) == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty, empty
    segment_start_mask = np.ones(len(cell_ids), dtype=bool)
    segment_start_mask[1:] = cell_ids[1:] != cell_ids[:-1]
    segment_start_indices = np.flatnonzero(segment_start_mask).astype(np.int64)
    next_start_indices = np.append(segment_start_indices[1:], len(cell_ids)).astype(np.int64)
    segment_end_indices = (next_start_indices - 1).astype(np.int64)
    row_to_segment_index = np.empty(len(cell_ids), dtype=np.int64)
    for segment_idx, (start_idx, end_idx) in enumerate(zip(segment_start_indices, segment_end_indices, strict=True)):
        row_to_segment_index[start_idx:end_idx + 1] = segment_idx
    return segment_start_indices, segment_end_indices, row_to_segment_index, ticks[segment_start_indices]


def _resolve_target_arrays(
    *,
    positions: np.ndarray,
    cell_ids: np.ndarray,
    cell_centers: np.ndarray,
    segment_start_indices: np.ndarray,
    row_to_segment_index: np.ndarray,
    lookahead_ticks: int,
    min_target_distance: float,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    total_rows = len(cell_ids)
    target_indices = np.arange(total_rows, dtype=np.int64)
    has_target = np.zeros(total_rows, dtype=np.int64)
    if total_rows == 0:
        return target_indices, has_target

    progress_step = max(1, total_rows // 10)
    min_distance_sq = float(min_target_distance) ** 2
    segment_cell_ids = cell_ids[segment_start_indices]
    segment_centers = cell_centers[segment_start_indices]
    segment_count = len(segment_start_indices)
    lookahead_offset = max(int(lookahead_ticks), 1)

    for idx in range(total_rows):
        if idx >= total_rows - 1:
            if progress_label is not None and ((idx + 1) % progress_step == 0 or idx + 1 == total_rows):
                progress_pct = (100.0 * float(idx + 1)) / float(total_rows)
                print(f'    {progress_label}: row {idx + 1}/{total_rows} ({progress_pct:5.1f}%)', flush=True)
            continue

        current_cell_id = int(cell_ids[idx])
        preferred_idx = min(idx + lookahead_offset, total_rows - 1)
        preferred_segment_idx = int(row_to_segment_index[preferred_idx])
        current_segment_idx = int(row_to_segment_index[idx])
        future_segment_slice = slice(current_segment_idx + 1, segment_count)
        future_start_indices = segment_start_indices[future_segment_slice]

        if future_start_indices.size > 0:
            future_cell_ids = segment_cell_ids[future_segment_slice]
            future_centers = segment_centers[future_segment_slice]
            deltas = future_centers - positions[idx]
            distances_sq = np.sum(deltas * deltas, axis=1)
            valid_future_mask = (future_cell_ids != current_cell_id) & (distances_sq >= min_distance_sq)
            if np.any(valid_future_mask):
                candidate_segment_indices = np.flatnonzero(valid_future_mask) + current_segment_idx + 1
                target_segment_idx = int(candidate_segment_indices[0])
                if preferred_segment_idx in candidate_segment_indices:
                    target_segment_idx = int(preferred_segment_idx)
                target_indices[idx] = int(segment_start_indices[target_segment_idx])
                has_target[idx] = 1
            else:
                different_future_mask = future_cell_ids != current_cell_id
                if np.any(different_future_mask):
                    first_different_offset = int(np.flatnonzero(different_future_mask)[0])
                    target_indices[idx] = int(future_start_indices[first_different_offset])
                    has_target[idx] = 1

        if progress_label is not None and ((idx + 1) % progress_step == 0 or idx + 1 == total_rows):
            progress_pct = (100.0 * float(idx + 1)) / float(total_rows)
            print(f'    {progress_label}: row {idx + 1}/{total_rows} ({progress_pct:5.1f}%)', flush=True)

    return target_indices, has_target


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
    progress_label: str | None = None,
) -> pd.DataFrame:
    ordered = df_group.sort_values(tick_column).reset_index(drop=True).copy()
    x_values = pd.to_numeric(ordered[x_col], errors='coerce').to_numpy(dtype=float, copy=False)
    y_values = pd.to_numeric(ordered[y_col], errors='coerce').to_numpy(dtype=float, copy=False)
    z_values = pd.to_numeric(ordered[z_col], errors='coerce').to_numpy(dtype=float, copy=False)
    tick_values = pd.to_numeric(ordered[tick_column], errors='coerce').to_numpy(dtype=np.int64, copy=False)

    current_cells = [grid_map.cell_from_position(x, y, z) for x, y, z in zip(x_values, y_values, z_values, strict=True)]
    ordered['current_cell_id'] = [cell.cell_id for cell in current_cells]
    ordered['cell_ix'] = [cell.ix for cell in current_cells]
    ordered['cell_iy'] = [cell.iy for cell in current_cells]
    ordered['cell_iz'] = [cell.iz for cell in current_cells]
    ordered['cell_center_x'] = [cell.center_x for cell in current_cells]
    ordered['cell_center_y'] = [cell.center_y for cell in current_cells]
    ordered['cell_center_z'] = [cell.center_z for cell in current_cells]
    current_cell_ids = ordered['current_cell_id'].to_numpy(dtype=np.int64, copy=False)
    cell_centers = ordered[['cell_center_x', 'cell_center_y', 'cell_center_z']].to_numpy(dtype=float, copy=False)
    positions = np.column_stack((x_values, y_values, z_values))
    segment_start_indices, segment_end_indices, row_to_segment_index, _ = _build_segment_arrays(current_cell_ids, tick_values)

    target_indices, has_target = _resolve_target_arrays(
        positions=positions,
        cell_ids=current_cell_ids,
        cell_centers=cell_centers,
        segment_start_indices=segment_start_indices,
        row_to_segment_index=row_to_segment_index,
        lookahead_ticks=lookahead_ticks,
        min_target_distance=min_target_distance,
        progress_label=progress_label,
    )
    target_centers = cell_centers[target_indices]
    ordered['next_cell_id'] = current_cell_ids[target_indices]
    ordered['next_cell_center_x'] = target_centers[:, 0]
    ordered['next_cell_center_y'] = target_centers[:, 1]
    ordered['next_cell_center_z'] = target_centers[:, 2]
    ordered['has_next_cell_target'] = has_target

    next_rel = target_centers - positions
    ordered['next_cell_rel_x'] = next_rel[:, 0]
    ordered['next_cell_rel_y'] = next_rel[:, 1]
    ordered['next_cell_rel_z'] = next_rel[:, 2]
    ordered['next_cell_distance'] = np.sqrt(np.sum(next_rel * next_rel, axis=1))

    ordered['segment_start_tick'] = 0
    ordered['segment_end_tick'] = 0
    ordered['dwell_ticks'] = 0
    ordered['dwell_bucket_id'] = 0
    ordered['dwell_pass_through'] = 0.0
    ordered['dwell_short_hold'] = 0.0
    ordered['dwell_medium_hold'] = 0.0
    ordered['dwell_long_hold'] = 0.0

    for segment_index, segment in enumerate(compress_cell_segments(ordered, tick_column=tick_column)):
        start_idx = int(segment_start_indices[segment_index])
        end_idx = int(segment_end_indices[segment_index]) + 1
        ordered.loc[start_idx:end_idx - 1, 'segment_start_tick'] = segment.segment_start_tick
        ordered.loc[start_idx:end_idx - 1, 'segment_end_tick'] = segment.segment_end_tick
        ordered.loc[start_idx:end_idx - 1, 'dwell_ticks'] = segment.dwell_ticks
        ordered.loc[start_idx:end_idx - 1, 'dwell_bucket_id'] = segment.dwell_bucket_id
        ordered.loc[start_idx:end_idx - 1, 'dwell_pass_through'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_PASS_THROUGH else 0.0
        ordered.loc[start_idx:end_idx - 1, 'dwell_short_hold'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_SHORT_HOLD else 0.0
        ordered.loc[start_idx:end_idx - 1, 'dwell_medium_hold'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_MEDIUM_HOLD else 0.0
        ordered.loc[start_idx:end_idx - 1, 'dwell_long_hold'] = 1.0 if segment.dwell_bucket_id == DWELL_BUCKET_LONG_HOLD else 0.0

    return ordered
