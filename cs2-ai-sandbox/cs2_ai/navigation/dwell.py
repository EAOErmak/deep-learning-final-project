from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DWELL_BUCKET_PASS_THROUGH = 0
DWELL_BUCKET_SHORT_HOLD = 1
DWELL_BUCKET_MEDIUM_HOLD = 2
DWELL_BUCKET_LONG_HOLD = 3


@dataclass(frozen=True, slots=True)
class DwellSegment:
    segment_index: int
    segment_start_tick: int
    segment_end_tick: int
    current_cell_id: int
    dwell_ticks: int
    dwell_bucket_id: int


def assign_dwell_bucket(dwell_ticks: int) -> int:
    dwell_ticks = int(dwell_ticks)
    if dwell_ticks <= 10:
        return DWELL_BUCKET_PASS_THROUGH
    if dwell_ticks <= 40:
        return DWELL_BUCKET_SHORT_HOLD
    if dwell_ticks <= 100:
        return DWELL_BUCKET_MEDIUM_HOLD
    return DWELL_BUCKET_LONG_HOLD


def compress_cell_segments(df_group: pd.DataFrame, tick_column: str = 'tick') -> list[DwellSegment]:
    if df_group.empty:
        return []
    if 'current_cell_id' not in df_group.columns:
        raise ValueError("compress_cell_segments requires 'current_cell_id' column.")
    ordered = df_group.sort_values(tick_column).reset_index(drop=True)
    segments: list[DwellSegment] = []
    segment_start = 0
    for idx in range(1, len(ordered) + 1):
        is_break = idx == len(ordered) or int(ordered.loc[idx, 'current_cell_id']) != int(ordered.loc[segment_start, 'current_cell_id'])
        if not is_break:
            continue
        start_tick = int(ordered.loc[segment_start, tick_column])
        end_tick = int(ordered.loc[idx - 1, tick_column])
        dwell_ticks = int(idx - segment_start)
        bucket_id = assign_dwell_bucket(dwell_ticks)
        segments.append(
            DwellSegment(
                segment_index=len(segments),
                segment_start_tick=start_tick,
                segment_end_tick=end_tick,
                current_cell_id=int(ordered.loc[segment_start, 'current_cell_id']),
                dwell_ticks=dwell_ticks,
                dwell_bucket_id=bucket_id,
            )
        )
        segment_start = idx
    return segments
