from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.navigation.dwell import (
    DWELL_BUCKET_LONG_HOLD,
    DWELL_BUCKET_MEDIUM_HOLD,
    DWELL_BUCKET_PASS_THROUGH,
    DWELL_BUCKET_SHORT_HOLD,
    assign_dwell_bucket,
    compress_cell_segments,
)
from cs2_ai.navigation.grid_config import DUST2_GRID_CONFIG
from cs2_ai.navigation.grid_map import GridMap
from cs2_ai.navigation.path_labeler import label_navigation_for_group


class NavigationGridTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = GridMap(DUST2_GRID_CONFIG)

    def test_negative_coordinates_map_to_non_negative_indices(self):
        ix, iy, iz = self.grid.position_to_indices(-2299.9, -1249.9, -255.9)
        self.assertGreaterEqual(ix, 0)
        self.assertGreaterEqual(iy, 0)
        self.assertGreaterEqual(iz, 0)
        self.assertEqual((ix, iy, iz), (0, 0, 0))

    def test_cell_id_roundtrip(self):
        original = (13, 27, 4)
        cell_id = self.grid.indices_to_cell_id(*original)
        decoded = self.grid.cell_id_to_indices(cell_id)
        self.assertEqual(decoded, original)

    def test_cell_center_calculation(self):
        center = self.grid.cell_center_from_indices(0, 0, 0)
        self.assertEqual(center, (-2287.5, -1237.5, -248.0))

    def test_lookahead_target_uses_future_cell(self):
        df = pd.DataFrame(
            [
                {'tick': 1, 'steamid': 1, 'X': -2200.0, 'Y': -1100.0, 'Z': 0.0},
                {'tick': 2, 'steamid': 1, 'X': -2190.0, 'Y': -1100.0, 'Z': 0.0},
                {'tick': 3, 'steamid': 1, 'X': -2100.0, 'Y': -1100.0, 'Z': 0.0},
                {'tick': 4, 'steamid': 1, 'X': -2000.0, 'Y': -1100.0, 'Z': 0.0},
            ]
        )
        labeled = label_navigation_for_group(
            df,
            grid_map=self.grid,
            lookahead_ticks=1,
            min_target_distance=75.0,
            x_col='X',
            y_col='Y',
            z_col='Z',
            tick_column='tick',
        )
        self.assertEqual(int(labeled.loc[0, 'has_next_cell_target']), 1)
        self.assertNotEqual(int(labeled.loc[0, 'next_cell_id']), int(labeled.loc[0, 'current_cell_id']))

    def test_dwell_bucket_assignment(self):
        self.assertEqual(assign_dwell_bucket(5), DWELL_BUCKET_PASS_THROUGH)
        self.assertEqual(assign_dwell_bucket(25), DWELL_BUCKET_SHORT_HOLD)
        self.assertEqual(assign_dwell_bucket(90), DWELL_BUCKET_MEDIUM_HOLD)
        self.assertEqual(assign_dwell_bucket(120), DWELL_BUCKET_LONG_HOLD)

    def test_compress_cell_segments(self):
        df = pd.DataFrame(
            [
                {'tick': 1, 'current_cell_id': 10},
                {'tick': 2, 'current_cell_id': 10},
                {'tick': 3, 'current_cell_id': 11},
                {'tick': 4, 'current_cell_id': 11},
                {'tick': 5, 'current_cell_id': 11},
            ]
        )
        segments = compress_cell_segments(df)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].dwell_ticks, 2)
        self.assertEqual(segments[1].dwell_ticks, 3)


if __name__ == '__main__':
    unittest.main()
