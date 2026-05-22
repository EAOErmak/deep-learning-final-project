from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.preprocessing import label_rounds_grid


def make_round_df(round_number: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tick in range(1, 21):
        for steamid in (101, 202):
            rows.append(
                {
                    'round_number': round_number,
                    'tick': tick,
                    'steamid': steamid,
                    'X': float(-2200 + tick * 12),
                    'Y': float(-1100 + steamid),
                    'Z': 0.0,
                    'velocity_X': 10.0,
                    'velocity_Y': 0.0,
                    'velocity_Z': 0.0,
                    'yaw': 0.0,
                    'is_walking': tick % 2 == 0,
                    'ducking': False,
                    'is_airborne': False,
                    'FORWARD': tick % 2 == 0,
                    'BACK': False,
                    'LEFT': False,
                    'RIGHT': tick % 3 == 0,
                    'WALK': tick % 2 == 0,
                }
            )
    return pd.DataFrame(rows)


class LabelRoundsGridTests(unittest.TestCase):
    def test_label_rounds_grid_preserves_structure(self):
        self.run_label_rounds_grid(workers='1')

    def test_label_rounds_grid_parallel_preserves_structure(self):
        self.run_label_rounds_grid(workers='2')

    def run_label_rounds_grid(self, *, workers: str) -> None:
        tmp_root = PROJECT_ROOT / '.tmp_test_label_rounds_grid'
        shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            input_root = tmp_root / 'data' / 'rounds-dataset'
            output_root = tmp_root / 'data' / 'rounds-dataset-grid'
            demo_dir = input_root / 'demo_abc' / 'rounds'
            demo_dir.mkdir(parents=True, exist_ok=True)
            make_round_df(1).to_parquet(demo_dir / 'round_1.parquet', index=False)
            make_round_df(2).to_parquet(demo_dir / 'round_2.parquet', index=False)
            (input_root / 'demo_abc' / 'manifest.json').write_text(json.dumps({'source_file_name': 'demo_abc.parquet'}), encoding='utf-8')
            (input_root / 'manifest.json').write_text(json.dumps({'total_output_demo_dirs': 1}), encoding='utf-8')

            argv_backup = sys.argv[:]
            try:
                sys.argv = [
                    'label_rounds_grid',
                    '--rounds-dataset-dir', str(input_root),
                    '--output-dir', str(output_root),
                    '--map', 'de_dust2',
                    '--lookahead-ticks', '10',
                    '--min-target-distance', '75',
                    '--workers', workers,
                ]
                result = label_rounds_grid.main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(result, 0)
            labeled_path = output_root / 'demo_abc' / 'rounds' / 'round_1.parquet'
            self.assertTrue(labeled_path.exists())
            labeled_df = pd.read_parquet(labeled_path)
            self.assertIn('current_cell_id', labeled_df.columns)
            self.assertIn('next_cell_id', labeled_df.columns)
            self.assertTrue((output_root / 'demo_abc' / 'manifest.json').exists())
            self.assertTrue((output_root / 'demo_abc' / 'rounds_summary.csv').exists())
            self.assertTrue((output_root / 'manifest.json').exists())
            self.assertTrue((output_root / 'rounds_summary.csv').exists())
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
