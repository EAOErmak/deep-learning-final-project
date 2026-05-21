from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
scripts_dir = PROJECT_ROOT / 'scripts'
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import pandas as pd

import build_rounds_dataset
from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset


def make_demo_df(demo_name: str, round_tick_counts: dict[int, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tick_cursor = 1
    for round_number, tick_count in round_tick_counts.items():
        for tick in range(tick_cursor, tick_cursor + tick_count):
            for player_idx, steamid in enumerate((101, 202), start=1):
                rows.append(
                    {
                        'demo_name': demo_name,
                        'round_number': round_number,
                        'total_rounds_played': round_number,
                        'tick': tick,
                        'steamid': steamid,
                        'is_alive': True,
                        'team_num': 3 if player_idx == 1 else 2,
                        'X': float(tick),
                        'Y': float(player_idx),
                        'Z': 0.0,
                        'yaw': 0.0,
                        'pitch': 0.0,
                        'FORWARD': tick % 2 == 0,
                        'BACK': False,
                        'LEFT': tick % 3 == 0,
                        'RIGHT': tick % 5 == 0,
                        'FIRE': tick % 7 == 0,
                        'WALK': tick % 2 == 1,
                        'JUMP': tick % 11 == 0,
                    }
                )
        tick_cursor += tick_count
    return pd.DataFrame(rows)


class BuildRoundsDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = PROJECT_ROOT / '.cache' / 'test_rounds_dataset' / self._testMethodName
        shutil.rmtree(self.tmp_root, ignore_errors=True)
        (self.tmp_root / 'data' / 'clean_play_ticks').mkdir(parents=True, exist_ok=True)

        demo_a = make_demo_df('demo_a', {1: 20, 2: 18, 3: 4})
        demo_b = make_demo_df('demo_b', {1: 17})
        demo_a.to_parquet(self.tmp_root / 'data' / 'clean_play_ticks' / 'demo_a.parquet', index=False)
        demo_b.to_parquet(self.tmp_root / 'data' / 'clean_play_ticks' / 'demo_b.parquet', index=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def run_builder(self, *extra_args: str) -> int:
        argv_backup = sys.argv[:]
        try:
            sys.argv = [
                'build_rounds_dataset',
                '--data-dir', str(self.tmp_root / 'data'),
                '--input-subdir', 'clean_play_ticks',
                '--output-subdir', 'rounds-dataset',
                *extra_args,
            ]
            return build_rounds_dataset.main()
        finally:
            sys.argv = argv_backup

    def test_build_rounds_dataset_creates_expected_structure_and_manifests(self):
        result = self.run_builder('--overwrite', '--min-ticks-per-round', '16')
        self.assertEqual(result, 0)

        dataset_root = self.tmp_root / 'data' / 'rounds-dataset'
        round_1_path = dataset_root / 'demo_a' / 'rounds' / 'round_1.parquet'
        round_2_path = dataset_root / 'demo_a' / 'rounds' / 'round_2.parquet'
        self.assertTrue(round_1_path.exists())
        self.assertTrue(round_2_path.exists())
        self.assertFalse((dataset_root / 'demo_a' / 'rounds' / 'round_3.parquet').exists())

        round_1_df = pd.read_parquet(round_1_path)
        round_2_df = pd.read_parquet(round_2_path)
        self.assertEqual(set(round_1_df['round_number'].unique().tolist()), {1})
        self.assertEqual(set(round_2_df['round_number'].unique().tolist()), {2})
        self.assertIn('FIRE', round_1_df.columns)
        self.assertIn('WALK', round_2_df.columns)

        demo_manifest = json.loads((dataset_root / 'demo_a' / 'manifest.json').read_text(encoding='utf-8'))
        global_manifest = json.loads((dataset_root / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(demo_manifest['round_files_count'], 2)
        self.assertEqual(len(demo_manifest['skipped_rounds']), 1)
        self.assertEqual(global_manifest['total_output_demo_dirs'], 2)
        self.assertEqual(global_manifest['total_output_round_files'], 3)

        self.assertTrue((dataset_root / 'demo_a' / 'rounds_summary.csv').exists())
        self.assertTrue((dataset_root / 'rounds_summary.csv').exists())

    def test_dry_run_does_not_create_files(self):
        result = self.run_builder('--dry-run', '--min-ticks-per-round', '16')
        self.assertEqual(result, 0)
        self.assertFalse((self.tmp_root / 'data' / 'rounds-dataset').exists())

    def test_rounds_dataset_loader_reads_recursive_layout(self):
        result = self.run_builder('--overwrite', '--min-ticks-per-round', '16')
        self.assertEqual(result, 0)

        dataset = MultiDemoSequenceDataset(
            dataset_dir=self.tmp_root / 'data',
            subdir='rounds-dataset',
            seq_len=4,
            stride=2,
            alive_only=True,
        )
        self.assertGreater(len(dataset), 0)
        demo_names = dataset.get_demo_names()
        self.assertEqual(demo_names, ['demo_a', 'demo_b'])

        metadata = dataset.get_sample_metadata(0)
        self.assertIn(metadata['demo_name'], {'demo_a', 'demo_b'})
        self.assertIn(metadata['round_number'], {1, 2})
        self.assertTrue(str(metadata['parquet_path']).endswith('.parquet'))


if __name__ == '__main__':
    unittest.main()
