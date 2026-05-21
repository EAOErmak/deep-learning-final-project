from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.dataset.multi_demo_sequence_dataset import MultiDemoSequenceDataset


def make_rows(round_number: int, steamids: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tick in range(1, 7):
        for steamid in steamids:
            rows.append(
                {
                    'total_rounds_played': round_number,
                    'tick': tick,
                    'steamid': steamid,
                    'is_alive': True,
                    'team_num': 3 if steamid % 2 else 2,
                    'name': f'p{steamid}',
                    'X': float(100 + tick + steamid),
                    'Y': float(50 + tick),
                    'Z': 0.0,
                    'velocity_X': 0.0,
                    'velocity_Y': 0.0,
                    'velocity_Z': 0.0,
                    'health': 100,
                    'armor_value': 100,
                    'has_helmet': True,
                    'balance': 1000,
                    'active_weapon_name': 'AK-47',
                    'active_weapon_ammo': 30,
                    'total_ammo_left': 90,
                    'pitch': 0.0,
                    'yaw': 0.0,
                    'is_scoped': False,
                    'is_walking': False,
                    'is_airborne': False,
                    'duck_amount': 0.0,
                    'ducking': False,
                    'shots_fired': 0,
                    'flash_duration': 0.0,
                    'spotted': False,
                    'last_place_name': 'mid',
                    'in_bomb_zone': False,
                    'in_buy_zone': False,
                    'which_bomb_zone': 0,
                    'FORWARD': False,
                    'BACK': False,
                    'LEFT': False,
                    'RIGHT': False,
                    'FIRE': False,
                    'RIGHTCLICK': False,
                    'RELOAD': False,
                    'USE': False,
                    'ZOOM': False,
                    'WALK': False,
                    'usercmd_mouse_dx': 0.0,
                    'usercmd_mouse_dy': 0.0,
                    'usercmd_forward_move': 0.0,
                    'usercmd_left_move': 0.0,
                    'round_start_time': 0.0,
                    'round_in_progress': True,
                    'is_freeze_period': False,
                    'is_warmup_period': False,
                    'game_phase': 0,
                    'round_win_status': 0,
                    'round_win_reason': 0,
                    'ct_losing_streak': 0,
                    't_losing_streak': 0,
                    'is_bomb_planted': False,
                    'is_bomb_dropped': False,
                }
            )
    return pd.DataFrame(rows)


class RoundsDatasetLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = PROJECT_ROOT / '.tmp_test_rounds_loader'
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_flat_clean_play_ticks_mock_works(self):
        clean_dir = self.tmp_root / 'data' / 'clean_play_ticks'
        clean_dir.mkdir(parents=True, exist_ok=True)
        make_rows(3, (1, 2)).to_parquet(clean_dir / 'demo_a_play_ticks.parquet', index=False)
        dataset = MultiDemoSequenceDataset(dataset_dir=self.tmp_root / 'data', subdir='clean_play_ticks', seq_len=2, stride=1)
        self.assertGreater(len(dataset), 0)
        metadata = dataset.get_sample_metadata(0)
        self.assertIn('round_uid', metadata)
        self.assertIn('round_file', metadata)

    def test_rounds_dataset_nested_structure_works(self):
        rounds_dir = self.tmp_root / 'data' / 'rounds-dataset' / 'demo_a_play_ticks' / 'rounds'
        rounds_dir.mkdir(parents=True, exist_ok=True)
        make_rows(1, (1, 2)).to_parquet(rounds_dir / 'round_1.parquet', index=False)
        make_rows(2, (1, 2)).to_parquet(rounds_dir / 'round_2.parquet', index=False)
        dataset = MultiDemoSequenceDataset(dataset_dir=self.tmp_root / 'data', subdir='rounds-dataset', seq_len=2, stride=1)
        self.assertEqual(dataset.dataset_format, 'rounds')
        metadata = dataset.get_sample_metadata(0)
        self.assertEqual(metadata['demo_dir'], 'demo_a_play_ticks')
        self.assertTrue(str(metadata['round_file']).startswith('round_'))
        self.assertIn('round_uid', metadata)

    def test_sequence_windows_do_not_cross_round_file(self):
        rounds_dir = self.tmp_root / 'data' / 'rounds-dataset' / 'demo_a_play_ticks' / 'rounds'
        rounds_dir.mkdir(parents=True, exist_ok=True)
        make_rows(1, (1, 2)).to_parquet(rounds_dir / 'round_1.parquet', index=False)
        make_rows(2, (1, 2)).to_parquet(rounds_dir / 'round_2.parquet', index=False)
        dataset = MultiDemoSequenceDataset(dataset_dir=self.tmp_root / 'data', subdir='rounds-dataset', seq_len=2, stride=1)
        round_files = {dataset.get_sample_metadata(idx)['round_file'] for idx in range(len(dataset))}
        self.assertEqual(round_files, {'round_1.parquet', 'round_2.parquet'})
        for idx in range(len(dataset)):
            metadata = dataset.get_sample_metadata(idx)
            self.assertEqual(metadata['round_file'], f'round_{metadata["round_number"]}.parquet')


if __name__ == '__main__':
    unittest.main()
