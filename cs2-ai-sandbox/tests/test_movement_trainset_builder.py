from __future__ import annotations

import sys
import unittest
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.ml.training.train_movement import build_prebuilt_split_dataset
from cs2_ai.preprocessing import build_movement_trainset


def make_builder_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_idx, steamid in enumerate((101, 202, 303, 404), start=1):
        for tick in range(1, 41):
            rows.append(
                {
                    'map_name': 'de_dust2',
                    'demo_name': f'demo_{group_idx // 2}',
                    'total_rounds_played': group_idx,
                    'round_id': group_idx,
                    'steamid': steamid,
                    'team_num': 3,
                    'tick': tick,
                    'X': float(-2200 + tick * 5 + group_idx),
                    'Y': float(-1100 + group_idx),
                    'Z': 0.0,
                    'velocity_X': 10.0,
                    'velocity_Y': 0.0,
                    'velocity_Z': 0.0,
                    'yaw': 0.0,
                    'is_alive': True,
                    'health': 100,
                    'armor_value': 100,
                    'has_helmet': True,
                    'balance': 1000,
                    'active_weapon_name': 'M4A1-S',
                    'active_weapon_ammo': 25,
                    'total_ammo_left': 90,
                    'pitch': 0.0,
                    'is_scoped': False,
                    'is_walking': tick % 2 == 0,
                    'is_airborne': False,
                    'duck_amount': 1.0 if tick % 7 == 0 else 0.0,
                    'ducking': tick % 7 == 0,
                    'shots_fired': 0,
                    'flash_duration': 0.0,
                    'spotted': False,
                    'last_place_name': 'mid',
                    'in_bomb_zone': False,
                    'in_buy_zone': False,
                    'which_bomb_zone': 0,
                    'FORWARD': tick % 3 == 0,
                    'BACK': False,
                    'LEFT': tick % 5 == 0,
                    'RIGHT': tick % 4 == 0,
                    'WALK': tick % 2 == 0,
                    'JUMP': tick % 11 == 0,
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
                    'next_cell_rel_x': 200.0,
                    'next_cell_rel_y': -100.0,
                    'next_cell_rel_z': 32.0,
                    'next_cell_distance': 225.0,
                    'has_next_cell_target': 1.0,
                    'dwell_pass_through': 1.0,
                    'dwell_short_hold': 0.0,
                    'dwell_medium_hold': 0.0,
                    'dwell_long_hold': 0.0,
                }
            )
    return pd.DataFrame(rows)


class MovementTrainsetBuilderTests(unittest.TestCase):
    def test_build_and_load_prebuilt_trainset(self):
        df = make_builder_df()
        tmp_path = PROJECT_ROOT / '.tmp_test_trainset'
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            input_dir = tmp_path / 'processed'
            output_dir = tmp_path / 'trainset'
            input_dir.mkdir(parents=True, exist_ok=True)
            source_path = input_dir / 'clean_play_ticks_grid.parquet'
            df.to_parquet(source_path, index=False)

            argv_backup = sys.argv[:]
            try:
                sys.argv = [
                    'build_movement_trainset',
                    '--input-dir', str(input_dir),
                    '--output-dir', str(output_dir),
                    '--map', 'de_dust2',
                    '--feature-mode', 'solo_grid',
                    '--require-grid-labels', 'true',
                    '--seed', '42',
                    '--min-group-rows', '32',
                ]
                result = build_movement_trainset.main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(result, 0)
            self.assertTrue((output_dir / 'train.parquet').exists())
            self.assertTrue((output_dir / 'val.parquet').exists())
            self.assertTrue((output_dir / 'test.parquet').exists())

            args = type('Args', (), {
                'seq_len': 8,
                'stride': 2,
                'alive_only': True,
                'max_samples': None,
                'show_index_progress': False,
                'target_mode': 'action_chunk',
                'chunk_len': 4,
                'use_grid_navigation_features': False,
                'movement_feature_mode': 'solo_grid',
            })()
            dataset = build_prebuilt_split_dataset(output_dir / 'train.parquet', args)
            features, targets, _ = dataset[0]
            self.assertEqual(features.shape[1], 19)
            self.assertEqual(targets.shape[1], 7)
            self.assertTrue('move_forward' in pd.read_parquet(output_dir / 'train.parquet').columns)
        finally:
            if tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
