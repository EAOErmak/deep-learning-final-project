from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.dataset.sequence_dataset import PerspectiveSequenceDataset
from cs2_ai.features.movement_features import (
    GRID_NAVIGATION_FEATURE_NAMES,
    MOVEMENT_FEATURE_MODE_SOLO_GRID,
    MOVEMENT_FEATURE_NAMES_SOLO_GRID,
    MOVEMENT_TARGET_MODE_ACTION_CHUNK,
    MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE,
)
from cs2_ai.ml.training.train_movement import MovementSequenceTorchDataset


def make_row(*, tick: int, steamid: int, team_num: int, forward: bool = False, back: bool = False, left: bool = False, right: bool = False, walk: bool = False, ducking: bool = False, jump: bool = False) -> dict[str, object]:
    return {
        "steamid": steamid,
        "name": f"p{steamid}",
        "team_num": team_num,
        "tick": tick,
        "X": float(steamid * 10 + tick),
        "Y": 0.0,
        "Z": 0.0,
        "velocity_X": 0.0,
        "velocity_Y": 0.0,
        "velocity_Z": 0.0,
        "health": 100,
        "armor_value": 100,
        "has_helmet": True,
        "is_alive": True,
        "balance": 1000,
        "active_weapon_name": "M4A1-S" if team_num == 3 else "AK-47",
        "active_weapon_ammo": 25,
        "total_ammo_left": 90,
        "pitch": 0.0,
        "yaw": 0.0,
        "is_scoped": False,
        "is_walking": walk,
        "is_airborne": jump,
        "duck_amount": 1.0 if ducking else 0.0,
        "ducking": ducking,
        "shots_fired": 0,
        "flash_duration": 0.0,
        "spotted": steamid != 1,
        "last_place_name": "mid",
        "in_bomb_zone": False,
        "in_buy_zone": False,
        "which_bomb_zone": 0,
        "FORWARD": forward,
        "BACK": back,
        "LEFT": left,
        "RIGHT": right,
        "FIRE": False,
        "RIGHTCLICK": False,
        "RELOAD": False,
        "USE": False,
        "ZOOM": False,
        "WALK": walk,
        "JUMP": jump,
        "usercmd_mouse_dx": 0.0,
        "usercmd_mouse_dy": 0.0,
        "usercmd_forward_move": 0.0,
        "usercmd_left_move": 0.0,
        "round_start_time": 0.0,
        "total_rounds_played": 1 if tick <= 8 else 2,
        "round_in_progress": True,
        "is_freeze_period": False,
        "is_warmup_period": False,
        "game_phase": 0,
        "round_win_status": 0,
        "round_win_reason": 0,
        "ct_losing_streak": 0,
        "t_losing_streak": 0,
        "is_bomb_planted": False,
        "is_bomb_dropped": False,
    }


def build_tick_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    action_pattern = {
        5: dict(forward=True),
        6: dict(right=True, walk=True),
        7: dict(left=True, ducking=True),
        8: dict(back=True, jump=True),
        9: dict(forward=True, jump=True),
        10: dict(right=True),
    }
    for tick in range(1, 11):
        actions = action_pattern.get(tick, {})
        rows.append(make_row(tick=tick, steamid=1, team_num=3, **actions))
        rows.append(make_row(tick=tick, steamid=2, team_num=2))
    return pd.DataFrame(rows)


class MovementActionChunkTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        base_dataset = PerspectiveSequenceDataset(build_tick_df(), seq_len=4, stride=1, alive_only=True)
        self.base_dataset = base_dataset

    def test_old_target_mode_not_broken(self):
        dataset = MovementSequenceTorchDataset(self.base_dataset, target_mode=MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE, chunk_len=4)
        features, targets, _ = dataset[0]
        self.assertEqual(features.shape, (4, dataset.feature_extractor.feature_dim()))
        self.assertEqual(targets.shape, (4, 6))

    def test_action_chunk_target_shape(self):
        dataset = MovementSequenceTorchDataset(self.base_dataset, target_mode=MOVEMENT_TARGET_MODE_ACTION_CHUNK, chunk_len=4)
        features, targets, _ = dataset[0]
        self.assertEqual(features.shape, (4, dataset.feature_extractor.feature_dim()))
        self.assertEqual(targets.shape, (4, 7))

    def test_action_chunk_does_not_cross_round(self):
        dataset = MovementSequenceTorchDataset(self.base_dataset, target_mode=MOVEMENT_TARGET_MODE_ACTION_CHUNK, chunk_len=4)
        for idx in range(len(dataset)):
            metadata = dataset.get_sample_metadata(idx)
            target_ticks = dataset._resolve_target_ticks(metadata)
            self.assertIsNotNone(target_ticks)
            self.assertEqual(len(target_ticks), 4)
            self.assertTrue(all(tick <= 8 or tick >= 9 for tick in target_ticks))
            if int(metadata['round_number']) == 1:
                self.assertTrue(all(tick <= 8 for tick in target_ticks))

    def test_action_chunk_targets_are_binary(self):
        dataset = MovementSequenceTorchDataset(self.base_dataset, target_mode=MOVEMENT_TARGET_MODE_ACTION_CHUNK, chunk_len=4)
        _, targets, _ = dataset[0]
        self.assertTrue(((targets == 0.0) | (targets == 1.0)).all())
        self.assertEqual(float(targets[:, 6].max()), 1.0)

    def test_grid_navigation_features_extend_feature_dim(self):
        df = build_tick_df().copy()
        for column, value in {
            'next_cell_rel_x': 200.0,
            'next_cell_rel_y': -100.0,
            'next_cell_rel_z': 32.0,
            'next_cell_distance': 225.0,
            'has_next_cell_target': 1.0,
            'dwell_pass_through': 1.0,
            'dwell_short_hold': 0.0,
            'dwell_medium_hold': 0.0,
            'dwell_long_hold': 0.0,
        }.items():
            df[column] = value
        dataset = MovementSequenceTorchDataset(
            PerspectiveSequenceDataset(df, seq_len=4, stride=1, alive_only=True),
            target_mode=MOVEMENT_TARGET_MODE_ACTION_CHUNK,
            chunk_len=4,
            use_grid_navigation_features=True,
        )
        features, _, _ = dataset[0]
        self.assertEqual(features.shape[1], len(dataset.feature_extractor.feature_names()))
        self.assertEqual(features.shape[1], len(GRID_NAVIGATION_FEATURE_NAMES) + 37)
        self.assertAlmostEqual(float(features[0, -9]), 0.2, places=6)
        self.assertAlmostEqual(float(features[0, -8]), -0.1, places=6)
        self.assertAlmostEqual(float(features[0, -7]), 0.125, places=6)

    def test_solo_grid_feature_mode_uses_only_expected_features(self):
        df = build_tick_df().copy()
        for column, value in {
            'next_cell_rel_x': 200.0,
            'next_cell_rel_y': -100.0,
            'next_cell_rel_z': 32.0,
            'next_cell_distance': 225.0,
            'has_next_cell_target': 1.0,
            'dwell_pass_through': 1.0,
            'dwell_short_hold': 0.0,
            'dwell_medium_hold': 0.0,
            'dwell_long_hold': 0.0,
        }.items():
            df[column] = value
        dataset = MovementSequenceTorchDataset(
            PerspectiveSequenceDataset(df, seq_len=4, stride=1, alive_only=True),
            target_mode=MOVEMENT_TARGET_MODE_ACTION_CHUNK,
            chunk_len=4,
            movement_feature_mode=MOVEMENT_FEATURE_MODE_SOLO_GRID,
        )
        features, _, _ = dataset[0]
        feature_names = dataset.feature_extractor.feature_names()
        self.assertEqual(feature_names, MOVEMENT_FEATURE_NAMES_SOLO_GRID)
        self.assertEqual(features.shape, (4, 19))
        self.assertTrue(all(not name.startswith('teammate_') for name in feature_names))
        self.assertTrue(all(feature in feature_names for feature in GRID_NAVIGATION_FEATURE_NAMES))
        self.assertAlmostEqual(float(features[0, 0]), float(df.iloc[0]['X']) / 4000.0, places=6)
        self.assertAlmostEqual(float(features[0, 1]), float(df.iloc[0]['Y']) / 4000.0, places=6)
        self.assertAlmostEqual(float(features[0, 2]), float(df.iloc[0]['Z']) / 512.0, places=6)
        self.assertAlmostEqual(float(features[0, 10]), 0.2, places=6)
        self.assertAlmostEqual(float(features[0, 11]), -0.1, places=6)
        self.assertAlmostEqual(float(features[0, 12]), 0.125, places=6)


if __name__ == "__main__":
    unittest.main()
