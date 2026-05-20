from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.features.aim_features import AIM_FEATURE_NAMES, AimFeatureExtractor
from cs2_ai.features.enemy_tracker_features import TRACKER_FEATURE_NAMES, EnemyTrackerFeatureExtractor, build_enemy_confidence_target, build_enemy_position_target, build_enemy_roster
from cs2_ai.features.feature_contract import validate_checkpoint_schema
from cs2_ai.features.movement_features import MOVEMENT_FEATURE_NAMES, MovementFeatureExtractor
from cs2_ai.schemas.game_state import GameStateSequence, VisibilityStatus
from cs2_ai.state.game_state_builder import GameStateBuilder
from neural_runtime_agent import FullNeuralRuntimeAgent


def make_tick_rows(visible_enemy: bool = True, tick: int = 100) -> pd.DataFrame:
    rows = [
        {
            "steamid": 1,
            "name": "self",
            "team_num": 3,
            "tick": tick,
            "X": 0.0,
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
            "active_weapon_name": "M4A1-S",
            "active_weapon_ammo": 25,
            "total_ammo_left": 90,
            "pitch": 0.0,
            "yaw": 0.0,
            "is_scoped": False,
            "is_walking": False,
            "is_airborne": False,
            "duck_amount": 0.0,
            "ducking": False,
            "shots_fired": 0,
            "flash_duration": 0.0,
            "spotted": True,
            "last_place_name": "mid",
            "in_bomb_zone": False,
            "in_buy_zone": False,
            "which_bomb_zone": 0,
            "FORWARD": False,
            "BACK": False,
            "LEFT": False,
            "RIGHT": False,
            "FIRE": False,
            "RIGHTCLICK": False,
            "RELOAD": False,
            "USE": False,
            "ZOOM": False,
            "WALK": False,
            "usercmd_mouse_dx": 0.0,
            "usercmd_mouse_dy": 0.0,
            "usercmd_forward_move": 0.0,
            "usercmd_left_move": 0.0,
            "round_start_time": 0.0,
            "total_rounds_played": 1,
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
        },
        {
            "steamid": 2,
            "name": "enemy_visible",
            "team_num": 2,
            "tick": tick,
            "X": 100.0,
            "Y": 0.0,
            "Z": 0.0,
            "velocity_X": 0.0,
            "velocity_Y": 0.0,
            "velocity_Z": 0.0,
            "health": 100,
            "armor_value": 0,
            "has_helmet": False,
            "is_alive": True,
            "balance": 0,
            "active_weapon_name": "AK-47",
            "active_weapon_ammo": 30,
            "total_ammo_left": 90,
            "pitch": 0.0,
            "yaw": 180.0,
            "is_scoped": False,
            "is_walking": False,
            "is_airborne": False,
            "duck_amount": 0.0,
            "ducking": False,
            "shots_fired": 0,
            "flash_duration": 0.0,
            "spotted": visible_enemy,
            "last_place_name": "a_site",
            "in_bomb_zone": False,
            "in_buy_zone": False,
            "which_bomb_zone": 0,
            "round_start_time": 0.0,
            "total_rounds_played": 1,
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
        },
        {
            "steamid": 3,
            "name": "enemy_hidden",
            "team_num": 2,
            "tick": tick,
            "X": 500.0,
            "Y": 0.0,
            "Z": 0.0,
            "velocity_X": 0.0,
            "velocity_Y": 0.0,
            "velocity_Z": 0.0,
            "health": 100,
            "armor_value": 0,
            "has_helmet": False,
            "is_alive": True,
            "balance": 0,
            "active_weapon_name": "AK-47",
            "active_weapon_ammo": 30,
            "total_ammo_left": 90,
            "pitch": 0.0,
            "yaw": 180.0,
            "is_scoped": False,
            "is_walking": False,
            "is_airborne": False,
            "duck_amount": 0.0,
            "ducking": False,
            "shots_fired": 0,
            "flash_duration": 0.0,
            "spotted": False,
            "last_place_name": "b_site",
            "in_bomb_zone": False,
            "in_buy_zone": False,
            "which_bomb_zone": 0,
            "round_start_time": 0.0,
            "total_rounds_played": 1,
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
        },
    ]
    return pd.DataFrame(rows)


class DataContractParityTests(unittest.TestCase):
    def test_hidden_truth_isolation(self):
        bundle = GameStateBuilder().build_state_bundle_from_tick_rows(make_tick_rows(visible_enemy=True), perspective_steamid=1)
        observed_ids = {enemy.steamid for enemy in bundle.observed_state.enemies}
        truth_by_id = {enemy.steamid: enemy for enemy in bundle.truth_state.enemies}
        self.assertEqual(observed_ids, {2})
        self.assertEqual(set(truth_by_id), {2, 3})
        self.assertEqual(truth_by_id[3].visibility, VisibilityStatus.HIDDEN_TRUTH_ONLY.value)
        self.assertTrue(all(enemy.steamid != 3 for enemy in bundle.observed_state.enemies))

    def test_visibility_and_last_seen_contract(self):
        builder = GameStateBuilder()
        state_visible = builder.build_from_tick_rows(make_tick_rows(visible_enemy=True, tick=100), perspective_steamid=1)
        state_hidden = builder.build_from_tick_rows(make_tick_rows(visible_enemy=False, tick=101), perspective_steamid=1)
        sequence = GameStateSequence(perspective_steamid=1, states=[state_visible, state_hidden])
        features = EnemyTrackerFeatureExtractor(seq_len=2).extract(sequence)
        name_to_idx = {name: idx for idx, name in enumerate(TRACKER_FEATURE_NAMES)}
        last_frame = features[-1]
        self.assertEqual(last_frame[name_to_idx["enemy_0_visible_mask"]], 0.0)
        self.assertEqual(last_frame[name_to_idx["enemy_0_last_seen_mask"]], 1.0)
        self.assertEqual(last_frame[name_to_idx["enemy_0_unavailable_mask"]], 0.0)
        aim_features = AimFeatureExtractor(seq_len=2).extract(sequence)
        aim_idx = {name: idx for idx, name in enumerate(AIM_FEATURE_NAMES)}
        self.assertEqual(aim_features[-1][aim_idx["enemy_visible_mask"]], 0.0)
        self.assertEqual(aim_features[-1][aim_idx["enemy_last_seen_mask"]], 1.0)

    def test_enemy_tracker_roster_does_not_use_future_truth(self):
        builder = GameStateBuilder()
        hidden_only_state = builder.build_from_tick_rows(make_tick_rows(visible_enemy=False, tick=100), perspective_steamid=1)
        sequence = GameStateSequence(perspective_steamid=1, states=[hidden_only_state])
        roster = build_enemy_roster(sequence)
        self.assertLess(roster[0], 0)
        truth_state = builder.build_truth_from_tick_rows(make_tick_rows(visible_enemy=False, tick=100), perspective_steamid=1)
        target_positions = build_enemy_position_target(truth_state, roster)
        target_confidence = build_enemy_confidence_target(truth_state, roster)
        self.assertEqual(target_positions.shape, (5, 3))
        self.assertEqual(target_confidence.shape, (5,))
        self.assertEqual(float(target_confidence[0]), 0.0)

    def test_train_runtime_schema_parity_and_validation(self):
        tracker = EnemyTrackerFeatureExtractor(seq_len=4)
        movement = MovementFeatureExtractor(seq_len=4)
        aim = AimFeatureExtractor(seq_len=4)
        self.assertEqual(tracker.schema().seq_len, 4)
        self.assertEqual(movement.schema().seq_len, 4)
        self.assertEqual(aim.schema().seq_len, 4)
        self.assertEqual(tracker.schema().feature_dim, len(TRACKER_FEATURE_NAMES))
        self.assertEqual(movement.schema().feature_dim, len(MOVEMENT_FEATURE_NAMES))
        checkpoint = {"feature_schema": tracker.schema().to_metadata()}
        validate_checkpoint_schema(checkpoint, tracker.schema(), "tracker.ckpt")
        with self.assertRaises(ValueError):
            validate_checkpoint_schema({"feature_schema": {**tracker.schema().to_metadata(), "schema_hash": "bad"}}, tracker.schema(), "bad.ckpt")

    def test_checkpoint_safety_fast_fail(self):
        with self.assertRaises(ValueError):
            FullNeuralRuntimeAgent(aim_checkpoint=None, movement_checkpoint=None, tracker_checkpoint=None)


if __name__ == "__main__":
    unittest.main()
