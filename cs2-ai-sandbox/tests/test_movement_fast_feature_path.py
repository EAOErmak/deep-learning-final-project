from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from cs2_ai.dataset.sequence_dataset import PerspectiveSequenceDataset
from cs2_ai.ml.training.train_movement import MovementSequenceTorchDataset
from test_movement_action_chunk_target import build_tick_df


class TestMovementFastFeaturePath(unittest.TestCase):
    def test_parity_between_fast_and_slow_paths(self):
        # 1. Setup tick dataframe with a teammate and some actions
        tick_df = build_tick_df()
        
        # Add another player on the CT team (steamid 3) to test teammate features
        # Note: perspective player steamid is 1 (team CT=3) in the tick df.
        rows_to_append = []
        for tick in range(1, 11):
            # Steamid 3 is on CT team (3), so they are a teammate of player 1 (CT).
            row_data = {
                "steamid": 3,
                "name": "teammate3",
                "team_num": 3,
                "tick": tick,
                "X": 100.0,
                "Y": 50.0,
                "Z": 10.0,
                "velocity_X": 5.0,
                "velocity_Y": 0.0,
                "velocity_Z": 0.0,
                "health": 100,
                "armor_value": 100,
                "has_helmet": True,
                "is_alive": True,
                "balance": 1000,
                "active_weapon_name": "USP-S",
                "active_weapon_ammo": 12,
                "total_ammo_left": 24,
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
                "last_place_name": "A-site",
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
                "JUMP": False,
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
            rows_to_append.append(row_data)
        
        full_df = pd.concat([tick_df, pd.DataFrame(rows_to_append)], ignore_index=True)
        
        # 2. Build the datasets
        # PerspectiveSequenceDataset uses perspective_steamid = 1 (CT)
        base_dataset = PerspectiveSequenceDataset(full_df, seq_len=4, stride=1, alive_only=True)
        
        # legacy feature mode, chunk target mode
        dataset = MovementSequenceTorchDataset(
            base_dataset,
            target_mode="action_chunk",
            chunk_len=4,
            movement_feature_mode="legacy",
            use_grid_navigation_features=False,
            profile_dataloader=True  # This triggers the assert_allclose checks inside dataset __getitem__
        )
        
        # 3. Access every item in the dataset to verify they match exactly
        self.assertGreater(len(dataset), 0)
        for idx in range(len(dataset)):
            # Calling __getitem__ directly runs both fast and slow paths and does the assertion inside train_movement.py
            features, target, meta = dataset[idx]
            self.assertEqual(features.shape, (4, 37))
            self.assertEqual(target.shape, (4, 7))


if __name__ == "__main__":
    unittest.main()
