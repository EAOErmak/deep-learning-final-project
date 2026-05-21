from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.features.aim_features import build_aim_structured_target
from cs2_ai.ml.models.aim_attention import AIM_HEAD_MODE_MULTI_HEAD, AimAttentionModel
from cs2_ai.ml.training.train_aim import AimTrainer, collate_aim_batch
from cs2_ai.ml.utils.torch_utils import torch_available
from cs2_ai.state.game_state_builder import GameStateBuilder

if torch_available():
    import torch
    from torch.utils.data import DataLoader, Dataset
else:
    torch = None
    DataLoader = None
    Dataset = object


def make_tick_rows(tick: int = 100, visible_enemy: bool = True, fire: bool = False, rightclick: bool = False, zoom: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
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
                "FIRE": fire,
                "RIGHTCLICK": rightclick,
                "RELOAD": False,
                "USE": False,
                "ZOOM": zoom,
                "WALK": False,
                "usercmd_mouse_dx": 15.0 if visible_enemy else 0.0,
                "usercmd_mouse_dy": -10.0 if visible_enemy else 0.0,
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
                "name": "enemy",
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
        ]
    )


class SingleBatchAimDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


class AimMultiHeadTests(unittest.TestCase):
    def test_structured_target_shapes(self):
        builder = GameStateBuilder()
        current_state = builder.build_from_tick_rows(make_tick_rows(tick=100, visible_enemy=True), perspective_steamid=1)
        next_state = builder.build_from_tick_rows(make_tick_rows(tick=101, visible_enemy=True, fire=True, rightclick=True, zoom=True), perspective_steamid=1)
        target = build_aim_structured_target(current_state, next_state)
        self.assertEqual(tuple(target.aim_delta.shape), (4,))
        self.assertEqual(tuple(target.binary_actions.shape), (3,))
        self.assertEqual(tuple(target.valid_aim_mask.shape), (1,))
        self.assertEqual(float(target.valid_aim_mask[0]), 1.0)

    def test_valid_mask_zero_without_visible_enemy(self):
        builder = GameStateBuilder()
        current_state = builder.build_from_tick_rows(make_tick_rows(tick=100, visible_enemy=False), perspective_steamid=1)
        next_state = builder.build_from_tick_rows(make_tick_rows(tick=101, visible_enemy=False), perspective_steamid=1)
        target = build_aim_structured_target(current_state, next_state)
        self.assertEqual(float(target.valid_aim_mask[0]), 0.0)

    def test_multi_head_loss_stays_finite_without_valid_samples(self):
        if not torch_available():
            self.skipTest("PyTorch not available")
        model = AimAttentionModel(input_dim=6, head_mode=AIM_HEAD_MODE_MULTI_HEAD)
        trainer = AimTrainer(
            model=model,
            device='cpu',
            learning_rate=1e-3,
            head_mode=AIM_HEAD_MODE_MULTI_HEAD,
        )
        sample = (
            torch.zeros((1, 6), dtype=torch.float32).numpy(),
            torch.zeros((4,), dtype=torch.float32).numpy(),
            torch.zeros((3,), dtype=torch.float32).numpy(),
            torch.zeros((1,), dtype=torch.float32).numpy(),
            {'sample_id': 's1', 'demo_name': 'demo'},
        )
        loader = DataLoader(SingleBatchAimDataset(sample), batch_size=1, shuffle=False, collate_fn=collate_aim_batch)
        metrics = trainer.eval_epoch(loader, epoch=1, writer=None)
        self.assertFalse(math.isnan(metrics['loss']))
        self.assertFalse(math.isnan(metrics['aim_loss']))


if __name__ == '__main__':
    unittest.main()
