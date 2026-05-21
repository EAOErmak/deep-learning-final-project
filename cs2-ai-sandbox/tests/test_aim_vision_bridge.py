from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.features.aim_features import AIM_FEATURE_MODE_DEMO_PROJECTED, AIM_FEATURE_MODE_VISION_LIKE, AIM_VISION_FEATURE_NAMES, AimFeatureExtractor
from cs2_ai.pipeline.neural_ai_pipeline import NeuralAIPipeline
from cs2_ai.state.game_state_builder import GameStateBuilder
from cs2_ai.vision.yolo_pipeline import VisionTarget
from cs2_ai.ml.utils.torch_utils import torch_available

if torch_available():
    import torch
else:
    torch = None


def make_tick_rows(tick: int = 100, visible_enemy: bool = True) -> pd.DataFrame:
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


class AimVisionBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        builder = GameStateBuilder()
        self.state = builder.build_from_tick_rows(make_tick_rows(), perspective_steamid=1)
        self.sequence = type("Sequence", (), {"states": [self.state]})()

    def test_extract_without_vision_target_zeros_vision_features(self):
        extractor = AimFeatureExtractor(seq_len=1, feature_mode=AIM_FEATURE_MODE_VISION_LIKE)
        features = extractor.extract(self.sequence, vision_target=None)
        name_to_idx = {name: idx for idx, name in enumerate(extractor.schema().feature_names)}
        self.assertEqual(features[0][name_to_idx["vision_enemy_visible"]], 0.0)
        self.assertEqual(features[0][name_to_idx["vision_screen_dx"]], 0.0)
        self.assertEqual(features[0][name_to_idx["vision_screen_dy"]], 0.0)
        self.assertEqual(features[0][name_to_idx["vision_confidence"]], 0.0)

    def test_extract_with_vision_target_populates_vision_features(self):
        extractor = AimFeatureExtractor(seq_len=1, feature_mode=AIM_FEATURE_MODE_VISION_LIKE)
        target = VisionTarget(screen_dx=0.2, screen_dy=-0.1, confidence=0.8, label="ct_head")
        features = extractor.extract(self.sequence, vision_target=target)
        name_to_idx = {name: idx for idx, name in enumerate(extractor.schema().feature_names)}
        self.assertEqual(features[0][name_to_idx["vision_enemy_visible"]], 1.0)
        self.assertAlmostEqual(float(features[0][name_to_idx["vision_screen_dx"]]), 0.2, places=6)
        self.assertAlmostEqual(float(features[0][name_to_idx["vision_screen_dy"]]), -0.1, places=6)
        self.assertAlmostEqual(float(features[0][name_to_idx["vision_confidence"]]), 0.8, places=6)
        self.assertEqual(features[0][name_to_idx["vision_is_head_target"]], 1.0)

    def test_old_extract_call_still_works(self):
        extractor = AimFeatureExtractor(seq_len=1, feature_mode=AIM_FEATURE_MODE_DEMO_PROJECTED)
        features = extractor.extract(self.sequence)
        self.assertEqual(features.shape, (1, extractor.feature_dim()))

    def test_neural_pipeline_passes_vision_target_into_aim_extractor(self):
        if not torch_available():
            self.skipTest("PyTorch not available")

        class DummyTrackerModel(torch.nn.Module):
            def forward(self, x):
                batch, seq_len, _ = x.shape
                return torch.zeros((batch, seq_len, 5, 3), dtype=x.dtype), torch.zeros((batch, seq_len, 5), dtype=x.dtype)

        class DummyMovementModel(torch.nn.Module):
            def forward(self, x):
                batch, seq_len, _ = x.shape
                return torch.zeros((batch, seq_len, 6), dtype=x.dtype)

        class DummyAimModel(torch.nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return (
                    torch.zeros((batch, 2), dtype=x.dtype),
                    torch.zeros((batch, 1), dtype=x.dtype),
                    torch.zeros((batch, 1), dtype=x.dtype),
                )

        class SpyAimExtractor(AimFeatureExtractor):
            def __init__(self):
                super().__init__(seq_len=1, feature_mode=AIM_FEATURE_MODE_VISION_LIKE)
                self.seen_vision_target = None

            def extract(self, sequence, belief_state=None, vision_target=None):
                self.seen_vision_target = vision_target
                return super().extract(sequence, belief_state=belief_state, vision_target=vision_target)

        pipeline = NeuralAIPipeline(
            aim_model=DummyAimModel(),
            movement_model=DummyMovementModel(),
            tracker_model=DummyTrackerModel(),
            memory_len=1,
            seq_lens={"aim": 1, "movement": 1, "tracker": 1},
            device="cpu",
            strict_readiness=True,
        )
        pipeline.aim_extractor = SpyAimExtractor()
        target = VisionTarget(screen_dx=0.2, screen_dy=-0.1, confidence=0.8, label="ct_head")
        pipeline.step(self.state, vision_target=target)
        self.assertIs(pipeline.aim_extractor.seen_vision_target, target)


if __name__ == "__main__":
    unittest.main()
