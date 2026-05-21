from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from cs2_ai.features.aim_features import AIM_FEATURE_MODE_VISION_LIKE, AIM_VISION_FEATURE_NAMES, AimFeatureExtractor
from cs2_ai.state.game_state_builder import GameStateBuilder
from cs2_ai.vision.yolo_pipeline import VisionTarget, YoloVisionModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dry-run the YOLO -> aim feature bridge without sending inputs.')
    parser.add_argument('--fake-target', action='store_true')
    parser.add_argument('--yolo-weights', type=Path, default=None)
    parser.add_argument('--team', type=str, default='CT')
    return parser.parse_args()


def make_tick_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "steamid": 1, "name": "self", "team_num": 3, "tick": 100, "X": 0.0, "Y": 0.0, "Z": 0.0,
                "velocity_X": 0.0, "velocity_Y": 0.0, "velocity_Z": 0.0, "health": 100, "armor_value": 100,
                "has_helmet": True, "is_alive": True, "balance": 1000, "active_weapon_name": "M4A1-S",
                "active_weapon_ammo": 25, "total_ammo_left": 90, "pitch": 0.0, "yaw": 0.0, "is_scoped": False,
                "is_walking": False, "is_airborne": False, "duck_amount": 0.0, "ducking": False, "shots_fired": 0,
                "flash_duration": 0.0, "spotted": True, "last_place_name": "mid", "in_bomb_zone": False,
                "in_buy_zone": False, "which_bomb_zone": 0, "FORWARD": False, "BACK": False, "LEFT": False,
                "RIGHT": False, "FIRE": False, "RIGHTCLICK": False, "RELOAD": False, "USE": False, "ZOOM": False,
                "WALK": False, "usercmd_mouse_dx": 0.0, "usercmd_mouse_dy": 0.0, "usercmd_forward_move": 0.0,
                "usercmd_left_move": 0.0, "round_start_time": 0.0, "total_rounds_played": 1, "round_in_progress": True,
                "is_freeze_period": False, "is_warmup_period": False, "game_phase": 0, "round_win_status": 0,
                "round_win_reason": 0, "ct_losing_streak": 0, "t_losing_streak": 0, "is_bomb_planted": False,
                "is_bomb_dropped": False,
            }
        ]
    )


def resolve_vision_target(args: argparse.Namespace) -> VisionTarget | None:
    if args.fake_target:
        return VisionTarget(screen_dx=0.2, screen_dy=-0.1, confidence=0.85, label='ct_head')
    if args.yolo_weights is None:
        return None
    vision_module = YoloVisionModule(args.yolo_weights)
    vision_module.update_context(args.team)
    try:
        vision_module.start()
        time.sleep(0.25)
        return vision_module.get_latest_target()
    finally:
        if vision_module.is_running:
            vision_module.stop()


def main() -> int:
    args = parse_args()
    builder = GameStateBuilder()
    state = builder.build_from_tick_rows(make_tick_rows(), perspective_steamid=1)
    sequence = type("Sequence", (), {"states": [state]})()
    target = resolve_vision_target(args)

    extractor = AimFeatureExtractor(seq_len=1, feature_mode=AIM_FEATURE_MODE_VISION_LIKE)
    features = extractor.extract(sequence, vision_target=target)
    feature_names = extractor.schema().feature_names
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    vision_features = {name: float(features[-1][name_to_idx[name]]) for name in AIM_VISION_FEATURE_NAMES}

    print(json.dumps(
        {
            'vision_target': None if target is None else {
                'screen_dx': float(target.screen_dx),
                'screen_dy': float(target.screen_dy),
                'confidence': float(target.confidence),
                'label': str(target.label),
            },
            'feature_dim': extractor.feature_dim(),
            'vision_features': vision_features,
        },
        indent=2,
        ensure_ascii=True,
    ))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
