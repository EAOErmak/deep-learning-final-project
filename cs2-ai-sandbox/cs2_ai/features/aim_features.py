from __future__ import annotations

import numpy as np

from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_hp, normalize_velocity, world_to_screen_delta, weapon_to_id_normalized
from cs2_ai.features.feature_contract import FeatureSchema, NORMALIZATION_CONSTANTS, SCHEMA_VERSION, pad_or_trim_sequence
from cs2_ai.schemas.game_state import GameState, GameStateSequence, VisibilityStatus
from cs2_ai.schemas.module_outputs import BeliefStateData

try:
    from cs2_ai.vision.yolo_pipeline import VisionTarget
except ImportError:
    VisionTarget = None


AIM_MOUSE_SCALE = 500.0
AIM_FEATURE_NAMES = (
    "self_yaw",
    "self_pitch",
    "self_vel_x",
    "self_vel_y",
    "self_vel_z",
    "weapon_id",
    "ammo",
    "shots_fired",
    "is_scoped",
    "is_walking",
    "is_airborne",
    "flash_duration",
    "enemy_screen_dx",
    "enemy_screen_dy",
    "enemy_visible_mask",
    "enemy_last_seen_mask",
    "enemy_unavailable_mask",
    "has_observed_target_mask",
)


def normalize_mouse_delta(value: float) -> float:
    return float(value) / AIM_MOUSE_SCALE


def denormalize_mouse_delta(value: float) -> float:
    return float(value) * AIM_MOUSE_SCALE


class AimFeatureExtractor:
    def __init__(self, seq_len: int | None = None):
        self.seq_len = seq_len

    def schema(self, seq_len: int | None = None) -> FeatureSchema:
        resolved_seq_len = int(seq_len if seq_len is not None else self.seq_len or 0)
        if resolved_seq_len <= 0:
            raise ValueError("AimFeatureExtractor requires seq_len for schema generation.")
        return FeatureSchema(
            model_key="aim",
            version=SCHEMA_VERSION,
            seq_len=resolved_seq_len,
            feature_names=AIM_FEATURE_NAMES,
            default_value=0.0,
            normalization=dict(NORMALIZATION_CONSTANTS),
        )

    def extract(self, sequence, belief_state: BeliefStateData | None = None, vision_target: 'VisionTarget' | None = None) -> np.ndarray:
        last_seen_enemy = None
        frames = []
        for state in sequence.states:
            visible_enemy = next((enemy for enemy in state.enemies if enemy.visibility == VisibilityStatus.VISIBLE.value and enemy.is_alive), None)
            if visible_enemy is not None:
                last_seen_enemy = visible_enemy
            frames.append(self._state_to_vector(state, visible_enemy, last_seen_enemy))
        if self.seq_len is not None:
            frames = pad_or_trim_sequence(frames, self.seq_len, self.feature_dim(), default_value=0.0)
        return np.asarray(frames, dtype=np.float32)

    def feature_dim(self) -> int:
        return len(AIM_FEATURE_NAMES)

    def _state_to_vector(self, state: GameState, visible_enemy, last_seen_enemy) -> list[float]:
        self_player = state.self_player
        enemy_screen = [0.0, 0.0]
        visible_mask = 0.0
        last_seen_mask = 0.0
        unavailable_mask = 1.0
        has_observed_target = 0.0

        if visible_enemy is not None and self_player.position is not None and visible_enemy.position is not None:
            enemy_screen_dx, enemy_screen_dy = world_to_screen_delta(
                self_player.position,
                self_player.yaw,
                self_player.pitch,
                visible_enemy.position,
            )
            enemy_screen = [enemy_screen_dx, enemy_screen_dy]
            visible_mask = 1.0
            unavailable_mask = 0.0
            has_observed_target = 1.0
        elif last_seen_enemy is not None and self_player.position is not None and last_seen_enemy.position is not None:
            enemy_screen_dx, enemy_screen_dy = world_to_screen_delta(
                self_player.position,
                self_player.yaw,
                self_player.pitch,
                last_seen_enemy.position,
            )
            enemy_screen = [enemy_screen_dx, enemy_screen_dy]
            last_seen_mask = 1.0
            unavailable_mask = 0.0
            has_observed_target = 1.0

        return [
            normalize_angle(self_player.yaw),
            normalize_angle(self_player.pitch),
            *map(normalize_velocity, self_player.velocity if self_player.velocity else [0.0, 0.0, 0.0]),
            weapon_to_id_normalized(self_player.weapon),
            normalize_hp(self_player.ammo),
            normalize_hp(self_player.shots_fired),
            bool_to_float(self_player.is_scoped),
            bool_to_float(self_player.is_walking),
            bool_to_float(self_player.is_airborne),
            normalize_hp(self_player.flash_duration),
            *enemy_screen,
            visible_mask,
            last_seen_mask,
            unavailable_mask,
            has_observed_target,
        ]


def build_aim_target(current_state: GameState, next_state: GameState) -> np.ndarray:
    target = [
        normalize_angle(next_state.self_player.yaw - current_state.self_player.yaw),
        normalize_angle(next_state.self_player.pitch - current_state.self_player.pitch),
        bool_to_float(next_state.self_input.fire),
        bool_to_float(next_state.self_input.rightclick),
        bool_to_float(next_state.self_input.zoom),
        normalize_mouse_delta(next_state.self_input.usercmd_mouse_dx),
        normalize_mouse_delta(next_state.self_input.usercmd_mouse_dy),
    ]
    return np.asarray(target, dtype=np.float32)
