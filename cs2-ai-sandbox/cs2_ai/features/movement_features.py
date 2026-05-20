from __future__ import annotations

import numpy as np

from cs2_ai.config import MAX_TEAMMATES
from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_position, normalize_velocity, pad_or_trim_vector, relative_position
from cs2_ai.features.feature_contract import FeatureSchema, NORMALIZATION_CONSTANTS, SCHEMA_VERSION, pad_or_trim_sequence
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, DecisionOutput


MOVEMENT_FEATURE_NAMES = (
    "self_pos_x",
    "self_pos_y",
    "self_pos_z",
    "self_vel_x",
    "self_vel_y",
    "self_vel_z",
    "self_yaw",
    "self_is_walking",
    "self_is_airborne",
    "self_is_ducking",
    "teammate_0_rel_x",
    "teammate_0_rel_y",
    "teammate_0_rel_z",
    "teammate_1_rel_x",
    "teammate_1_rel_y",
    "teammate_1_rel_z",
    "teammate_2_rel_x",
    "teammate_2_rel_y",
    "teammate_2_rel_z",
    "teammate_3_rel_x",
    "teammate_3_rel_y",
    "teammate_3_rel_z",
    "teammate_0_present",
    "teammate_1_present",
    "teammate_2_present",
    "teammate_3_present",
    "target_rel_x",
    "target_rel_y",
    "target_rel_z",
    "belief_top_enemy_rel_x",
    "belief_top_enemy_rel_y",
    "belief_top_enemy_rel_z",
    "belief_top_enemy_confidence",
    "belief_enemy_count",
    "belief_a_count",
    "belief_b_count",
    "belief_mid_count",
)


class MovementFeatureExtractor:
    def __init__(self, seq_len: int | None = None):
        self.seq_len = seq_len

    def schema(self, seq_len: int | None = None) -> FeatureSchema:
        resolved_seq_len = int(seq_len if seq_len is not None else self.seq_len or 0)
        if resolved_seq_len <= 0:
            raise ValueError("MovementFeatureExtractor requires seq_len for schema generation.")
        return FeatureSchema(
            model_key="movement",
            version=SCHEMA_VERSION,
            seq_len=resolved_seq_len,
            feature_names=MOVEMENT_FEATURE_NAMES,
            default_value=0.0,
            normalization=dict(NORMALIZATION_CONSTANTS),
        )

    def extract(self, sequence, decision: DecisionOutput | None = None, belief_state: BeliefStateData | None = None) -> np.ndarray:
        frames = [self._state_to_vector(state, decision=None, belief_state=None) for state in sequence.states]
        if self.seq_len is not None:
            frames = pad_or_trim_sequence(frames, self.seq_len, self.feature_dim(), default_value=0.0)
        return np.asarray(frames, dtype=np.float32)

    def feature_dim(self) -> int:
        return len(MOVEMENT_FEATURE_NAMES)

    def _state_to_vector(self, state: GameState, decision: DecisionOutput | None, belief_state: BeliefStateData | None) -> list[float]:
        self_player = state.self_player
        teammate_features: list[float] = []
        teammate_present: list[float] = []
        for teammate in state.teammates[:MAX_TEAMMATES]:
            teammate_features.extend(normalize_position(v) for v in relative_position(self_player.position, teammate.position))
            teammate_present.append(1.0)
        target_position = [0.0, 0.0, 0.0]
        top_enemy_rel = [0.0, 0.0, 0.0]
        top_enemy_confidence = 0.0
        predicted_enemy_count = 0.0
        coarse_enemy_counts = [0.0, 0.0, 0.0]
        if belief_state is not None:
            top_enemy_rel = [normalize_position(v) for v in belief_state.top_enemy_rel_pos]
            top_enemy_confidence = float(belief_state.top_enemy_confidence)
            predicted_enemy_count = float(belief_state.predicted_enemy_count) / 5.0
            coarse_enemy_counts = [float(belief_state.coarse_enemy_counts.get(key, 0.0)) / 5.0 for key in ("A", "B", "mid")]
        return [
            *map(normalize_position, self_player.position),
            *map(normalize_velocity, self_player.velocity),
            normalize_angle(self_player.yaw),
            bool_to_float(self_player.is_walking),
            bool_to_float(self_player.is_airborne),
            bool_to_float(self_player.ducking),
            *pad_or_trim_vector(teammate_features, MAX_TEAMMATES * 3),
            *pad_or_trim_vector(teammate_present, MAX_TEAMMATES),
            *[normalize_position(v) for v in relative_position(self_player.position, target_position)],
            *top_enemy_rel,
            top_enemy_confidence,
            predicted_enemy_count,
            *coarse_enemy_counts,
        ]


def build_movement_target(game_state: GameState) -> np.ndarray:
    self_input = game_state.self_input
    values = [
        bool_to_float(self_input.forward),
        bool_to_float(self_input.back),
        bool_to_float(self_input.left),
        bool_to_float(self_input.right),
        bool_to_float(self_input.walk),
        bool_to_float(game_state.self_player.ducking),
    ]
    return np.asarray(values, dtype=np.float32)
