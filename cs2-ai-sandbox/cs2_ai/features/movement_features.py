from __future__ import annotations

import numpy as np

from cs2_ai.config import MAX_TEAMMATES
from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_position, normalize_velocity, pad_or_trim_vector, relative_position
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, DecisionOutput


class MovementFeatureExtractor:
    def extract(self, sequence, decision: DecisionOutput | None = None, belief_state: BeliefStateData | None = None) -> np.ndarray:
        return np.asarray([self._state_to_vector(state, decision, belief_state) for state in sequence.states], dtype=np.float32)

    def feature_dim(self) -> int:
        return 28

    def _state_to_vector(self, state, decision: DecisionOutput | None, belief_state: BeliefStateData | None) -> list[float]:
        self_player = state.self_player
        target_position = decision.target_position if decision and decision.target_position else [0.0, 0.0, 0.0]
        teammate_features: list[float] = []
        for teammate in state.teammates[:MAX_TEAMMATES]:
            teammate_features.extend(normalize_position(v) for v in relative_position(self_player.position, teammate.position))
        danger_values = [float((belief_state.danger_zones if belief_state else {}).get(key, 0.0)) for key in ("A", "B", "mid")]
        return [*map(normalize_position, self_player.position), *map(normalize_velocity, self_player.velocity), normalize_angle(self_player.yaw), bool_to_float(self_player.is_walking), bool_to_float(self_player.is_airborne), bool_to_float(self_player.ducking), *pad_or_trim_vector(teammate_features, MAX_TEAMMATES * 3), *[normalize_position(v) for v in relative_position(self_player.position, target_position)], *danger_values]


def build_movement_target(game_state: GameState) -> np.ndarray:
    self_input = game_state.self_input
    values = [bool_to_float(self_input.forward), bool_to_float(self_input.back), bool_to_float(self_input.left), bool_to_float(self_input.right), bool_to_float(self_input.walk), bool_to_float(game_state.self_player.ducking), float(self_input.usercmd_forward_move) / 450.0, float(self_input.usercmd_left_move) / 450.0]
    return np.asarray(values, dtype=np.float32)
