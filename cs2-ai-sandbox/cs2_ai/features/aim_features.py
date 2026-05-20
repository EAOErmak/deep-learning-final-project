from __future__ import annotations

import numpy as np

from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_hp, normalize_velocity, relative_position, weapon_to_id_normalized
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData


class AimFeatureExtractor:
    def extract(self, sequence, belief_state: BeliefStateData | None = None) -> np.ndarray:
        return np.asarray([self._state_to_vector(state, belief_state) for state in sequence.states], dtype=np.float32)

    def feature_dim(self) -> int:
        return 20

    def _state_to_vector(self, state, belief_state: BeliefStateData | None) -> list[float]:
        self_player = state.self_player
        spotted_enemy = next((enemy for enemy in state.enemies if enemy.spotted and enemy.is_alive), None)
        if spotted_enemy is not None:
            enemy_relative = [coord / 10000.0 for coord in relative_position(self_player.position, spotted_enemy.position)]
            enemy_visible = 1.0
        else:
            enemy_relative = [0.0, 0.0, 0.0]
            enemy_visible = 0.0
        predicted_position = [0.0, 0.0, 0.0]
        predicted_confidence = 0.0
        if belief_state and belief_state.predicted_enemies:
            best_prediction = max(belief_state.predicted_enemies, key=lambda item: item.confidence)
            predicted_position = [coord / 10000.0 for coord in best_prediction.predicted_position]
            predicted_confidence = float(best_prediction.confidence)
        return [normalize_angle(self_player.yaw), normalize_angle(self_player.pitch), *map(normalize_velocity, self_player.velocity), weapon_to_id_normalized(self_player.weapon), normalize_hp(self_player.ammo), normalize_hp(self_player.shots_fired), bool_to_float(self_player.is_scoped), bool_to_float(self_player.is_walking), bool_to_float(self_player.is_airborne), normalize_hp(self_player.flash_duration), *enemy_relative, enemy_visible, *predicted_position, predicted_confidence]


def build_aim_target(current_state: GameState, next_state: GameState) -> np.ndarray:
    target = [normalize_angle(next_state.self_player.yaw - current_state.self_player.yaw), normalize_angle(next_state.self_player.pitch - current_state.self_player.pitch), bool_to_float(next_state.self_input.fire), bool_to_float(next_state.self_input.rightclick), bool_to_float(next_state.self_input.zoom), float(next_state.self_input.usercmd_mouse_dx) / 500.0, float(next_state.self_input.usercmd_mouse_dy) / 500.0]
    return np.asarray(target, dtype=np.float32)
