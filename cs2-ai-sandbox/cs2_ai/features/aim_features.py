from __future__ import annotations

import numpy as np

from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_hp, normalize_velocity, world_to_screen_delta, weapon_to_id_normalized
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData

try:
    from cs2_ai.vision.yolo_pipeline import VisionTarget
except ImportError:
    VisionTarget = None


class AimFeatureExtractor:
    def extract(self, sequence, belief_state: BeliefStateData | None = None, vision_target: 'VisionTarget' | None = None) -> np.ndarray:
        return np.asarray([self._state_to_vector(state, belief_state, vision_target) for state in sequence.states], dtype=np.float32)

    def feature_dim(self) -> int:
        return 18

    def _state_to_vector(self, state, belief_state: BeliefStateData | None, vision_target: 'VisionTarget' | None) -> list[float]:
        self_player = state.self_player
        
        if vision_target is not None:
            enemy_screen = [vision_target.screen_dx, vision_target.screen_dy]
            enemy_visible = 1.0
        else:
            spotted_enemy = next((enemy for enemy in state.enemies if enemy.spotted and enemy.is_alive), None)
            if spotted_enemy is not None and self_player.position is not None and spotted_enemy.position is not None:
            enemy_screen_dx, enemy_screen_dy = world_to_screen_delta(
                self_player.position,
                self_player.yaw,
                self_player.pitch,
                spotted_enemy.position
            )
            enemy_screen = [enemy_screen_dx, enemy_screen_dy]
            enemy_visible = 1.0
        else:
            enemy_screen = [0.0, 0.0]
            enemy_visible = 0.0
            
        predicted_screen = [0.0, 0.0]
        predicted_confidence = 0.0
        if belief_state and belief_state.predicted_enemies and self_player.position is not None:
            best_prediction = max(belief_state.predicted_enemies, key=lambda item: item.confidence)
            pred_dx, pred_dy = world_to_screen_delta(
                self_player.position,
                self_player.yaw,
                self_player.pitch,
                best_prediction.predicted_position
            )
            predicted_screen = [pred_dx, pred_dy]
            predicted_confidence = float(best_prediction.confidence)
            
        return [normalize_angle(self_player.yaw), normalize_angle(self_player.pitch), *map(normalize_velocity, self_player.velocity if self_player.velocity else [0.0, 0.0, 0.0]), weapon_to_id_normalized(self_player.weapon), normalize_hp(self_player.ammo), normalize_hp(self_player.shots_fired), bool_to_float(self_player.is_scoped), bool_to_float(self_player.is_walking), bool_to_float(self_player.is_airborne), normalize_hp(self_player.flash_duration), *enemy_screen, enemy_visible, *predicted_screen, predicted_confidence]


def build_aim_target(current_state: GameState, next_state: GameState) -> np.ndarray:
    target = [normalize_angle(next_state.self_player.yaw - current_state.self_player.yaw), normalize_angle(next_state.self_player.pitch - current_state.self_player.pitch), bool_to_float(next_state.self_input.fire), bool_to_float(next_state.self_input.rightclick), bool_to_float(next_state.self_input.zoom), float(next_state.self_input.usercmd_mouse_dx) / 500.0, float(next_state.self_input.usercmd_mouse_dy) / 500.0]
    return np.asarray(target, dtype=np.float32)
