from __future__ import annotations

import math

from cs2_ai.ml.models.aim_attention import AimAttentionModel as AttentionAimShootModel
from cs2_ai.schemas.module_outputs import AimShootOutput


def calculate_yaw_pitch_delta(self_pos, current_yaw, current_pitch, target_pos):
    dx = float(target_pos[0]) - float(self_pos[0])
    dy = float(target_pos[1]) - float(self_pos[1])
    dz = float(target_pos[2]) - float(self_pos[2])
    target_yaw = math.degrees(math.atan2(dy, dx)) if dx or dy else current_yaw
    flat_distance = math.sqrt(dx * dx + dy * dy)
    target_pitch = math.degrees(math.atan2(dz, max(flat_distance, 1e-6)))
    return [target_yaw - float(current_yaw), target_pitch - float(current_pitch)]


class SimpleAimShootModule:
    def reset(self) -> None:
        return None

    def decide(self, game_state, belief_state, decision) -> AimShootOutput:
        self_player = game_state.self_player
        visible_enemy = next((enemy for enemy in game_state.enemies if enemy.spotted and enemy.is_alive), None)
        if visible_enemy is not None:
            aim_delta = calculate_yaw_pitch_delta(self_player.position, self_player.yaw, self_player.pitch, visible_enemy.position)
            shoot = True
            confidence = 1.0
            aim_position = list(visible_enemy.position)
        else:
            predicted = max(belief_state.predicted_enemies, key=lambda item: item.confidence, default=None)
            if predicted is not None and predicted.confidence > 0.5:
                aim_delta = calculate_yaw_pitch_delta(self_player.position, self_player.yaw, self_player.pitch, predicted.predicted_position)
                shoot = False
                confidence = float(predicted.confidence)
                aim_position = list(predicted.predicted_position)
            else:
                aim_delta = [0.0, 0.0]
                shoot = False
                confidence = 0.0
                aim_position = None
        speed = abs(self_player.velocity[0]) + abs(self_player.velocity[1])
        return AimShootOutput(aim_delta=aim_delta, aim_position=aim_position, shoot=shoot, rightclick=False, burst_length=3 if shoot else 0, counter_strafe=shoot and speed > 40.0, confidence=confidence)
