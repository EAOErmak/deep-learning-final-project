from __future__ import annotations

from math import sqrt

from cs2_ai.schemas.module_outputs import MovementOutput


class SimpleMovementModule:
    def reset(self) -> None:
        return None

    def decide(self, game_state, belief_state, decision) -> MovementOutput:
        if decision.tactical_action == "fallback":
            return MovementOutput(move_direction=[-1.0, 0.0], movement_mode="run", target_position=None, should_jump=False, should_crouch=False)
        if decision.target_position is not None:
            dx = float(decision.target_position[0]) - game_state.self_player.position[0]
            dy = float(decision.target_position[1]) - game_state.self_player.position[1]
            norm = max(1.0, sqrt(dx * dx + dy * dy))
            return MovementOutput(move_direction=[dx / norm, dy / norm], movement_mode="run", target_position=list(decision.target_position), should_jump=False, should_crouch=False)
        if decision.tactical_action == "clear_angles":
            return MovementOutput(move_direction=[0.5, 0.0], movement_mode="walk", target_position=None, should_jump=False, should_crouch=False)
        return MovementOutput(move_direction=[1.0, 0.0], movement_mode="run", target_position=None, should_jump=False, should_crouch=False)
