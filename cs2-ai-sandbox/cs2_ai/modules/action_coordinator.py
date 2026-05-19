from __future__ import annotations

from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import ActionPlan, AimShootOutput, BuyOutput, DecisionOutput, InputCommand, MovementOutput


class ActionCoordinator:
    def build_action_plan(self, game_state: GameState, decision: DecisionOutput, movement: MovementOutput, aim: AimShootOutput, buy: BuyOutput) -> ActionPlan:
        keyboard_inputs: list[InputCommand] = []
        mouse_inputs: list[InputCommand] = []
        duration_ms = 100
        if decision.strategic_action == "buy" and buy.should_buy:
            for item in buy.buy_list:
                keyboard_inputs.append(InputCommand(device="keyboard", command=f"buy:{item}", value=True, duration_ms=duration_ms))
            return ActionPlan(keyboard_inputs=keyboard_inputs, mouse_inputs=mouse_inputs, duration_ms=duration_ms)
        forward, side = movement.move_direction
        if forward > 0:
            keyboard_inputs.append(InputCommand("keyboard", "W", True, duration_ms))
        elif forward < 0:
            keyboard_inputs.append(InputCommand("keyboard", "S", True, duration_ms))
        if side > 0:
            keyboard_inputs.append(InputCommand("keyboard", "D", True, duration_ms))
        elif side < 0:
            keyboard_inputs.append(InputCommand("keyboard", "A", True, duration_ms))
        if movement.movement_mode == "walk":
            keyboard_inputs.append(InputCommand("keyboard", "SHIFT", True, duration_ms))
        if movement.should_crouch or movement.movement_mode == "crouch":
            keyboard_inputs.append(InputCommand("keyboard", "CTRL", True, duration_ms))
        mouse_inputs.append(InputCommand("mouse", "mouse_move_yaw", float(aim.aim_delta[0]), duration_ms))
        mouse_inputs.append(InputCommand("mouse", "mouse_move_pitch", float(aim.aim_delta[1]), duration_ms))
        if aim.shoot:
            mouse_inputs.append(InputCommand("mouse", "mouse_left", True, duration_ms))
        if aim.rightclick:
            mouse_inputs.append(InputCommand("mouse", "mouse_right", True, duration_ms))
        return ActionPlan(keyboard_inputs=keyboard_inputs, mouse_inputs=mouse_inputs, duration_ms=duration_ms)
