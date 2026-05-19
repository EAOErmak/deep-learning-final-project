from __future__ import annotations

from cs2_ai.ml.models.decision_dqn import DecisionDQN as RLDecisionMaker
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, DecisionOutput


class RuleBasedDecisionMaker:
    def reset(self) -> None:
        return None

    def decide(self, game_state: GameState, belief_state: BeliefStateData) -> DecisionOutput:
        strategic_action = "move_to_objective"
        tactical_action = "hold"
        target_position = None
        confidence = 0.5
        if game_state.round.is_freeze_period:
            strategic_action = "buy"
            tactical_action = "prepare"
        elif game_state.bomb.is_bomb_planted and game_state.self_player.team_num == 3:
            strategic_action = "retake"
        elif game_state.bomb.is_bomb_planted and game_state.self_player.team_num == 2:
            strategic_action = "defend_site"
        elif game_state.self_player.health < 30:
            tactical_action = "fallback"
        else:
            high_conf = [prediction for prediction in belief_state.predicted_enemies if prediction.confidence > 0.6]
            if high_conf:
                tactical_action = "clear_angles"
                best = max(high_conf, key=lambda item: item.confidence)
                target_position = list(best.predicted_position)
                confidence = float(best.confidence)
        aggression_level = 0.7 if game_state.self_player.health > 60 else 0.3
        return DecisionOutput(strategic_action=strategic_action, tactical_action=tactical_action, target_zone=None, target_position=target_position, aggression_level=aggression_level, confidence=confidence)
