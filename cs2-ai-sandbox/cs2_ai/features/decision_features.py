from __future__ import annotations

import numpy as np

from cs2_ai.features.encoding import bool_to_float, normalize_armor, normalize_hp, normalize_money, weapon_to_id_normalized
from cs2_ai.schemas.module_outputs import BeliefStateData


class DecisionFeatureExtractor:
    def extract(self, sequence, belief_state: BeliefStateData | None = None) -> np.ndarray:
        return np.asarray([self._state_to_vector(state, belief_state) for state in sequence.states], dtype=np.float32)

    def feature_dim(self) -> int:
        return 16

    def _state_to_vector(self, state, belief_state: BeliefStateData | None) -> list[float]:
        self_player = state.self_player
        team_alive = sum(player.is_alive for player in [state.self_player, *state.teammates])
        predicted = belief_state.predicted_enemies if belief_state else []
        visible_predictions = [prediction for prediction in predicted if prediction.confidence > 0.0]
        max_conf = max((prediction.confidence for prediction in visible_predictions), default=0.0)
        return [normalize_hp(self_player.health), normalize_armor(self_player.armor), normalize_money(self_player.money), weapon_to_id_normalized(self_player.weapon), normalize_hp(self_player.ammo), bool_to_float(self_player.is_alive), float(team_alive) / 5.0, float(len(visible_predictions)) / 5.0, float(max_conf), bool_to_float(state.bomb.is_bomb_planted), bool_to_float(state.bomb.is_bomb_dropped), bool_to_float(state.round.is_freeze_period), bool_to_float(state.round.round_in_progress), float(state.self_player.which_bomb_zone), bool_to_float(state.self_player.in_bomb_zone), bool_to_float(state.self_player.in_buy_zone)]
