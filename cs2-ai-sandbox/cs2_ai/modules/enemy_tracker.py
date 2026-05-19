from __future__ import annotations

from typing import Any

from cs2_ai.config import MAX_ENEMIES
from cs2_ai.ml.models.enemy_tracker_lstm import EnemyTrackerLSTM as LSTMEnemyTracker
from cs2_ai.schemas.game_state import GameStateSequence
from cs2_ai.schemas.module_outputs import EnemyPrediction, EnemyTrackerOutput


class RuleBasedEnemyTracker:
    def __init__(self) -> None:
        self.last_seen: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self.last_seen.clear()

    def predict(self, sequence: GameStateSequence) -> EnemyTrackerOutput:
        if not sequence.states:
            return EnemyTrackerOutput(predictions=[])
        state = sequence.states[-1]
        predictions: list[EnemyPrediction] = []
        for slot in range(MAX_ENEMIES):
            if slot < len(state.enemies):
                enemy = state.enemies[slot]
                if enemy.spotted and enemy.is_alive:
                    self.last_seen[enemy.steamid] = {"position": list(enemy.position), "velocity": list(enemy.velocity), "tick": state.tick}
                    predictions.append(EnemyPrediction(slot, enemy.steamid, list(enemy.position), list(enemy.velocity), 1.0, 0.0))
                elif enemy.steamid in self.last_seen:
                    cached = self.last_seen[enemy.steamid]
                    last_seen_ticks = max(0, state.tick - int(cached.get("tick", state.tick)))
                    last_seen_seconds = last_seen_ticks / 64.0
                    confidence = max(0.0, 1.0 - last_seen_seconds / 8.0)
                    predictions.append(EnemyPrediction(slot, enemy.steamid, list(cached.get("position", [0.0, 0.0, 0.0])), list(cached.get("velocity", [0.0, 0.0, 0.0])), confidence, last_seen_seconds))
                else:
                    predictions.append(EnemyPrediction(slot, enemy.steamid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, None))
            else:
                predictions.append(EnemyPrediction(slot, None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, None))
        return EnemyTrackerOutput(predictions=predictions)
