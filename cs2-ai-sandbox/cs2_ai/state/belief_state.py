from __future__ import annotations

from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, EnemyTrackerOutput


class BeliefState:
    def update(self, game_state: GameState, enemy_tracker_output: EnemyTrackerOutput) -> BeliefStateData:
        danger_zones: dict[str, float] = {}
        enemy_places = {enemy.steamid: enemy.last_place_name or "unknown" for enemy in game_state.enemies}
        for prediction in enemy_tracker_output.predictions:
            zone = enemy_places.get(prediction.steamid or -1, "unknown")
            danger_zones[zone] = max(danger_zones.get(zone, 0.0), float(prediction.confidence))
        safe_zones = {zone: max(0.0, 1.0 - danger) for zone, danger in danger_zones.items()}
        return BeliefStateData(
            predicted_enemies=enemy_tracker_output.predictions,
            danger_zones=danger_zones,
            safe_zones=safe_zones,
            site_control={"A": 0.0, "B": 0.0, "mid": 0.0},
        )
