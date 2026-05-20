from __future__ import annotations

from cs2_ai.features.encoding import relative_position
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, EnemyTrackerOutput


class BeliefState:
    def update(self, game_state: GameState, enemy_tracker_output: EnemyTrackerOutput) -> BeliefStateData:
        danger_zones: dict[str, float] = {}
        enemy_places = {enemy.steamid: enemy.last_place_name or "unknown" for enemy in game_state.enemies}
        best_prediction = None
        coarse_enemy_counts = {"A": 0.0, "B": 0.0, "mid": 0.0}
        for prediction in enemy_tracker_output.predictions:
            zone = enemy_places.get(prediction.steamid or -1, "unknown")
            danger_zones[zone] = max(danger_zones.get(zone, 0.0), float(prediction.confidence))
            if best_prediction is None or prediction.confidence > best_prediction.confidence:
                best_prediction = prediction
            zone_key = self._coarse_zone_from_place(zone)
            if zone_key is not None and prediction.confidence > 0.0:
                coarse_enemy_counts[zone_key] += 1.0
        safe_zones = {zone: max(0.0, 1.0 - danger) for zone, danger in danger_zones.items()}
        top_enemy_rel_pos = [0.0, 0.0, 0.0]
        top_enemy_confidence = 0.0
        if best_prediction is not None and game_state.self_player.position is not None:
            top_enemy_rel_pos = [
                float(value)
                for value in relative_position(game_state.self_player.position, best_prediction.predicted_position)
            ]
            top_enemy_confidence = float(best_prediction.confidence)
        return BeliefStateData(
            predicted_enemies=enemy_tracker_output.predictions,
            top_enemy_rel_pos=top_enemy_rel_pos,
            top_enemy_confidence=top_enemy_confidence,
            predicted_enemy_count=sum(1 for prediction in enemy_tracker_output.predictions if prediction.confidence > 0.0),
            coarse_enemy_counts=coarse_enemy_counts,
            danger_zones=danger_zones,
            safe_zones=safe_zones,
            site_control={"A": 0.0, "B": 0.0, "mid": 0.0},
        )

    def _coarse_zone_from_place(self, place_name: str) -> str | None:
        normalized = str(place_name or "").strip().lower()
        if not normalized or normalized == "unknown":
            return None
        if normalized.startswith("a") or "long" in normalized or "short" in normalized or "cat" in normalized:
            return "A"
        if normalized.startswith("b") or "tunnels" in normalized or "upper" in normalized or "lower" in normalized:
            return "B"
        if "mid" in normalized or "spawn" in normalized or "door" in normalized:
            return "mid"
        return None
