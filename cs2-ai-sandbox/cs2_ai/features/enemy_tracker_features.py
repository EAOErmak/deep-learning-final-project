from __future__ import annotations

import numpy as np

from cs2_ai.config import MAX_ENEMIES, MAX_TEAMMATES
from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_armor, normalize_hp, normalize_position, normalize_velocity, pad_or_trim_vector, relative_position
from cs2_ai.schemas.game_state import GameState, GameStateSequence


class EnemyTrackerFeatureExtractor:
    def extract(self, sequence: GameStateSequence) -> np.ndarray:
        return np.asarray([self._state_to_vector(state) for state in sequence.states], dtype=np.float32)

    def feature_dim(self) -> int:
        return 54

    def _state_to_vector(self, state) -> list[float]:
        self_player = state.self_player
        vector = [*map(normalize_position, self_player.position), *map(normalize_velocity, self_player.velocity), normalize_hp(self_player.health), normalize_armor(self_player.armor), bool_to_float(self_player.is_alive), normalize_angle(self_player.yaw), normalize_angle(self_player.pitch)]
        teammate_pos: list[float] = []
        teammate_alive: list[float] = []
        for teammate in state.teammates[:MAX_TEAMMATES]:
            teammate_pos.extend(normalize_position(v) for v in relative_position(self_player.position, teammate.position))
            teammate_alive.append(bool_to_float(teammate.is_alive))
        vector.extend(pad_or_trim_vector(teammate_pos, MAX_TEAMMATES * 3))
        vector.extend(pad_or_trim_vector(teammate_alive, MAX_TEAMMATES))
        enemy_values: list[float] = []
        enemy_visible: list[float] = []
        for enemy in sorted(state.enemies, key=lambda item: int(item.steamid))[:MAX_ENEMIES]:
            if enemy.spotted:
                enemy_values.extend(normalize_position(v) for v in relative_position(self_player.position, enemy.position))
                enemy_visible.append(1.0)
            else:
                enemy_values.extend([0.0, 0.0, 0.0])
                enemy_visible.append(0.0)
        vector.extend(pad_or_trim_vector(enemy_values, MAX_ENEMIES * 3))
        vector.extend(pad_or_trim_vector(enemy_visible, MAX_ENEMIES))
        vector.extend([bool_to_float(state.round.round_in_progress), bool_to_float(state.round.is_freeze_period), bool_to_float(state.round.is_warmup_period), bool_to_float(state.bomb.is_bomb_planted), bool_to_float(state.bomb.is_bomb_dropped), normalize_hp(state.round.ct_losing_streak), normalize_hp(state.round.t_losing_streak)])
        return vector


def build_enemy_roster(sequence: GameStateSequence, target_state: GameState | None = None) -> list[int]:
    roster: list[int] = []
    seen: set[int] = set()
    states = list(sequence.states)
    if target_state is not None:
        states.append(target_state)
    for state in states:
        visible_first = sorted(state.enemies, key=lambda item: (not item.spotted, int(item.steamid)))
        for enemy in visible_first:
            steamid = int(enemy.steamid)
            if steamid in seen:
                continue
            seen.add(steamid)
            roster.append(steamid)
            if len(roster) >= MAX_ENEMIES:
                return roster
    return roster[:MAX_ENEMIES]


def build_enemy_position_target(game_state: GameState, roster_steamids: list[int]) -> np.ndarray:
    enemy_by_steamid = {int(enemy.steamid): enemy for enemy in game_state.enemies}
    positions: list[list[float]] = []
    for steamid in roster_steamids[:MAX_ENEMIES]:
        enemy = enemy_by_steamid.get(int(steamid))
        if enemy is None:
            positions.append([0.0, 0.0, 0.0])
            continue
        positions.append([normalize_position(v) for v in enemy.position])
    while len(positions) < MAX_ENEMIES:
        positions.append([0.0, 0.0, 0.0])
    return np.asarray(positions, dtype=np.float32)


def build_enemy_confidence_target(game_state: GameState, roster_steamids: list[int]) -> np.ndarray:
    enemy_by_steamid = {int(enemy.steamid): enemy for enemy in game_state.enemies}
    confidences: list[float] = []
    for steamid in roster_steamids[:MAX_ENEMIES]:
        enemy = enemy_by_steamid.get(int(steamid))
        confidences.append(1.0 if enemy is not None and enemy.is_alive else 0.0)
    while len(confidences) < MAX_ENEMIES:
        confidences.append(0.0)
    return np.asarray(confidences, dtype=np.float32)
