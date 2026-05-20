from __future__ import annotations

import numpy as np

from cs2_ai.config import MAX_ENEMIES, MAX_TEAMMATES
from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_armor, normalize_hp, normalize_position, normalize_velocity, pad_or_trim_vector, relative_position
from cs2_ai.features.feature_contract import FeatureSchema, NORMALIZATION_CONSTANTS, SCHEMA_VERSION, pad_or_trim_sequence
from cs2_ai.schemas.game_state import DemoTruthState, GameState, GameStateSequence, PlayerState, VisibilityStatus


def _enemy_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for slot in range(MAX_ENEMIES):
        prefix = f"enemy_{slot}"
        names.extend(
            [
                f"{prefix}_rel_x",
                f"{prefix}_rel_y",
                f"{prefix}_rel_z",
                f"{prefix}_visible_mask",
                f"{prefix}_last_seen_mask",
                f"{prefix}_unavailable_mask",
            ]
        )
    return tuple(names)


TRACKER_FEATURE_NAMES = (
    "self_pos_x",
    "self_pos_y",
    "self_pos_z",
    "self_vel_x",
    "self_vel_y",
    "self_vel_z",
    "self_hp",
    "self_armor",
    "self_alive",
    "self_yaw",
    "self_pitch",
    "teammate_0_rel_x",
    "teammate_0_rel_y",
    "teammate_0_rel_z",
    "teammate_1_rel_x",
    "teammate_1_rel_y",
    "teammate_1_rel_z",
    "teammate_2_rel_x",
    "teammate_2_rel_y",
    "teammate_2_rel_z",
    "teammate_3_rel_x",
    "teammate_3_rel_y",
    "teammate_3_rel_z",
    "teammate_0_alive",
    "teammate_1_alive",
    "teammate_2_alive",
    "teammate_3_alive",
    *_enemy_feature_names(),
    "round_in_progress",
    "freeze_period",
    "warmup_period",
    "bomb_planted",
    "bomb_dropped",
    "ct_losing_streak",
    "t_losing_streak",
)


class EnemyTrackerFeatureExtractor:
    def __init__(self, seq_len: int | None = None):
        self.seq_len = seq_len

    def schema(self, seq_len: int | None = None) -> FeatureSchema:
        resolved_seq_len = int(seq_len if seq_len is not None else self.seq_len or 0)
        if resolved_seq_len <= 0:
            raise ValueError("EnemyTrackerFeatureExtractor requires seq_len for schema generation.")
        return FeatureSchema(
            model_key="enemy_tracker",
            version=SCHEMA_VERSION,
            seq_len=resolved_seq_len,
            feature_names=TRACKER_FEATURE_NAMES,
            default_value=0.0,
            normalization=dict(NORMALIZATION_CONSTANTS),
        )

    def extract(self, sequence: GameStateSequence) -> np.ndarray:
        roster = build_enemy_roster(sequence)
        last_seen_by_steamid: dict[int, PlayerState] = {}
        frames = [self._state_to_vector(state, roster, last_seen_by_steamid) for state in sequence.states]
        if self.seq_len is not None:
            frames = pad_or_trim_sequence(frames, self.seq_len, self.feature_dim(), default_value=0.0)
        return np.asarray(frames, dtype=np.float32)

    def feature_dim(self) -> int:
        return len(TRACKER_FEATURE_NAMES)

    def _state_to_vector(self, state: GameState, roster: list[int], last_seen_by_steamid: dict[int, PlayerState]) -> list[float]:
        self_player = state.self_player
        vector = [
            *map(normalize_position, self_player.position),
            *map(normalize_velocity, self_player.velocity),
            normalize_hp(self_player.health),
            normalize_armor(self_player.armor),
            bool_to_float(self_player.is_alive),
            normalize_angle(self_player.yaw),
            normalize_angle(self_player.pitch),
        ]
        teammate_pos: list[float] = []
        teammate_alive: list[float] = []
        for teammate in state.teammates[:MAX_TEAMMATES]:
            teammate_pos.extend(normalize_position(v) for v in relative_position(self_player.position, teammate.position))
            teammate_alive.append(bool_to_float(teammate.is_alive))
        vector.extend(pad_or_trim_vector(teammate_pos, MAX_TEAMMATES * 3))
        vector.extend(pad_or_trim_vector(teammate_alive, MAX_TEAMMATES))
        visible_enemies = {int(enemy.steamid): enemy for enemy in state.enemies if enemy.visibility == VisibilityStatus.VISIBLE.value}
        enemy_values: list[float] = []
        for steamid in roster[:MAX_ENEMIES]:
            if steamid in visible_enemies:
                enemy = visible_enemies[steamid]
                last_seen_by_steamid[steamid] = enemy
                rel = [normalize_position(v) for v in relative_position(self_player.position, enemy.position)]
                enemy_values.extend([*rel, 1.0, 0.0, 0.0])
                continue
            if steamid in last_seen_by_steamid:
                enemy = last_seen_by_steamid[steamid]
                rel = [normalize_position(v) for v in relative_position(self_player.position, enemy.position)]
                enemy_values.extend([*rel, 0.0, 1.0, 0.0])
                continue
            enemy_values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        while len(enemy_values) < MAX_ENEMIES * 6:
            enemy_values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        vector.extend(enemy_values)
        vector.extend(
            [
                bool_to_float(state.round.round_in_progress),
                bool_to_float(state.round.is_freeze_period),
                bool_to_float(state.round.is_warmup_period),
                bool_to_float(state.bomb.is_bomb_planted),
                bool_to_float(state.bomb.is_bomb_dropped),
                normalize_hp(state.round.ct_losing_streak),
                normalize_hp(state.round.t_losing_streak),
            ]
        )
        return vector


def build_enemy_roster(sequence: GameStateSequence) -> list[int]:
    roster: list[int] = []
    seen: set[int] = set()
    for state in sequence.states:
        visible_first = sorted(
            (enemy for enemy in state.enemies if enemy.visibility == VisibilityStatus.VISIBLE.value),
            key=lambda item: int(item.steamid),
        )
        for enemy in visible_first:
            steamid = int(enemy.steamid)
            if steamid in seen:
                continue
            seen.add(steamid)
            roster.append(steamid)
            if len(roster) >= MAX_ENEMIES:
                return roster
    while len(roster) < MAX_ENEMIES:
        roster.append(-(len(roster) + 1))
    return roster[:MAX_ENEMIES]


def build_enemy_position_target(game_state: DemoTruthState, roster_steamids: list[int]) -> np.ndarray:
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


def build_enemy_confidence_target(game_state: DemoTruthState, roster_steamids: list[int]) -> np.ndarray:
    enemy_by_steamid = {int(enemy.steamid): enemy for enemy in game_state.enemies}
    confidences: list[float] = []
    for steamid in roster_steamids[:MAX_ENEMIES]:
        enemy = enemy_by_steamid.get(int(steamid))
        confidences.append(1.0 if enemy is not None and enemy.is_alive else 0.0)
    while len(confidences) < MAX_ENEMIES:
        confidences.append(0.0)
    return np.asarray(confidences, dtype=np.float32)
