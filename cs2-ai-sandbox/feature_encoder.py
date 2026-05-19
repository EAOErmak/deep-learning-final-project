from __future__ import annotations

import math
from typing import Any

from game_state import GameState, PlayerState


ZERO_FEATURES: dict[str, float | int | bool] = {
    'self_x': 0.0,
    'self_y': 0.0,
    'self_z': 0.0,
    'self_hp': 0,
    'self_money': 0,
    'ammo': 0,
    'yaw': 0.0,
    'pitch': 0.0,
    'enemy_visible': 0,
    'enemy_rel_x': 0.0,
    'enemy_rel_y': 0.0,
    'enemy_rel_z': 0.0,
    'enemy_hp': 0,
    'enemy_distance': 0.0,
}


def encode_state(raw_state: dict[str, Any] | GameState) -> dict[str, float | int | bool]:
    if isinstance(raw_state, GameState):
        return _encode_gsi_state(raw_state)
    return _encode_mock_state(raw_state)


def _encode_mock_state(raw_state: dict[str, Any]) -> dict[str, float | int | bool]:
    enemy_visible = bool(raw_state.get('enemy_visible', False))
    enemy_rel = raw_state.get('enemy_relative_position', {})
    enemy_hp = int(raw_state.get('enemy_hp', 0)) if enemy_visible else 0
    enemy_distance = math.sqrt(
        float(enemy_rel.get('x', 0.0)) ** 2 + float(enemy_rel.get('y', 0.0)) ** 2 + float(enemy_rel.get('z', 0.0)) ** 2
    ) if enemy_visible else 0.0
    return {
        'self_x': float(raw_state.get('self_x', 0.0)),
        'self_y': float(raw_state.get('self_y', 0.0)),
        'self_z': float(raw_state.get('self_z', 0.0)),
        'self_hp': int(raw_state.get('self_hp', 0)),
        'self_money': int(raw_state.get('self_money', 0)),
        'ammo': int(raw_state.get('ammo', 0)),
        'yaw': float(raw_state.get('yaw', 0.0)),
        'pitch': float(raw_state.get('pitch', 0.0)),
        'enemy_visible': int(enemy_visible),
        'enemy_rel_x': float(enemy_rel.get('x', 0.0)) if enemy_visible else 0.0,
        'enemy_rel_y': float(enemy_rel.get('y', 0.0)) if enemy_visible else 0.0,
        'enemy_rel_z': float(enemy_rel.get('z', 0.0)) if enemy_visible else 0.0,
        'enemy_hp': enemy_hp,
        'enemy_distance': enemy_distance,
    }


def _encode_gsi_state(game_state: GameState) -> dict[str, float | int | bool]:
    controlled = game_state.controlled_player
    if controlled is None:
        return dict(ZERO_FEATURES)

    position = controlled.position
    features: dict[str, float | int | bool] = {
        'self_x': float(position.x) if position is not None else 0.0,
        'self_y': float(position.y) if position is not None else 0.0,
        'self_z': float(position.z) if position is not None else 0.0,
        'self_hp': int(controlled.health or 0),
        'self_money': int(controlled.money or 0),
        'ammo': int(controlled.ammo or 0),
        'yaw': 0.0,
        'pitch': 0.0,
        'enemy_visible': 0,
        'enemy_rel_x': 0.0,
        'enemy_rel_y': 0.0,
        'enemy_rel_z': 0.0,
        'enemy_hp': 0,
        'enemy_distance': 0.0,
    }
    if position is None:
        return features

    enemies = [player for player in game_state.players if _is_enemy(controlled, player) and player.position is not None and player.is_alive]
    nearest_enemy = min(enemies, key=lambda player: _distance(controlled, player), default=None)

    if nearest_enemy is None or nearest_enemy.position is None:
        return features

    enemy_rel_x = nearest_enemy.position.x - position.x
    enemy_rel_y = nearest_enemy.position.y - position.y
    enemy_rel_z = nearest_enemy.position.z - position.z
    features['enemy_visible'] = 1
    features['enemy_rel_x'] = enemy_rel_x
    features['enemy_rel_y'] = enemy_rel_y
    features['enemy_rel_z'] = enemy_rel_z
    features['enemy_hp'] = int(nearest_enemy.health or 0)
    features['enemy_distance'] = math.sqrt(enemy_rel_x ** 2 + enemy_rel_y ** 2 + enemy_rel_z ** 2)
    return features


def _is_enemy(controlled: PlayerState, candidate: PlayerState) -> bool:
    return candidate.id != controlled.id and candidate.team != controlled.team


def _distance(controlled: PlayerState, enemy: PlayerState) -> float:
    if controlled.position is None or enemy.position is None:
        return float('inf')
    return math.sqrt(
        (enemy.position.x - controlled.position.x) ** 2
        + (enemy.position.y - controlled.position.y) ** 2
        + (enemy.position.z - controlled.position.z) ** 2
    )
