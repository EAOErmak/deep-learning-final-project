from __future__ import annotations

import math
from typing import Any

from game_state import GameState, PlayerState, Vector3


ZERO_FEATURES: dict[str, float | int | bool] = {
    'self_x': 0.0,
    'self_y': 0.0,
    'self_z': 0.0,
    'self_vel_x': 0.0,
    'self_vel_y': 0.0,
    'self_vel_z': 0.0,
    'self_hp': 0,
    'self_armor': 0,
    'self_money': 0,
    'self_alive': 0,
    'ammo': 0,
    'ammo_reserve': 0,
    'yaw': 0.0,
    'pitch': 0.0,
    'team_is_ct': 0,
    'round_live': 0,
    'round_freeze': 0,
    'round_warmup': 0,
    'bomb_planted': 0,
    'visible_players_count': 0,
    'available_players_count': 0,
    'has_spatial_state': 0,
    'has_enemy_context': 0,
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
    features = dict(ZERO_FEATURES)
    features.update(
        {
            'self_x': float(raw_state.get('self_x', 0.0)),
            'self_y': float(raw_state.get('self_y', 0.0)),
            'self_z': float(raw_state.get('self_z', 0.0)),
            'self_hp': int(raw_state.get('self_hp', 0)),
            'self_money': int(raw_state.get('self_money', 0)),
            'ammo': int(raw_state.get('ammo', 0)),
            'yaw': float(raw_state.get('yaw', 0.0)),
            'pitch': float(raw_state.get('pitch', 0.0)),
            'has_spatial_state': 1,
            'has_enemy_context': int(enemy_visible),
            'enemy_visible': int(enemy_visible),
            'enemy_rel_x': float(enemy_rel.get('x', 0.0)) if enemy_visible else 0.0,
            'enemy_rel_y': float(enemy_rel.get('y', 0.0)) if enemy_visible else 0.0,
            'enemy_rel_z': float(enemy_rel.get('z', 0.0)) if enemy_visible else 0.0,
            'enemy_hp': enemy_hp,
            'enemy_distance': enemy_distance,
        }
    )
    return features


def _encode_gsi_state(game_state: GameState) -> dict[str, float | int | bool]:
    controlled = game_state.controlled_player
    if controlled is None:
        return dict(ZERO_FEATURES)

    position = controlled.position
    velocity = controlled.velocity
    phase = (game_state.round_state.phase or game_state.map_state.phase or '').lower()
    bomb_state = (game_state.round_state.bomb_state or '').lower()
    enemies = [player for player in game_state.players if _is_enemy(controlled, player) and player.position is not None and player.is_alive]
    nearest_enemy = min(enemies, key=lambda player: _distance(controlled, player), default=None)

    features = dict(ZERO_FEATURES)
    features.update(
        {
            'self_x': float(position.x) if position is not None else 0.0,
            'self_y': float(position.y) if position is not None else 0.0,
            'self_z': float(position.z) if position is not None else 0.0,
            'self_vel_x': float(velocity.x) if velocity is not None else 0.0,
            'self_vel_y': float(velocity.y) if velocity is not None else 0.0,
            'self_vel_z': float(velocity.z) if velocity is not None else 0.0,
            'self_hp': int(controlled.health or 0),
            'self_armor': int(controlled.armor or 0),
            'self_money': int(controlled.money or 0),
            'self_alive': int(bool(controlled.is_alive)),
            'ammo': int(controlled.ammo or 0),
            'ammo_reserve': int(controlled.ammo_reserve or 0),
            'yaw': _yaw_from_forward(controlled.forward),
            'pitch': _pitch_from_forward(controlled.forward),
            'team_is_ct': int((controlled.team or '').upper() == 'CT'),
            'round_live': int(phase == 'live'),
            'round_freeze': int(phase in {'freezetime', 'freeze'}),
            'round_warmup': int(phase == 'warmup'),
            'bomb_planted': int(bomb_state == 'planted'),
            'visible_players_count': len(enemies),
            'available_players_count': len(game_state.players),
            'has_spatial_state': int(game_state.capabilities.has_spatial_state),
            'has_enemy_context': int(game_state.capabilities.has_enemy_players),
        }
    )

    if nearest_enemy is None or nearest_enemy.position is None or position is None:
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


def _yaw_from_forward(forward: Vector3 | None) -> float:
    if forward is None:
        return 0.0
    if forward.x == 0.0 and forward.y == 0.0:
        return 0.0
    return math.degrees(math.atan2(forward.y, forward.x))


def _pitch_from_forward(forward: Vector3 | None) -> float:
    if forward is None:
        return 0.0
    horizontal = math.sqrt(forward.x ** 2 + forward.y ** 2)
    if horizontal == 0.0 and forward.z == 0.0:
        return 0.0
    return math.degrees(math.atan2(forward.z, horizontal))
