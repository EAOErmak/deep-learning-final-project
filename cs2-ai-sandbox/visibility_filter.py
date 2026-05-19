from __future__ import annotations

import math
from dataclasses import replace

from game_state import GameState, PlayerState, Vector3


def _vector_to_tuple(vec: Vector3) -> tuple[float, float, float]:
    return (vec.x, vec.y, vec.z)


def _distance(a: Vector3, b: Vector3) -> float:
    ax, ay, az = _vector_to_tuple(a)
    bx, by, bz = _vector_to_tuple(b)
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def _normalize(vec: Vector3) -> tuple[float, float, float]:
    x, y, z = _vector_to_tuple(vec)
    length = math.sqrt(x * x + y * y + z * z)
    if length == 0:
        return (0.0, 0.0, 0.0)
    return (x / length, y / length, z / length)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def filter_visible_enemies(game_state: GameState, fov_degrees: float = 90.0, max_distance: float = 3000.0) -> GameState:
    controlled = game_state.controlled_player
    if controlled is None or controlled.position is None or controlled.forward is None:
        visible_players = [player for player in game_state.players if controlled is None or player.team == controlled.team or player.id == controlled.id]
        return replace(game_state, players=visible_players)

    forward_norm = _normalize(controlled.forward)
    fov_threshold = math.cos(math.radians(fov_degrees / 2.0))
    filtered_players: list[PlayerState] = []

    for player in game_state.players:
        if player.id == controlled.id:
            filtered_players.append(player)
            continue
        if player.team == controlled.team:
            filtered_players.append(player)
            continue
        if not player.is_alive or player.position is None:
            continue
        distance = _distance(controlled.position, player.position)
        if distance > max_distance:
            continue
        direction = Vector3(
            x=player.position.x - controlled.position.x,
            y=player.position.y - controlled.position.y,
            z=player.position.z - controlled.position.z,
        )
        direction_norm = _normalize(direction)
        if _dot(forward_norm, direction_norm) >= fov_threshold:
            filtered_players.append(player)

    # TODO: later replace FOV-only visibility with map raycast / line-of-sight.
    return replace(game_state, players=filtered_players)
