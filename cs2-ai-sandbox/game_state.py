from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class PlayerState:
    id: str
    name: str | None
    team: str | None
    position: Vector3 | None
    forward: Vector3 | None
    health: int | None
    armor: int | None
    money: int | None
    weapon: str | None
    ammo: int | None
    is_alive: bool | None


@dataclass(slots=True)
class GameState:
    provider: str
    timestamp: float
    controlled_player: PlayerState | None
    players: list[PlayerState]
    raw: dict
