from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(slots=True)
class MapState:
    name: str | None = None
    mode: str | None = None
    phase: str | None = None
    round_number: int | None = None
    ct_score: int | None = None
    t_score: int | None = None
    ct_consecutive_round_losses: int | None = None
    t_consecutive_round_losses: int | None = None


@dataclass(slots=True)
class RoundState:
    phase: str | None = None
    bomb_state: str | None = None
    phase_ends_in: float | None = None


@dataclass(slots=True)
class LiveCapabilities:
    has_player_position: bool = False
    has_player_forward: bool = False
    has_allplayers: bool = False
    has_enemy_players: bool = False
    has_spatial_state: bool = False
    has_round_state: bool = False
    has_bomb_state: bool = False


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
    velocity: Vector3 | None = None
    helmet: bool | None = None
    flashed: int | None = None
    smoked: int | None = None
    burning: int | None = None
    round_kills: int | None = None
    round_killhs: int | None = None
    equip_value: int | None = None
    ammo_reserve: int | None = None
    observer_slot: int | None = None
    activity: str | None = None


@dataclass(slots=True)
class GameState:
    provider: str
    timestamp: float
    controlled_player: PlayerState | None
    players: list[PlayerState]
    raw: dict
    map_state: MapState = field(default_factory=MapState)
    round_state: RoundState = field(default_factory=RoundState)
    capabilities: LiveCapabilities = field(default_factory=LiveCapabilities)
