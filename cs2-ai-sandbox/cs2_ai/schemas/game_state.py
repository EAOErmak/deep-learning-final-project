from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VisibilityStatus(StrEnum):
    VISIBLE = "visible"
    LAST_SEEN = "last_seen"
    HIDDEN_TRUTH_ONLY = "hidden_truth_only"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class PlayerState:
    steamid: int
    name: str
    team_num: int
    position: list[float]
    velocity: list[float]
    health: float
    armor: float
    has_helmet: bool
    is_alive: bool
    money: float
    weapon: str
    weapon_id: int
    ammo: float
    total_ammo: float
    pitch: float
    yaw: float
    is_scoped: bool
    is_walking: bool
    is_airborne: bool
    duck_amount: float
    ducking: bool
    shots_fired: int
    flash_duration: float
    spotted: bool
    last_place_name: str
    in_bomb_zone: bool
    in_buy_zone: bool
    which_bomb_zone: int
    visibility: str = VisibilityStatus.VISIBLE.value


@dataclass(slots=True)
class PlayerInputState:
    forward: bool
    back: bool
    left: bool
    right: bool
    fire: bool
    rightclick: bool
    reload: bool
    use: bool
    zoom: bool
    walk: bool
    usercmd_mouse_dx: float
    usercmd_mouse_dy: float
    usercmd_forward_move: float
    usercmd_left_move: float


@dataclass(slots=True)
class RoundState:
    tick: int
    round_number: int
    round_start_time: float
    round_in_progress: bool
    is_freeze_period: bool
    is_warmup_period: bool
    game_phase: int
    round_win_status: int
    round_win_reason: int
    ct_losing_streak: int
    t_losing_streak: int


@dataclass(slots=True)
class BombState:
    is_bomb_planted: bool
    is_bomb_dropped: bool
    bomb_position: list[float] | None


@dataclass(slots=True)
class GameState:
    tick: int
    perspective_steamid: int
    self_player: PlayerState
    self_input: PlayerInputState
    teammates: list[PlayerState]
    enemies: list[PlayerState]
    round: RoundState
    bomb: BombState


@dataclass(slots=True)
class GameStateSequence:
    perspective_steamid: int
    states: list[GameState]


@dataclass(slots=True)
class DemoTruthState:
    tick: int
    perspective_steamid: int
    self_player: PlayerState
    teammates: list[PlayerState]
    enemies: list[PlayerState]
    round: RoundState
    bomb: BombState


@dataclass(slots=True)
class StateBundle:
    observed_state: GameState
    truth_state: DemoTruthState
