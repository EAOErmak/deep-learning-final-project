from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EnemyPrediction:
    enemy_slot: int
    steamid: int | None
    predicted_position: list[float]
    predicted_velocity: list[float]
    confidence: float
    last_seen_seconds: float | None


@dataclass(slots=True)
class EnemyTrackerOutput:
    predictions: list[EnemyPrediction]


@dataclass(slots=True)
class BeliefStateData:
    predicted_enemies: list[EnemyPrediction]
    top_enemy_rel_pos: list[float]
    top_enemy_confidence: float
    predicted_enemy_count: int
    coarse_enemy_counts: dict[str, float]
    danger_zones: dict[str, float]
    safe_zones: dict[str, float]
    site_control: dict[str, float]


@dataclass(slots=True)
class DecisionOutput:
    strategic_action: str
    tactical_action: str
    target_zone: str | None
    target_position: list[float] | None
    aggression_level: float
    confidence: float


@dataclass(slots=True)
class MovementOutput:
    move_direction: list[float]
    movement_mode: str
    target_position: list[float] | None
    should_jump: bool
    should_crouch: bool


@dataclass(slots=True)
class AimShootOutput:
    aim_delta: list[float]
    aim_position: list[float] | None
    shoot: bool
    rightclick: bool
    burst_length: int
    counter_strafe: bool
    confidence: float


@dataclass(slots=True)
class BuyOutput:
    should_buy: bool
    buy_type: str
    buy_list: list[str]


@dataclass(slots=True)
class InputCommand:
    device: str
    command: str
    value: float | bool | str
    duration_ms: int


@dataclass(slots=True)
class ActionPlan:
    keyboard_inputs: list[InputCommand]
    mouse_inputs: list[InputCommand]
    duration_ms: int
