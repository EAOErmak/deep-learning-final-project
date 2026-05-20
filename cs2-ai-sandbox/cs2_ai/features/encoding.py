from __future__ import annotations

import math

from cs2_ai.config import ANGLE_SCALE, ARMOR_SCALE, HP_SCALE, MONEY_SCALE, POSITION_SCALE, VELOCITY_SCALE, WEAPON_TO_ID, weapon_to_id

def calculate_target_angles(player_pos: list[float], enemy_pos: list[float]) -> tuple[float, float]:
    """Returns (target_yaw, target_pitch) in degrees."""
    dx = enemy_pos[0] - player_pos[0]
    dy = enemy_pos[1] - player_pos[1]
    dz = (enemy_pos[2] + 50.0) - (player_pos[2] + 64.0)
    
    yaw = math.atan2(dy, dx) * 180.0 / math.pi
    xy_dist = math.sqrt(dx*dx + dy*dy)
    pitch = math.atan2(-dz, xy_dist) * 180.0 / math.pi
    
    return yaw, pitch

def normalize_angle_delta(delta: float) -> float:
    while delta > 180.0:
        delta -= 360.0
    while delta <= -180.0:
        delta += 360.0
    return delta

def world_to_screen_delta(player_pos: list[float], current_yaw: float, current_pitch: float, enemy_pos: list[float]) -> tuple[float, float]:
    target_yaw, target_pitch = calculate_target_angles(player_pos, enemy_pos)
    delta_yaw = normalize_angle_delta(target_yaw - current_yaw)
    delta_pitch = normalize_angle_delta(target_pitch - current_pitch)
    
    screen_dx = delta_yaw / 45.0
    screen_dy = delta_pitch / 45.0
    
    return max(-4.0, min(4.0, screen_dx)), max(-4.0, min(4.0, screen_dy))

def normalize_position(value: float) -> float:
    return float(value) / POSITION_SCALE


def normalize_velocity(value: float) -> float:
    return float(value) / VELOCITY_SCALE


def normalize_money(value: float) -> float:
    return float(value) / MONEY_SCALE


def normalize_hp(value: float) -> float:
    return float(value) / HP_SCALE


def normalize_armor(value: float) -> float:
    return float(value) / ARMOR_SCALE


def normalize_angle(value: float) -> float:
    return float(value) / ANGLE_SCALE


def bool_to_float(value: bool) -> float:
    return 1.0 if bool(value) else 0.0


def weapon_to_id_normalized(name: str | None) -> float:
    max_id = max(WEAPON_TO_ID.values()) or 1
    return weapon_to_id(name) / float(max_id)


def relative_position(a: list[float], b: list[float]) -> list[float]:
    return [float(b[i]) - float(a[i]) for i in range(3)]


def pad_or_trim_vector(values: list[float], target_len: int) -> list[float]:
    if len(values) >= target_len:
        return values[:target_len]
    return values + [0.0] * (target_len - len(values))
