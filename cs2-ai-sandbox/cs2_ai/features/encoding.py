from __future__ import annotations

from cs2_ai.config import ANGLE_SCALE, ARMOR_SCALE, HP_SCALE, MONEY_SCALE, POSITION_SCALE, VELOCITY_SCALE, WEAPON_TO_ID, weapon_to_id


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
