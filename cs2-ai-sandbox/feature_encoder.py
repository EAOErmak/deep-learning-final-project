from __future__ import annotations

from typing import Any


def encode_state(raw_state: dict[str, Any]) -> dict[str, float | int | bool]:
    """
    Converts raw game state into a stable feature dictionary.

    Today the input is a mock object. Later the same function can accept
    GSI payloads, replay parser output, or other structured observations.
    """

    enemy_visible = bool(raw_state.get("enemy_visible", False))
    enemy_rel = raw_state.get("enemy_relative_position", {})
    enemy_hp = int(raw_state.get("enemy_hp", 0)) if enemy_visible else 0

    return {
        "self_x": float(raw_state.get("self_x", 0.0)),
        "self_y": float(raw_state.get("self_y", 0.0)),
        "self_z": float(raw_state.get("self_z", 0.0)),
        "self_hp": int(raw_state.get("self_hp", 0)),
        "self_money": int(raw_state.get("self_money", 0)),
        "ammo": int(raw_state.get("ammo", 0)),
        "yaw": float(raw_state.get("yaw", 0.0)),
        "pitch": float(raw_state.get("pitch", 0.0)),
        "enemy_visible": enemy_visible,
        "enemy_rel_x": float(enemy_rel.get("x", 0.0)) if enemy_visible else 0.0,
        "enemy_rel_y": float(enemy_rel.get("y", 0.0)) if enemy_visible else 0.0,
        "enemy_rel_z": float(enemy_rel.get("z", 0.0)) if enemy_visible else 0.0,
        "enemy_hp": enemy_hp,
    }
