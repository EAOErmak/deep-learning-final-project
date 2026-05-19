from __future__ import annotations

POSITION_SCALE = 10000.0
VELOCITY_SCALE = 1000.0
MONEY_SCALE = 16000.0
HP_SCALE = 100.0
ARMOR_SCALE = 100.0
ANGLE_SCALE = 180.0
DEFAULT_SEQ_LEN = 128
DEFAULT_STRIDE = 8
MAX_TEAMMATES = 4
MAX_ENEMIES = 5

WEAPON_TO_ID = {
    "none": 0,
    "Glock-18": 1,
    "USP-S": 2,
    "P2000": 3,
    "Desert Eagle": 4,
    "Five-SeveN": 5,
    "Tec-9": 6,
    "P250": 7,
    "AK-47": 8,
    "M4A1-S": 9,
    "M4A4": 10,
    "AWP": 11,
    "Galil AR": 12,
    "FAMAS": 13,
    "MP9": 14,
    "MAC-10": 15,
    "UMP-45": 16,
    "SSG 08": 17,
}


def weapon_to_id(name: str | None) -> int:
    if not name:
        return 0
    return WEAPON_TO_ID.get(str(name), 0)
