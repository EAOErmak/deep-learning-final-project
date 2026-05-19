from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class MockStateReader:
    """
    Mock state source for local experiments.

    This class intentionally hides the state source behind a small
    interface so the project can later swap to a real provider such as:
    - Counter-Strike 2 Game State Integration (GSI)
    - replay-derived state, for example from demoparser2
    """

    started_at: float = time.perf_counter()

    def read_state(self) -> dict[str, Any]:
        t = time.perf_counter() - self.started_at

        enemy_visible = int(t) % 6 < 3

        return {
            "self_x": round(120.0 + math.sin(t) * 25.0, 2),
            "self_y": round(80.0 + math.cos(t * 0.8) * 20.0, 2),
            "self_z": 4.0,
            "self_hp": max(1, 100 - int((t * 2) % 15)),
            "self_money": 800 + int((t * 100) % 2500),
            "ammo": 30 - int(t % 7),
            "yaw": round((t * 18.0) % 360.0, 2),
            "pitch": round(math.sin(t * 0.4) * 8.0, 2),
            "enemy_visible": enemy_visible,
            "enemy_relative_position": {
                "x": round(math.cos(t * 1.1) * 15.0, 2),
                "y": round(math.sin(t * 0.9) * 12.0, 2),
                "z": 0.0,
            },
            "enemy_hp": 87 if enemy_visible else 0,
        }
