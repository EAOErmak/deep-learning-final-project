from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from game_state import GameState
from gsi_server import GSIServer
from gsi_state_reader import GSIStateReader
from visibility_filter import filter_visible_enemies
from cs2_ai.vision.radar_pipeline import RadarVisionModule, augment_live_state_with_radar


@dataclass
class MockStateReader:
    started_at: float = time.perf_counter()

    def read_state(self) -> dict[str, Any]:
        t = time.perf_counter() - self.started_at
        enemy_visible = int(t) % 6 < 3
        return {
            'self_x': round(120.0 + math.sin(t) * 25.0, 2),
            'self_y': round(80.0 + math.cos(t * 0.8) * 20.0, 2),
            'self_z': 4.0,
            'self_hp': max(1, 100 - int((t * 2) % 15)),
            'self_money': 800 + int((t * 100) % 2500),
            'ammo': 30 - int(t % 7),
            'yaw': round((t * 18.0) % 360.0, 2),
            'pitch': round(math.sin(t * 0.4) * 8.0, 2),
            'enemy_visible': enemy_visible,
            'enemy_relative_position': {
                'x': round(math.cos(t * 1.1) * 15.0, 2),
                'y': round(math.sin(t * 0.9) * 12.0, 2),
                'z': 0.0,
            },
            'enemy_hp': 87 if enemy_visible else 0,
        }


class StateReader:
    def __init__(self, mode: str = 'mock', gsi_server: GSIServer | None = None, radar_vision: RadarVisionModule | None = None):
        self.mode = mode
        self.mock_reader = MockStateReader()
        self.gsi_reader = GSIStateReader(gsi_server) if mode == 'gsi' and gsi_server is not None else None
        self.radar_vision = radar_vision

    def read_state(self) -> dict[str, Any] | GameState | None:
        if self.mode == 'mock':
            return self.mock_reader.read_state()
        if self.mode == 'gsi':
            if self.gsi_reader is None:
                raise RuntimeError('GSI mode requested but GSIStateReader is not configured.')
            state = self.gsi_reader.read_state()
            if state is None:
                return None
            if self.radar_vision is not None:
                state = augment_live_state_with_radar(state, self.radar_vision.capture())
            return filter_visible_enemies(state)
        raise ValueError(f'Unsupported state reader mode: {self.mode}')
