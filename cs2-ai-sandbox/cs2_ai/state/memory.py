from __future__ import annotations

from collections import deque

from cs2_ai.schemas.game_state import GameState


class TickMemory:
    def __init__(self, max_len: int = 128):
        self.max_len = max_len
        self._memory: deque[GameState] = deque(maxlen=max_len)

    def push(self, game_state: GameState) -> None:
        self._memory.append(game_state)

    def get_sequence(self) -> list[GameState]:
        return list(self._memory)

    def clear(self) -> None:
        self._memory.clear()
