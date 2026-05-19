from __future__ import annotations

import time
from typing import Any


ActionDict = dict[str, Any]


class DummyAgent:
    """
    Simple rule-based agent for smoke testing the sandbox loop.

    The agent does not know anything about operating system input.
    It only returns action dictionaries that can later be produced by
    imitation learning, reinforcement learning, or sequence models.
    """

    def __init__(self) -> None:
        self._cycle_started_at = time.perf_counter()

        self._timeline: list[tuple[float, ActionDict]] = [
            (
                2.0,
                self._make_action(
                    forward=True,
                ),
            ),
            (
                2.6,
                self._make_action(
                    mouse_dx=18,
                    mouse_dy=0,
                ),
            ),
            (
                4.0,
                self._make_action(
                    right=True,
                ),
            ),
            (
                4.2,
                self._make_action(
                    jump=True,
                ),
            ),
            (
                5.2,
                self._make_action(),
            ),
            (
                5.6,
                self._make_action(
                    fire=True,
                ),
            ),
            (
                6.0,
                self._make_action(),
            ),
        ]
        self._cycle_duration = self._timeline[-1][0]

    def predict(self, _features: dict[str, float | int | bool]) -> ActionDict:
        cycle_time = (time.perf_counter() - self._cycle_started_at) % self._cycle_duration

        for threshold, action in self._timeline:
            if cycle_time < threshold:
                return action.copy()

        return self._make_action()

    @staticmethod
    def _make_action(
        *,
        forward: bool = False,
        back: bool = False,
        left: bool = False,
        right: bool = False,
        jump: bool = False,
        crouch: bool = False,
        walk: bool = False,
        fire: bool = False,
        mouse_dx: int = 0,
        mouse_dy: int = 0,
    ) -> ActionDict:
        return {
            "forward": forward,
            "back": back,
            "left": left,
            "right": right,
            "jump": jump,
            "crouch": crouch,
            "walk": walk,
            "fire": fire,
            "mouse_dx": mouse_dx,
            "mouse_dy": mouse_dy,
        }
