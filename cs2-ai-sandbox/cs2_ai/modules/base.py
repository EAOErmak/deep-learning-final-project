from __future__ import annotations

from typing import Protocol

from cs2_ai.schemas.game_state import GameState, GameStateSequence
from cs2_ai.schemas.module_outputs import AimShootOutput, BeliefStateData, BuyOutput, DecisionOutput, EnemyTrackerOutput, MovementOutput


class EnemyTrackerModule(Protocol):
    def reset(self) -> None:
        ...

    def predict(self, sequence: GameStateSequence) -> EnemyTrackerOutput:
        ...


class DecisionModule(Protocol):
    def reset(self) -> None:
        ...

    def decide(self, game_state: GameState, belief_state: BeliefStateData) -> DecisionOutput:
        ...


class MovementModule(Protocol):
    def reset(self) -> None:
        ...

    def decide(self, game_state: GameState, belief_state: BeliefStateData, decision: DecisionOutput) -> MovementOutput:
        ...


class AimShootModule(Protocol):
    def reset(self) -> None:
        ...

    def decide(self, game_state: GameState, belief_state: BeliefStateData, decision: DecisionOutput) -> AimShootOutput:
        ...


class BuyModule(Protocol):
    def reset(self) -> None:
        ...

    def decide(self, game_state: GameState) -> BuyOutput:
        ...
