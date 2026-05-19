from __future__ import annotations

from dataclasses import dataclass

from .game_state import GameState, GameStateSequence, PlayerInputState


@dataclass(slots=True)
class PerspectiveSample:
    perspective_steamid: int
    tick: int
    round_number: int
    game_state: GameState
    target_input: PlayerInputState


@dataclass(slots=True)
class SequenceSample:
    perspective_steamid: int
    start_tick: int
    end_tick: int
    round_number: int
    sequence: GameStateSequence
    target_input: PlayerInputState


@dataclass(slots=True)
class RLTransition:
    perspective_steamid: int
    state_sequence: GameStateSequence
    action_id: int
    reward: float
    next_state_sequence: GameStateSequence
    done: bool
