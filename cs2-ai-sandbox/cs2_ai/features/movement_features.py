from __future__ import annotations

import numpy as np
import pandas as pd

from cs2_ai.config import MAX_TEAMMATES
from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_position, normalize_velocity, pad_or_trim_vector, relative_position
from cs2_ai.features.feature_contract import FeatureSchema, NORMALIZATION_CONSTANTS, SCHEMA_VERSION, pad_or_trim_sequence
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, DecisionOutput


MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE = "next_tick_sequence"
MOVEMENT_TARGET_MODE_ACTION_CHUNK = "action_chunk"

MOVEMENT_FEATURE_NAMES = (
    "self_pos_x",
    "self_pos_y",
    "self_pos_z",
    "self_vel_x",
    "self_vel_y",
    "self_vel_z",
    "self_yaw",
    "self_is_walking",
    "self_is_airborne",
    "self_is_ducking",
    "teammate_0_rel_x",
    "teammate_0_rel_y",
    "teammate_0_rel_z",
    "teammate_1_rel_x",
    "teammate_1_rel_y",
    "teammate_1_rel_z",
    "teammate_2_rel_x",
    "teammate_2_rel_y",
    "teammate_2_rel_z",
    "teammate_3_rel_x",
    "teammate_3_rel_y",
    "teammate_3_rel_z",
    "teammate_0_present",
    "teammate_1_present",
    "teammate_2_present",
    "teammate_3_present",
    "target_rel_x",
    "target_rel_y",
    "target_rel_z",
    "belief_top_enemy_rel_x",
    "belief_top_enemy_rel_y",
    "belief_top_enemy_rel_z",
    "belief_top_enemy_confidence",
    "belief_enemy_count",
    "belief_a_count",
    "belief_b_count",
    "belief_mid_count",
)

MOVEMENT_ACTION_NAMES = (
    "forward",
    "back",
    "left",
    "right",
    "walk",
    "crouch",
)

MOVEMENT_ACTION_CHUNK_NAMES = MOVEMENT_ACTION_NAMES + ("jump",)

JUMP_COLUMNS = (
    "JUMP",
    "jump",
    "IN_JUMP",
    "in_jump",
    "usercmd_jump",
    "jump_pressed",
)


class MovementFeatureExtractor:
    def __init__(self, seq_len: int | None = None):
        self.seq_len = seq_len

    def schema(self, seq_len: int | None = None) -> FeatureSchema:
        resolved_seq_len = int(seq_len if seq_len is not None else self.seq_len or 0)
        if resolved_seq_len <= 0:
            raise ValueError("MovementFeatureExtractor requires seq_len for schema generation.")
        return FeatureSchema(
            model_key="movement",
            version=SCHEMA_VERSION,
            seq_len=resolved_seq_len,
            feature_names=MOVEMENT_FEATURE_NAMES,
            default_value=0.0,
            normalization=dict(NORMALIZATION_CONSTANTS),
        )

    def extract(self, sequence, decision: DecisionOutput | None = None, belief_state: BeliefStateData | None = None) -> np.ndarray:
        frames = [self._state_to_vector(state, decision=None, belief_state=None) for state in sequence.states]
        if self.seq_len is not None:
            frames = pad_or_trim_sequence(frames, self.seq_len, self.feature_dim(), default_value=0.0)
        return np.asarray(frames, dtype=np.float32)

    def feature_dim(self) -> int:
        return len(MOVEMENT_FEATURE_NAMES)

    def _state_to_vector(self, state: GameState, decision: DecisionOutput | None, belief_state: BeliefStateData | None) -> list[float]:
        self_player = state.self_player
        teammate_features: list[float] = []
        teammate_present: list[float] = []
        for teammate in state.teammates[:MAX_TEAMMATES]:
            teammate_features.extend(normalize_position(v) for v in relative_position(self_player.position, teammate.position))
            teammate_present.append(1.0)
        target_position = [0.0, 0.0, 0.0]
        top_enemy_rel = [0.0, 0.0, 0.0]
        top_enemy_confidence = 0.0
        predicted_enemy_count = 0.0
        coarse_enemy_counts = [0.0, 0.0, 0.0]
        if belief_state is not None:
            top_enemy_rel = [normalize_position(v) for v in belief_state.top_enemy_rel_pos]
            top_enemy_confidence = float(belief_state.top_enemy_confidence)
            predicted_enemy_count = float(belief_state.predicted_enemy_count) / 5.0
            coarse_enemy_counts = [float(belief_state.coarse_enemy_counts.get(key, 0.0)) / 5.0 for key in ("A", "B", "mid")]
        return [
            *map(normalize_position, self_player.position),
            *map(normalize_velocity, self_player.velocity),
            normalize_angle(self_player.yaw),
            bool_to_float(self_player.is_walking),
            bool_to_float(self_player.is_airborne),
            bool_to_float(self_player.ducking),
            *pad_or_trim_vector(teammate_features, MAX_TEAMMATES * 3),
            *pad_or_trim_vector(teammate_present, MAX_TEAMMATES),
            *[normalize_position(v) for v in relative_position(self_player.position, target_position)],
            *top_enemy_rel,
            top_enemy_confidence,
            predicted_enemy_count,
            *coarse_enemy_counts,
        ]


def movement_action_names_for_target_mode(target_mode: str) -> tuple[str, ...]:
    if target_mode == MOVEMENT_TARGET_MODE_ACTION_CHUNK:
        return MOVEMENT_ACTION_CHUNK_NAMES
    if target_mode == MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE:
        return MOVEMENT_ACTION_NAMES
    raise ValueError(f"Unsupported movement target mode: {target_mode}")


def build_movement_target(game_state: GameState) -> np.ndarray:
    self_input = game_state.self_input
    values = [
        bool_to_float(self_input.forward),
        bool_to_float(self_input.back),
        bool_to_float(self_input.left),
        bool_to_float(self_input.right),
        bool_to_float(self_input.walk),
        bool_to_float(game_state.self_player.ducking),
    ]
    return np.asarray(values, dtype=np.float32)


def build_movement_target_from_tick_rows(tick_rows: pd.DataFrame, perspective_steamid: int) -> np.ndarray:
    if tick_rows.empty or "steamid" not in tick_rows.columns:
        raise ValueError('tick_rows is empty or missing steamid.')
    steamids = pd.to_numeric(tick_rows["steamid"], errors="coerce")
    self_rows = tick_rows.loc[steamids == int(perspective_steamid)]
    if self_rows.empty:
        raise ValueError(f'Perspective player {perspective_steamid} not found on tick rows.')
    self_row = self_rows.iloc[0]
    values = [
        bool_to_float(bool(self_row.get("FORWARD", False))),
        bool_to_float(bool(self_row.get("BACK", False))),
        bool_to_float(bool(self_row.get("LEFT", False))),
        bool_to_float(bool(self_row.get("RIGHT", False))),
        bool_to_float(bool(self_row.get("WALK", self_row.get("is_walking", False)))),
        bool_to_float(bool(self_row.get("ducking", False))),
    ]
    return np.asarray(values, dtype=np.float32)


def extract_jump_target_from_tick_rows(tick_rows: pd.DataFrame, perspective_steamid: int) -> float:
    if tick_rows.empty or "steamid" not in tick_rows.columns:
        return 0.0
    steamids = pd.to_numeric(tick_rows["steamid"], errors="coerce")
    self_rows = tick_rows.loc[steamids == int(perspective_steamid)]
    if self_rows.empty:
        return 0.0
    self_row = self_rows.iloc[0]
    for column in JUMP_COLUMNS:
        if column not in self_row.index or pd.isna(self_row[column]):
            continue
        return bool_to_float(bool(self_row[column]))
    buttons_value = self_row.get("buttons")
    if isinstance(buttons_value, str) and "jump" in buttons_value.lower():
        return 1.0
    return 0.0


def build_movement_action_chunk_target(game_state: GameState, jump_value: float) -> np.ndarray:
    return np.asarray(
        [
            *build_movement_target(game_state).astype(np.float32).tolist(),
            float(np.clip(jump_value, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_movement_action_chunk_target_from_tick_rows(tick_rows: pd.DataFrame, perspective_steamid: int) -> np.ndarray:
    base_target = build_movement_target_from_tick_rows(tick_rows, perspective_steamid)
    jump_value = extract_jump_target_from_tick_rows(tick_rows, perspective_steamid)
    return np.asarray([*base_target.tolist(), float(np.clip(jump_value, 0.0, 1.0))], dtype=np.float32)
