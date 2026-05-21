from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cs2_ai.config import MAX_TEAMMATES
from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_position, normalize_velocity, pad_or_trim_vector, relative_position
from cs2_ai.features.feature_contract import FeatureSchema, NORMALIZATION_CONSTANTS, SCHEMA_VERSION, pad_or_trim_sequence
from cs2_ai.schemas.game_state import GameState
from cs2_ai.schemas.module_outputs import BeliefStateData, DecisionOutput


MOVEMENT_TARGET_MODE_NEXT_TICK = "next_tick"
MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE = MOVEMENT_TARGET_MODE_NEXT_TICK
MOVEMENT_TARGET_MODE_ACTION_CHUNK = "action_chunk"
MOVEMENT_FEATURE_MODE_LEGACY = "legacy"
MOVEMENT_FEATURE_MODE_SOLO_GRID = "solo_grid"

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

MOVEMENT_FEATURE_NAMES_SOLO_GRID = (
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
    "next_cell_rel_x",
    "next_cell_rel_y",
    "next_cell_rel_z",
    "next_cell_distance",
    "has_next_cell_target",
    "dwell_pass_through",
    "dwell_short_hold",
    "dwell_medium_hold",
    "dwell_long_hold",
)

GRID_NAVIGATION_FEATURE_NAMES = (
    "next_cell_rel_x",
    "next_cell_rel_y",
    "next_cell_rel_z",
    "next_cell_distance",
    "has_next_cell_target",
    "dwell_pass_through",
    "dwell_short_hold",
    "dwell_medium_hold",
    "dwell_long_hold",
)

GRID_NAVIGATION_REQUIRED_COLUMNS = GRID_NAVIGATION_FEATURE_NAMES

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

logger = logging.getLogger(__name__)
_missing_jump_warning_emitted = False


class MovementFeatureExtractor:
    def __init__(
        self,
        seq_len: int | None = None,
        use_grid_navigation_features: bool = False,
        movement_feature_mode: str = MOVEMENT_FEATURE_MODE_LEGACY,
    ):
        self.seq_len = seq_len
        self.movement_feature_mode = normalize_movement_feature_mode(movement_feature_mode)
        self.use_grid_navigation_features = bool(use_grid_navigation_features) or self.movement_feature_mode == MOVEMENT_FEATURE_MODE_SOLO_GRID

    @property
    def requires_grid_navigation_features(self) -> bool:
        return self.use_grid_navigation_features

    def feature_names(self) -> tuple[str, ...]:
        if self.movement_feature_mode == MOVEMENT_FEATURE_MODE_SOLO_GRID:
            return MOVEMENT_FEATURE_NAMES_SOLO_GRID
        if self.use_grid_navigation_features:
            return MOVEMENT_FEATURE_NAMES + GRID_NAVIGATION_FEATURE_NAMES
        return MOVEMENT_FEATURE_NAMES

    def schema(self, seq_len: int | None = None) -> FeatureSchema:
        resolved_seq_len = int(seq_len if seq_len is not None else self.seq_len or 0)
        if resolved_seq_len <= 0:
            raise ValueError("MovementFeatureExtractor requires seq_len for schema generation.")
        return FeatureSchema(
            model_key="movement",
            version=SCHEMA_VERSION,
            seq_len=resolved_seq_len,
            feature_names=self.feature_names(),
            default_value=0.0,
            normalization=dict(NORMALIZATION_CONSTANTS),
        )

    def extract(
        self,
        sequence,
        decision: DecisionOutput | None = None,
        belief_state: BeliefStateData | None = None,
        grid_navigation_frames: list[dict[str, float]] | None = None,
    ) -> np.ndarray:
        frames = []
        for idx, state in enumerate(sequence.states):
            base_vector = self._state_to_vector(state, decision=None, belief_state=None)
            if self.use_grid_navigation_features:
                nav_frame = grid_navigation_frames[idx] if grid_navigation_frames is not None and idx < len(grid_navigation_frames) else {}
                base_vector.extend(self._grid_navigation_vector(nav_frame))
            frames.append(base_vector)
        if self.seq_len is not None:
            frames = pad_or_trim_sequence(frames, self.seq_len, self.feature_dim(), default_value=0.0)
        return np.asarray(frames, dtype=np.float32)

    def feature_dim(self) -> int:
        return len(self.feature_names())

    def _state_to_vector(self, state: GameState, decision: DecisionOutput | None, belief_state: BeliefStateData | None) -> list[float]:
        if self.movement_feature_mode == MOVEMENT_FEATURE_MODE_SOLO_GRID:
            self_player = state.self_player
            return [
                float(self_player.position[0]) / 4000.0,
                float(self_player.position[1]) / 4000.0,
                float(self_player.position[2]) / 512.0,
                float(self_player.velocity[0]) / 500.0,
                float(self_player.velocity[1]) / 500.0,
                float(self_player.velocity[2]) / 500.0,
                normalize_angle(self_player.yaw),
                bool_to_float(self_player.is_walking),
                bool_to_float(self_player.is_airborne),
                bool_to_float(self_player.ducking),
            ]
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

    def _grid_navigation_vector(self, nav_frame: dict[str, float] | None) -> list[float]:
        nav_frame = nav_frame or {}
        return [
            float(nav_frame.get("next_cell_rel_x", 0.0)) / 1000.0,
            float(nav_frame.get("next_cell_rel_y", 0.0)) / 1000.0,
            float(nav_frame.get("next_cell_rel_z", 0.0)) / 256.0,
            float(nav_frame.get("next_cell_distance", 0.0)) / 1000.0,
            float(nav_frame.get("has_next_cell_target", 0.0)),
            float(nav_frame.get("dwell_pass_through", 0.0)),
            float(nav_frame.get("dwell_short_hold", 0.0)),
            float(nav_frame.get("dwell_medium_hold", 0.0)),
            float(nav_frame.get("dwell_long_hold", 0.0)),
        ]


def normalize_movement_feature_mode(movement_feature_mode: str | None) -> str:
    normalized = str(movement_feature_mode or MOVEMENT_FEATURE_MODE_LEGACY).strip().lower()
    if normalized in {MOVEMENT_FEATURE_MODE_LEGACY, "default"}:
        return MOVEMENT_FEATURE_MODE_LEGACY
    if normalized == MOVEMENT_FEATURE_MODE_SOLO_GRID:
        return MOVEMENT_FEATURE_MODE_SOLO_GRID
    raise ValueError(f"Unsupported movement feature mode: {movement_feature_mode}")


def build_grid_navigation_feature_frame_from_row(row: pd.Series, *, strict: bool = False) -> dict[str, float]:
    missing = [column for column in GRID_NAVIGATION_REQUIRED_COLUMNS if column not in row.index]
    if missing:
        if strict:
            raise ValueError(
                'Grid navigation features were requested, but required columns are missing: '
                f'{missing}. Run label_dust2_grid preprocessing first.'
            )
        return {column: 0.0 for column in GRID_NAVIGATION_REQUIRED_COLUMNS}
    result: dict[str, float] = {}
    for column in GRID_NAVIGATION_REQUIRED_COLUMNS:
        value = row.get(column, 0.0)
        result[column] = 0.0 if pd.isna(value) else float(value)
    return result


def movement_action_names_for_target_mode(target_mode: str) -> tuple[str, ...]:
    target_mode = normalize_movement_target_mode(target_mode)
    if target_mode == MOVEMENT_TARGET_MODE_ACTION_CHUNK:
        return MOVEMENT_ACTION_CHUNK_NAMES
    if target_mode == MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE:
        return MOVEMENT_ACTION_NAMES
    raise ValueError(f"Unsupported movement target mode: {target_mode}")


def normalize_movement_target_mode(target_mode: str | None) -> str:
    normalized = str(target_mode or MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE).strip().lower()
    if normalized in {MOVEMENT_TARGET_MODE_NEXT_TICK, "next_tick_sequence"}:
        return MOVEMENT_TARGET_MODE_NEXT_TICK_SEQUENCE
    if normalized == MOVEMENT_TARGET_MODE_ACTION_CHUNK:
        return MOVEMENT_TARGET_MODE_ACTION_CHUNK
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
        bool_to_float(bool(self_row.get("move_forward", self_row.get("FORWARD", False)))),
        bool_to_float(bool(self_row.get("move_back", self_row.get("BACK", False)))),
        bool_to_float(bool(self_row.get("move_left", self_row.get("LEFT", False)))),
        bool_to_float(bool(self_row.get("move_right", self_row.get("RIGHT", False)))),
        bool_to_float(bool(self_row.get("move_walk", self_row.get("WALK", self_row.get("is_walking", False))))),
        bool_to_float(bool(self_row.get("move_crouch", self_row.get("ducking", False)))),
    ]
    return np.asarray(values, dtype=np.float32)


def extract_jump_target_from_tick_rows(tick_rows: pd.DataFrame, perspective_steamid: int) -> float:
    global _missing_jump_warning_emitted
    if tick_rows.empty or "steamid" not in tick_rows.columns:
        return 0.0
    steamids = pd.to_numeric(tick_rows["steamid"], errors="coerce")
    self_rows = tick_rows.loc[steamids == int(perspective_steamid)]
    if self_rows.empty:
        return 0.0
    self_row = self_rows.iloc[0]
    if "move_jump" in self_row.index and not pd.isna(self_row["move_jump"]):
        return bool_to_float(bool(self_row["move_jump"]))
    for column in JUMP_COLUMNS:
        if column not in self_row.index or pd.isna(self_row[column]):
            continue
        return bool_to_float(bool(self_row[column]))
    buttons_value = self_row.get("buttons")
    if isinstance(buttons_value, str) and "jump" in buttons_value.lower():
        return 1.0
    if not _missing_jump_warning_emitted:
        _missing_jump_warning_emitted = True
        logger.warning(
            "Movement jump target is missing explicit jump/button fields for at least one sample. "
            "Falling back to 0.0 for jump."
        )
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
