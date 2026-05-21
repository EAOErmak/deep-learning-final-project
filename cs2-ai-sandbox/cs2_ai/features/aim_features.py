from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs2_ai.features.encoding import bool_to_float, normalize_angle, normalize_hp, normalize_velocity, world_to_screen_delta, weapon_to_id_normalized
from cs2_ai.features.feature_contract import FeatureSchema, NORMALIZATION_CONSTANTS, SCHEMA_VERSION, pad_or_trim_sequence
from cs2_ai.schemas.game_state import GameState, GameStateSequence, VisibilityStatus
from cs2_ai.schemas.module_outputs import BeliefStateData

try:
    from cs2_ai.vision.yolo_pipeline import VisionTarget
except ImportError:
    VisionTarget = None


AIM_MOUSE_SCALE = 500.0
AIM_FEATURE_MODE_DEMO_PROJECTED = "demo_projected"
AIM_FEATURE_MODE_VISION_LIKE = "vision_like"
AIM_VISION_SCHEMA_VERSION = "v2_vision"

AIM_FEATURE_NAMES = (
    "self_yaw",
    "self_pitch",
    "self_vel_x",
    "self_vel_y",
    "self_vel_z",
    "weapon_id",
    "ammo",
    "shots_fired",
    "is_scoped",
    "is_walking",
    "is_airborne",
    "flash_duration",
    "enemy_screen_dx",
    "enemy_screen_dy",
    "enemy_visible_mask",
    "enemy_last_seen_mask",
    "enemy_unavailable_mask",
    "has_observed_target_mask",
)

AIM_VISION_FEATURE_NAMES = (
    "vision_enemy_visible",
    "vision_screen_dx",
    "vision_screen_dy",
    "vision_confidence",
    "vision_is_head_target",
)

AIM_BINARY_ACTION_NAMES = (
    "fire",
    "rightclick",
    "zoom",
)


@dataclass(frozen=True, slots=True)
class AimStructuredTarget:
    aim_delta: np.ndarray
    binary_actions: np.ndarray
    valid_aim_mask: np.ndarray


def normalize_mouse_delta(value: float) -> float:
    return float(value) / AIM_MOUSE_SCALE


def denormalize_mouse_delta(value: float) -> float:
    return float(value) * AIM_MOUSE_SCALE


def aim_feature_names_for_mode(feature_mode: str) -> tuple[str, ...]:
    validated_mode = validate_aim_feature_mode(feature_mode)
    if validated_mode == AIM_FEATURE_MODE_VISION_LIKE:
        return AIM_FEATURE_NAMES + AIM_VISION_FEATURE_NAMES
    return AIM_FEATURE_NAMES


def aim_schema_version_for_mode(feature_mode: str) -> str:
    validated_mode = validate_aim_feature_mode(feature_mode)
    if validated_mode == AIM_FEATURE_MODE_VISION_LIKE:
        return AIM_VISION_SCHEMA_VERSION
    return SCHEMA_VERSION


def aim_schema_supports_vision_metadata(metadata: dict[str, object] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    feature_names = tuple(str(name) for name in metadata.get("feature_names", ()))
    return tuple(feature_names) == aim_feature_names_for_mode(AIM_FEATURE_MODE_VISION_LIKE)


def validate_aim_feature_mode(feature_mode: str) -> str:
    mode = str(feature_mode or AIM_FEATURE_MODE_DEMO_PROJECTED)
    if mode not in {AIM_FEATURE_MODE_DEMO_PROJECTED, AIM_FEATURE_MODE_VISION_LIKE}:
        raise ValueError(f"Unsupported aim feature mode: {mode}")
    return mode


def build_demo_projected_vision_target(state: GameState) -> VisionTarget | None:
    if VisionTarget is None:
        return None
    self_player = state.self_player
    visible_enemy = next((enemy for enemy in state.enemies if enemy.visibility == VisibilityStatus.VISIBLE.value and enemy.is_alive), None)
    if visible_enemy is None or self_player.position is None or visible_enemy.position is None:
        return None
    screen_dx, screen_dy = world_to_screen_delta(
        self_player.position,
        self_player.yaw,
        self_player.pitch,
        visible_enemy.position,
    )
    label = "ct_head" if int(visible_enemy.team_num) == 3 else "t_head"
    return VisionTarget(
        screen_dx=float(screen_dx),
        screen_dy=float(screen_dy),
        confidence=1.0,
        label=label,
    )


def vision_target_to_feature_values(vision_target: VisionTarget | None) -> list[float]:
    if vision_target is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    label = str(getattr(vision_target, "label", "") or "").strip().lower()
    is_head_target = 1.0 if ("head" in label or label in {"ch", "th", "ct_head", "t_head"}) else 0.0
    return [
        1.0,
        float(getattr(vision_target, "screen_dx", 0.0)),
        float(getattr(vision_target, "screen_dy", 0.0)),
        float(getattr(vision_target, "confidence", 0.0)),
        is_head_target,
    ]


class AimFeatureExtractor:
    def __init__(self, seq_len: int | None = None, feature_mode: str = AIM_FEATURE_MODE_DEMO_PROJECTED):
        self.seq_len = seq_len
        self.feature_mode = validate_aim_feature_mode(feature_mode)

    def schema(self, seq_len: int | None = None) -> FeatureSchema:
        resolved_seq_len = int(seq_len if seq_len is not None else self.seq_len or 0)
        if resolved_seq_len <= 0:
            raise ValueError("AimFeatureExtractor requires seq_len for schema generation.")
        return FeatureSchema(
            model_key="aim",
            version=aim_schema_version_for_mode(self.feature_mode),
            seq_len=resolved_seq_len,
            feature_names=aim_feature_names_for_mode(self.feature_mode),
            default_value=0.0,
            normalization=dict(NORMALIZATION_CONSTANTS),
        )

    def extract(self, sequence, belief_state: BeliefStateData | None = None, vision_target: 'VisionTarget' | None = None) -> np.ndarray:
        last_seen_enemy = None
        frames = []
        states = list(sequence.states)
        for state_idx, state in enumerate(states):
            visible_enemy = next((enemy for enemy in state.enemies if enemy.visibility == VisibilityStatus.VISIBLE.value and enemy.is_alive), None)
            if visible_enemy is not None:
                last_seen_enemy = visible_enemy
            frame = self._state_to_vector(state, visible_enemy, last_seen_enemy)
            if self.feature_mode == AIM_FEATURE_MODE_VISION_LIKE:
                frame.extend(self._vision_features_for_state(state_idx, len(states), vision_target))
            frames.append(frame)
        if self.seq_len is not None:
            frames = pad_or_trim_sequence(frames, self.seq_len, self.feature_dim(), default_value=0.0)
        return np.asarray(frames, dtype=np.float32)

    def extract_live(self, sequence, vision_target: 'VisionTarget' | None = None) -> np.ndarray:
        return self.extract(sequence, vision_target=vision_target)

    def feature_dim(self) -> int:
        return len(aim_feature_names_for_mode(self.feature_mode))

    def _vision_features_for_state(self, state_idx: int, state_count: int, vision_target: 'VisionTarget' | None) -> list[float]:
        if state_idx != state_count - 1:
            return [0.0] * len(AIM_VISION_FEATURE_NAMES)
        return vision_target_to_feature_values(vision_target)

    def _state_to_vector(self, state: GameState, visible_enemy, last_seen_enemy) -> list[float]:
        self_player = state.self_player
        enemy_screen = [0.0, 0.0]
        visible_mask = 0.0
        last_seen_mask = 0.0
        unavailable_mask = 1.0
        has_observed_target = 0.0

        if visible_enemy is not None and self_player.position is not None and visible_enemy.position is not None:
            enemy_screen_dx, enemy_screen_dy = world_to_screen_delta(
                self_player.position,
                self_player.yaw,
                self_player.pitch,
                visible_enemy.position,
            )
            enemy_screen = [enemy_screen_dx, enemy_screen_dy]
            visible_mask = 1.0
            unavailable_mask = 0.0
            has_observed_target = 1.0
        elif last_seen_enemy is not None and self_player.position is not None and last_seen_enemy.position is not None:
            enemy_screen_dx, enemy_screen_dy = world_to_screen_delta(
                self_player.position,
                self_player.yaw,
                self_player.pitch,
                last_seen_enemy.position,
            )
            enemy_screen = [enemy_screen_dx, enemy_screen_dy]
            last_seen_mask = 1.0
            unavailable_mask = 0.0
            has_observed_target = 1.0

        return [
            normalize_angle(self_player.yaw),
            normalize_angle(self_player.pitch),
            *map(normalize_velocity, self_player.velocity if self_player.velocity else [0.0, 0.0, 0.0]),
            weapon_to_id_normalized(self_player.weapon),
            normalize_hp(self_player.ammo),
            normalize_hp(self_player.shots_fired),
            bool_to_float(self_player.is_scoped),
            bool_to_float(self_player.is_walking),
            bool_to_float(self_player.is_airborne),
            normalize_hp(self_player.flash_duration),
            *enemy_screen,
            visible_mask,
            last_seen_mask,
            unavailable_mask,
            has_observed_target,
        ]


def build_aim_target(current_state: GameState, next_state: GameState) -> np.ndarray:
    target = [
        normalize_angle(next_state.self_player.yaw - current_state.self_player.yaw),
        normalize_angle(next_state.self_player.pitch - current_state.self_player.pitch),
        bool_to_float(next_state.self_input.fire),
        bool_to_float(next_state.self_input.rightclick),
        bool_to_float(next_state.self_input.zoom),
        normalize_mouse_delta(next_state.self_input.usercmd_mouse_dx),
        normalize_mouse_delta(next_state.self_input.usercmd_mouse_dy),
    ]
    return np.asarray(target, dtype=np.float32)


def build_aim_structured_target(current_state: GameState, next_state: GameState) -> AimStructuredTarget:
    flat_target = build_aim_target(current_state, next_state)
    aim_delta = np.asarray(
        [
            float(flat_target[0]),
            float(flat_target[1]),
            float(flat_target[5]),
            float(flat_target[6]),
        ],
        dtype=np.float32,
    )
    binary_actions = np.asarray(
        [
            float(flat_target[2]),
            float(flat_target[3]),
            float(flat_target[4]),
        ],
        dtype=np.float32,
    )
    valid_aim_mask = np.asarray(
        [1.0 if any(enemy.spotted and enemy.is_alive for enemy in current_state.enemies) else 0.0],
        dtype=np.float32,
    )
    return AimStructuredTarget(
        aim_delta=aim_delta,
        binary_actions=binary_actions,
        valid_aim_mask=valid_aim_mask,
    )
