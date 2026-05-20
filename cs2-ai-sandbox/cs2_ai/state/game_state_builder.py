from __future__ import annotations

from typing import Any

import pandas as pd

from cs2_ai.config import weapon_to_id
from cs2_ai.schemas.game_state import BombState, DemoTruthState, GameState, PlayerInputState, PlayerState, RoundState, StateBundle, VisibilityStatus


class GameStateBuilder:
    def build_from_tick_rows(self, tick_rows: pd.DataFrame, perspective_steamid: int) -> GameState:
        return self.build_state_bundle_from_tick_rows(tick_rows, perspective_steamid).observed_state

    def build_truth_from_tick_rows(self, tick_rows: pd.DataFrame, perspective_steamid: int) -> DemoTruthState:
        return self.build_state_bundle_from_tick_rows(tick_rows, perspective_steamid).truth_state

    def build_state_bundle_from_tick_rows(self, tick_rows: pd.DataFrame, perspective_steamid: int) -> StateBundle:
        if tick_rows.empty:
            raise ValueError("tick_rows is empty")
        steamids = pd.to_numeric(tick_rows["steamid"], errors="coerce") if "steamid" in tick_rows.columns else pd.Series(dtype="float64")
        self_rows = tick_rows.loc[steamids == int(perspective_steamid)]
        if self_rows.empty:
            raise ValueError(f"Perspective player {perspective_steamid} not found on tick")
        self_row = self_rows.iloc[0]
        self_team = self._as_int(self._safe_get(self_row, "team_num", 0))
        teammates: list[PlayerState] = []
        observed_enemies: list[PlayerState] = []
        truth_enemies: list[PlayerState] = []
        for _, row in tick_rows.iterrows():
            is_visible = self._as_bool(self._safe_get(row, "spotted", False))
            player = self._build_player_state(row, visibility=VisibilityStatus.VISIBLE if is_visible else VisibilityStatus.HIDDEN_TRUTH_ONLY)
            if player.steamid == int(perspective_steamid):
                continue
            if player.team_num == self_team:
                teammates.append(player)
            else:
                truth_enemies.append(player)
                if is_visible:
                    observed_enemies.append(self._build_player_state(row, visibility=VisibilityStatus.VISIBLE))
        tick_value = self._as_int(self._safe_get(self_row, "tick", self._safe_get(tick_rows.iloc[0], "tick", 0)))
        round_row = self_row
        round_state = RoundState(
            tick=tick_value,
            round_number=self._as_int(self._safe_get(round_row, "total_rounds_played", 0)),
            round_start_time=self._as_float(self._safe_get(round_row, "round_start_time", 0.0)),
            round_in_progress=self._as_bool(self._safe_get(round_row, "round_in_progress", False)),
            is_freeze_period=self._as_bool(self._safe_get(round_row, "is_freeze_period", False)),
            is_warmup_period=self._as_bool(self._safe_get(round_row, "is_warmup_period", False)),
            game_phase=self._as_int(self._safe_get(round_row, "game_phase", 0)),
            round_win_status=self._as_int(self._safe_get(round_row, "round_win_status", 0)),
            round_win_reason=self._as_int(self._safe_get(round_row, "round_win_reason", 0)),
            ct_losing_streak=self._as_int(self._safe_get(round_row, "ct_losing_streak", 0)),
            t_losing_streak=self._as_int(self._safe_get(round_row, "t_losing_streak", 0)),
        )
        bomb_state = BombState(
            is_bomb_planted=self._as_bool(self._safe_get(round_row, "is_bomb_planted", False)),
            is_bomb_dropped=self._as_bool(self._safe_get(round_row, "is_bomb_dropped", False)),
            bomb_position=None,
        )
        observed_state = GameState(
            tick=tick_value,
            perspective_steamid=int(perspective_steamid),
            self_player=self._build_player_state(self_row, visibility=VisibilityStatus.VISIBLE),
            self_input=self._build_input_state(self_row),
            teammates=teammates,
            enemies=observed_enemies,
            round=round_state,
            bomb=bomb_state,
        )
        truth_state = DemoTruthState(
            tick=tick_value,
            perspective_steamid=int(perspective_steamid),
            self_player=self._build_player_state(self_row, visibility=VisibilityStatus.VISIBLE),
            teammates=teammates,
            enemies=truth_enemies,
            round=round_state,
            bomb=bomb_state,
        )
        return StateBundle(observed_state=observed_state, truth_state=truth_state)

    def _build_player_state(self, row: pd.Series, visibility: VisibilityStatus = VisibilityStatus.VISIBLE) -> PlayerState:
        weapon = str(self._safe_get(row, "active_weapon_name", "none") or "none")
        return PlayerState(
            steamid=self._as_int(self._safe_get(row, "steamid", 0)),
            name=str(self._safe_get(row, "name", "unknown")),
            team_num=self._as_int(self._safe_get(row, "team_num", 0)),
            position=[self._as_float(self._safe_get(row, axis, 0.0)) for axis in ("X", "Y", "Z")],
            velocity=[self._as_float(self._safe_get(row, axis, 0.0)) for axis in ("velocity_X", "velocity_Y", "velocity_Z")],
            health=self._as_float(self._safe_get(row, "health", 0.0)),
            armor=self._as_float(self._safe_get(row, "armor_value", 0.0)),
            has_helmet=self._as_bool(self._safe_get(row, "has_helmet", False)),
            is_alive=self._as_bool(self._safe_get(row, "is_alive", False)),
            money=self._as_float(self._safe_get(row, "balance", 0.0)),
            weapon=weapon,
            weapon_id=weapon_to_id(weapon),
            ammo=self._as_float(self._safe_get(row, "active_weapon_ammo", 0.0)),
            total_ammo=self._as_float(self._safe_get(row, "total_ammo_left", 0.0)),
            pitch=self._as_float(self._safe_get(row, "pitch", 0.0)),
            yaw=self._as_float(self._safe_get(row, "yaw", 0.0)),
            is_scoped=self._as_bool(self._safe_get(row, "is_scoped", False)),
            is_walking=self._as_bool(self._safe_get(row, "is_walking", False)),
            is_airborne=self._as_bool(self._safe_get(row, "is_airborne", False)),
            duck_amount=self._as_float(self._safe_get(row, "duck_amount", 0.0)),
            ducking=self._as_bool(self._safe_get(row, "ducking", False)),
            shots_fired=self._as_int(self._safe_get(row, "shots_fired", 0)),
            flash_duration=self._as_float(self._safe_get(row, "flash_duration", 0.0)),
            spotted=(str(visibility) == VisibilityStatus.VISIBLE.value),
            last_place_name=str(self._safe_get(row, "last_place_name", "unknown")),
            in_bomb_zone=self._as_bool(self._safe_get(row, "in_bomb_zone", False)),
            in_buy_zone=self._as_bool(self._safe_get(row, "in_buy_zone", False)),
            which_bomb_zone=self._as_int(self._safe_get(row, "which_bomb_zone", 0)),
            visibility=str(visibility),
        )

    def _build_input_state(self, row: pd.Series) -> PlayerInputState:
        walk_value = self._safe_get(row, "WALK", None)
        if walk_value is None:
            walk_value = self._safe_get(row, "is_walking", False)
        return PlayerInputState(
            forward=self._as_bool(self._safe_get(row, "FORWARD", False)),
            back=self._as_bool(self._safe_get(row, "BACK", False)),
            left=self._as_bool(self._safe_get(row, "LEFT", False)),
            right=self._as_bool(self._safe_get(row, "RIGHT", False)),
            fire=self._as_bool(self._safe_get(row, "FIRE", False)),
            rightclick=self._as_bool(self._safe_get(row, "RIGHTCLICK", False)),
            reload=self._as_bool(self._safe_get(row, "RELOAD", False)),
            use=self._as_bool(self._safe_get(row, "USE", False)),
            zoom=self._as_bool(self._safe_get(row, "ZOOM", False)),
            walk=self._as_bool(walk_value),
            usercmd_mouse_dx=self._as_float(self._safe_get(row, "usercmd_mouse_dx", 0.0)),
            usercmd_mouse_dy=self._as_float(self._safe_get(row, "usercmd_mouse_dy", 0.0)),
            usercmd_forward_move=self._as_float(self._safe_get(row, "usercmd_forward_move", 0.0)),
            usercmd_left_move=self._as_float(self._safe_get(row, "usercmd_left_move", 0.0)),
        )

    def _safe_get(self, row: pd.Series, column: str, default: Any) -> Any:
        if column not in row.index:
            return default
        value = row[column]
        if pd.isna(value):
            return default
        return value

    def _as_bool(self, value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes"}
        return bool(value)

    def _as_float(self, value: Any) -> float:
        if pd.isna(value):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _as_int(self, value: Any) -> int:
        if pd.isna(value):
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return 0
