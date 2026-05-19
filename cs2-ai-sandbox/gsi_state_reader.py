from __future__ import annotations

import re
from typing import Any

from game_state import GameState, LiveCapabilities, MapState, PlayerState, RoundState, Vector3
from gsi_server import GSIServer


class GSIStateReader:
    def __init__(self, gsi_server: GSIServer):
        self.gsi_server = gsi_server
        self._last_positions: dict[str, tuple[float, Vector3]] = {}

    def read_state(self) -> GameState | None:
        payload = self.gsi_server.get_latest_payload()
        if payload is None:
            return None
        return self._parse_payload(payload)

    def _parse_payload(self, payload: dict[str, Any]) -> GameState:
        provider_block = payload.get('provider') if isinstance(payload.get('provider'), dict) else {}
        provider = str(provider_block.get('name', 'cs2-gsi'))
        timestamp = self._safe_float(provider_block.get('timestamp', 0.0))
        map_state = self._parse_map_state(payload)
        round_state = self._parse_round_state(payload)
        controlled_player = self._parse_controlled_player(payload, timestamp)
        players = self._parse_allplayers(payload, timestamp)
        if controlled_player is not None and not any(player.id == controlled_player.id for player in players):
            players.insert(0, controlled_player)
        capabilities = self._build_capabilities(controlled_player, players, map_state, round_state, payload)
        return GameState(
            provider=provider,
            timestamp=timestamp,
            controlled_player=controlled_player,
            players=players,
            raw=payload,
            map_state=map_state,
            round_state=round_state,
            capabilities=capabilities,
        )

    def _parse_map_state(self, payload: dict[str, Any]) -> MapState:
        map_block = payload.get('map') if isinstance(payload.get('map'), dict) else {}
        team_ct = map_block.get('team_ct') if isinstance(map_block.get('team_ct'), dict) else {}
        team_t = map_block.get('team_t') if isinstance(map_block.get('team_t'), dict) else {}
        return MapState(
            name=self._safe_str(map_block.get('name')),
            mode=self._safe_str(map_block.get('mode')),
            phase=self._safe_str(map_block.get('phase')),
            round_number=self._safe_int(map_block.get('round')),
            ct_score=self._safe_int(team_ct.get('score')),
            t_score=self._safe_int(team_t.get('score')),
            ct_consecutive_round_losses=self._safe_int(team_ct.get('consecutive_round_losses')),
            t_consecutive_round_losses=self._safe_int(team_t.get('consecutive_round_losses')),
        )

    def _parse_round_state(self, payload: dict[str, Any]) -> RoundState:
        round_block = payload.get('round') if isinstance(payload.get('round'), dict) else {}
        phase_countdowns = payload.get('phase_countdowns') if isinstance(payload.get('phase_countdowns'), dict) else {}
        return RoundState(
            phase=self._safe_str(round_block.get('phase')),
            bomb_state=self._safe_str(round_block.get('bomb')),
            phase_ends_in=self._safe_float_or_none(phase_countdowns.get('phase_ends_in')),
        )

    def _parse_controlled_player(self, payload: dict[str, Any], timestamp: float) -> PlayerState | None:
        player_block = payload.get('player')
        if not isinstance(player_block, dict):
            return None
        player_id_block = payload.get('player_id') if isinstance(payload.get('player_id'), dict) else {}
        player_id = player_block.get('steamid') or player_id_block.get('steamid') or player_block.get('accountid') or 'controlled'
        return self._parse_player_block(str(player_id), player_block, timestamp)

    def _parse_allplayers(self, payload: dict[str, Any], timestamp: float) -> list[PlayerState]:
        allplayers = payload.get('allplayers')
        if not isinstance(allplayers, dict):
            return []
        players: list[PlayerState] = []
        for player_id, player_block in allplayers.items():
            if not isinstance(player_block, dict):
                continue
            players.append(self._parse_player_block(str(player_block.get('steamid') or player_id), player_block, timestamp))
        return players

    def _parse_player_block(self, player_id: str, player_block: dict[str, Any], timestamp: float) -> PlayerState:
        state_block = player_block.get('state', {}) if isinstance(player_block.get('state'), dict) else {}
        match_stats = player_block.get('match_stats', {}) if isinstance(player_block.get('match_stats'), dict) else {}
        weapons = player_block.get('weapons', {}) if isinstance(player_block.get('weapons'), dict) else {}
        active_weapon_name, active_ammo, active_ammo_reserve = self._extract_weapon_data(weapons)
        money_value = state_block.get('money') if state_block.get('money') is not None else match_stats.get('money')
        position = self._parse_vector(player_block.get('position'))
        forward = self._parse_vector(player_block.get('forward'))
        velocity = self._estimate_velocity(player_id, position, timestamp)
        return PlayerState(
            id=player_id,
            name=self._safe_str(player_block.get('name')),
            team=self._safe_str(player_block.get('team')),
            position=position,
            forward=forward,
            health=self._safe_int(state_block.get('health')),
            armor=self._safe_int(state_block.get('armor')),
            money=self._safe_int(money_value),
            weapon=active_weapon_name,
            ammo=active_ammo,
            is_alive=self._parse_is_alive(state_block.get('health')),
            velocity=velocity,
            helmet=self._safe_bool_or_none(state_block.get('helmet')),
            flashed=self._safe_int(state_block.get('flashed')),
            smoked=self._safe_int(state_block.get('smoked')),
            burning=self._safe_int(state_block.get('burning')),
            round_kills=self._safe_int(state_block.get('round_kills')),
            round_killhs=self._safe_int(state_block.get('round_killhs')),
            equip_value=self._safe_int(state_block.get('equip_value')),
            ammo_reserve=active_ammo_reserve,
            observer_slot=self._safe_int(player_block.get('observer_slot')),
            activity=self._safe_str(player_block.get('activity')),
        )

    def _build_capabilities(
        self,
        controlled_player: PlayerState | None,
        players: list[PlayerState],
        map_state: MapState,
        round_state: RoundState,
        payload: dict[str, Any],
    ) -> LiveCapabilities:
        has_allplayers = isinstance(payload.get('allplayers'), dict) and len(payload.get('allplayers')) > 0
        has_enemy_players = False
        if controlled_player is not None:
            has_enemy_players = any(player.id != controlled_player.id and player.team != controlled_player.team for player in players)
        return LiveCapabilities(
            has_player_position=controlled_player is not None and controlled_player.position is not None,
            has_player_forward=controlled_player is not None and controlled_player.forward is not None,
            has_allplayers=has_allplayers,
            has_enemy_players=has_enemy_players,
            has_spatial_state=controlled_player is not None and controlled_player.position is not None and controlled_player.forward is not None,
            has_round_state=bool(map_state.phase or round_state.phase),
            has_bomb_state=round_state.bomb_state is not None,
        )

    def _extract_weapon_data(self, weapons: dict[str, Any]) -> tuple[str | None, int | None, int | None]:
        active_weapon_name: str | None = None
        active_ammo: int | None = None
        active_ammo_reserve: int | None = None
        for weapon_info in weapons.values():
            if not isinstance(weapon_info, dict):
                continue
            if weapon_info.get('state') == 'active':
                active_weapon_name = self._safe_str(weapon_info.get('name'))
                active_ammo = self._safe_int(weapon_info.get('ammo_clip'))
                active_ammo_reserve = self._safe_int(weapon_info.get('ammo_reserve'))
                break
        if active_weapon_name is None and weapons:
            first_weapon = next(iter(weapons.values()))
            if isinstance(first_weapon, dict):
                active_weapon_name = self._safe_str(first_weapon.get('name'))
                active_ammo = self._safe_int(first_weapon.get('ammo_clip'))
                active_ammo_reserve = self._safe_int(first_weapon.get('ammo_reserve'))
        return active_weapon_name, active_ammo, active_ammo_reserve

    def _parse_vector(self, value: Any) -> Vector3 | None:
        if value is None:
            return None
        if isinstance(value, str):
            numbers = re.findall(r'-?\d+(?:\.\d+)?', value)
            if len(numbers) < 3:
                return None
            try:
                return Vector3(float(numbers[0]), float(numbers[1]), float(numbers[2]))
            except ValueError:
                return None
        if isinstance(value, dict):
            return Vector3(self._safe_float(value.get('x')), self._safe_float(value.get('y')), self._safe_float(value.get('z')))
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return Vector3(self._safe_float(value[0]), self._safe_float(value[1]), self._safe_float(value[2]))
        return None

    def _estimate_velocity(self, player_id: str, position: Vector3 | None, timestamp: float) -> Vector3 | None:
        if position is None or timestamp <= 0.0:
            return None
        last_entry = self._last_positions.get(player_id)
        self._last_positions[player_id] = (timestamp, position)
        if last_entry is None:
            return None
        last_timestamp, last_position = last_entry
        delta_t = timestamp - last_timestamp
        if delta_t <= 0.0:
            return None
        return Vector3(
            x=(position.x - last_position.x) / delta_t,
            y=(position.y - last_position.y) / delta_t,
            z=(position.z - last_position.z) / delta_t,
        )

    def _safe_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _safe_float_or_none(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _safe_bool_or_none(self, value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        lowered = str(value).strip().lower()
        if lowered in {'1', 'true', 'yes'}:
            return True
        if lowered in {'0', 'false', 'no'}:
            return False
        return None

    def _parse_is_alive(self, health: Any) -> bool | None:
        hp = self._safe_int(health)
        if hp is None:
            return None
        return hp > 0
