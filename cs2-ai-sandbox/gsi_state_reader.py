from __future__ import annotations

import re
from typing import Any

from game_state import GameState, PlayerState, Vector3
from gsi_server import GSIServer


class GSIStateReader:
    def __init__(self, gsi_server: GSIServer):
        self.gsi_server = gsi_server

    def read_state(self) -> GameState | None:
        payload = self.gsi_server.get_latest_payload()
        if payload is None:
            return None
        return self._parse_payload(payload)

    def _parse_payload(self, payload: dict[str, Any]) -> GameState:
        provider = str(payload.get('provider', {}).get('name', 'cs2-gsi')) if isinstance(payload.get('provider'), dict) else 'cs2-gsi'
        timestamp = self._safe_float(payload.get('provider', {}).get('timestamp', 0.0)) if isinstance(payload.get('provider'), dict) else 0.0
        controlled_player = self._parse_controlled_player(payload)
        players = self._parse_allplayers(payload)
        if controlled_player is not None and not any(player.id == controlled_player.id for player in players):
            players.insert(0, controlled_player)
        return GameState(provider=provider, timestamp=timestamp, controlled_player=controlled_player, players=players, raw=payload)

    def _parse_controlled_player(self, payload: dict[str, Any]) -> PlayerState | None:
        player_block = payload.get('player')
        if not isinstance(player_block, dict):
            return None
        player_id = payload.get('player', {}).get('steamid') or payload.get('player_id', {}).get('steamid') or payload.get('player', {}).get('accountid') or 'controlled'
        name = payload.get('player', {}).get('name')
        team = payload.get('player', {}).get('team')
        state_block = player_block.get('state', {}) if isinstance(player_block.get('state'), dict) else {}
        match_stats = player_block.get('match_stats', {}) if isinstance(player_block.get('match_stats'), dict) else {}
        weapons = player_block.get('weapons', {}) if isinstance(player_block.get('weapons'), dict) else {}
        active_weapon_name, active_ammo = self._extract_weapon_data(weapons)
        money_value = state_block.get('money') if state_block.get('money') is not None else match_stats.get('money')
        return PlayerState(
            id=str(player_id),
            name=name,
            team=team,
            position=self._parse_vector(player_block.get('position')),
            forward=self._parse_vector(player_block.get('forward')),
            health=self._safe_int(state_block.get('health')),
            armor=self._safe_int(state_block.get('armor')),
            money=self._safe_int(money_value),
            weapon=active_weapon_name,
            ammo=active_ammo,
            is_alive=self._parse_is_alive(state_block.get('health')),
        )

    def _parse_allplayers(self, payload: dict[str, Any]) -> list[PlayerState]:
        allplayers = payload.get('allplayers')
        if not isinstance(allplayers, dict):
            return []
        players: list[PlayerState] = []
        for player_id, player_block in allplayers.items():
            if not isinstance(player_block, dict):
                continue
            state_block = player_block.get('state', {}) if isinstance(player_block.get('state'), dict) else {}
            match_stats = player_block.get('match_stats', {}) if isinstance(player_block.get('match_stats'), dict) else {}
            weapons = player_block.get('weapons', {}) if isinstance(player_block.get('weapons'), dict) else {}
            active_weapon_name, active_ammo = self._extract_weapon_data(weapons)
            money_value = state_block.get('money') if state_block.get('money') is not None else match_stats.get('money')
            players.append(
                PlayerState(
                    id=str(player_block.get('steamid') or player_id),
                    name=player_block.get('name'),
                    team=player_block.get('team'),
                    position=self._parse_vector(player_block.get('position')),
                    forward=self._parse_vector(player_block.get('forward')),
                    health=self._safe_int(state_block.get('health')),
                    armor=self._safe_int(state_block.get('armor')),
                    money=self._safe_int(money_value),
                    weapon=active_weapon_name,
                    ammo=active_ammo,
                    is_alive=self._parse_is_alive(state_block.get('health')),
                )
            )
        return players

    def _extract_weapon_data(self, weapons: dict[str, Any]) -> tuple[str | None, int | None]:
        active_weapon_name: str | None = None
        active_ammo: int | None = None
        for weapon_info in weapons.values():
            if not isinstance(weapon_info, dict):
                continue
            if weapon_info.get('state') == 'active':
                active_weapon_name = weapon_info.get('name')
                active_ammo = self._safe_int(weapon_info.get('ammo_clip'))
                break
        if active_weapon_name is None and weapons:
            first_weapon = next(iter(weapons.values()))
            if isinstance(first_weapon, dict):
                active_weapon_name = first_weapon.get('name')
                active_ammo = self._safe_int(first_weapon.get('ammo_clip'))
        return active_weapon_name, active_ammo

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

    def _parse_is_alive(self, health: Any) -> bool | None:
        hp = self._safe_int(health)
        if hp is None:
            return None
        return hp > 0
