from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_state import GameState
from gsi_state_reader import GSIStateReader
from main import is_live_runtime_ready


class _StaticServer:
    def __init__(self, payload: dict):
        self._payload = payload

    def get_latest_payload(self) -> dict:
        return self._payload


def main() -> int:
    payload_path = PROJECT_ROOT / 'latest_gsi_payload.json'
    if not payload_path.exists():
        print(f'latest_gsi_payload.json not found: {payload_path}')
        return 1

    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    state = GSIStateReader(_StaticServer(payload)).read_state()
    if state is None:
        print('Failed to parse payload.')
        return 1

    assert isinstance(state, GameState)
    for mode in ('basic', 'spatial', 'observer'):
        ready, reason = is_live_runtime_ready(state, mode)
        print(f'{mode}: ready={ready} | reason={reason}')
    print(f'capabilities: {state.capabilities}')
    player = state.controlled_player
    if player is not None:
        print(f'player_activity: {player.activity!r}')
        print(f'player_position: {player.position}')
        print(f'player_forward: {player.forward}')
    print(f'players_count: {len(state.players)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
