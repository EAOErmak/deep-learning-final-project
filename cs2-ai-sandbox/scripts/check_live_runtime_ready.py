from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

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


def print_next_steps(state: GameState) -> None:
    caps = state.capabilities
    if caps.has_spatial_state and caps.has_allplayers:
        print('next_steps: observer-grade GSI is available. You can use --min-live-readiness observer.')
        return
    if caps.has_spatial_state:
        print('next_steps: spatial GSI is available, but allplayers is missing. Use --min-live-readiness spatial for self-only live models.')
        print('next_steps: for enemy-aware runtime, join as observer/GOTV/spectator and check allplayers again.')
        return

    print('next_steps: spatial fields are missing. Python already requests player_position/allplayers_position in config.')
    print('next_steps: make sure the cfg is installed in the CS2 CLIENT cfg directory, not only the dedicated server cfg directory.')
    print('next_steps: fully restart the CS2 client after changing the cfg, then join the local server again.')
    print('next_steps: if regular player mode still omits position/forward, test observer/spectator/GOTV mode.')
    print('next_steps: trained neural checkpoints should be run with --min-live-readiness spatial or observer.')

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
    print_next_steps(state)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

