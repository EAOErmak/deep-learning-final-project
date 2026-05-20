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
from pprint import pprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_encoder import encode_state
from gsi_state_reader import GSIStateReader


class _StaticServer:
    def __init__(self, payload: dict):
        self._payload = payload

    def get_latest_payload(self) -> dict:
        return self._payload


def main() -> int:
    payload_path = PROJECT_ROOT / 'latest_gsi_payload.json'
    if not payload_path.exists():
        print(f'latest_gsi_payload.json not found: {payload_path}')
        print('Run `python main.py --state-source gsi --gsi-port 3000` once, then rerun this script.')
        return 1

    payload = json.loads(payload_path.read_text(encoding='utf-8'))
    reader = GSIStateReader(_StaticServer(payload))
    game_state = reader.read_state()
    if game_state is None:
        print('Failed to parse payload into GameState.')
        return 1

    player_block = payload.get('player') if isinstance(payload.get('player'), dict) else {}
    allplayers_block = payload.get('allplayers') if isinstance(payload.get('allplayers'), dict) else {}

    print('Payload summary')
    print(f'  file: {payload_path}')
    print(f'  top-level keys: {sorted(payload.keys())}')
    print(f'  player keys: {sorted(player_block.keys()) if isinstance(player_block, dict) else []}')
    print(f'  allplayers count: {len(allplayers_block)}')
    print()

    print('Map state')
    pprint(game_state.map_state)
    print('Round state')
    pprint(game_state.round_state)
    print('Capabilities')
    pprint(game_state.capabilities)
    print()

    print('Controlled player parsed')
    pprint(game_state.controlled_player)
    print()

    print('Players parsed')
    print(f'  total players: {len(game_state.players)}')
    for player in game_state.players[:10]:
        print(
            f'  id={player.id} name={player.name} team={player.team} '
            f'pos={player.position} vel={player.velocity} hp={player.health} money={player.money} '
            f'weapon={player.weapon} ammo={player.ammo}/{player.ammo_reserve} alive={player.is_alive}'
        )
    print()

    print('Encoded features')
    pprint(encode_state(game_state))
    print()

    if 'position' not in player_block:
        print('Warning: player.position missing in payload.')
    if 'forward' not in player_block:
        print('Warning: player.forward missing in payload.')
    if not isinstance(allplayers_block, dict) or not allplayers_block:
        print('Warning: allplayers missing in payload.')
        print('This usually means regular player GSI, not observer/GOTV-style full-state GSI.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
