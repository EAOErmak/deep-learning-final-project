from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

from feature_encoder import encode_state
from gsi_state_reader import GSIStateReader


class _StaticServer:
    def __init__(self, payload: dict):
        self._payload = payload

    def get_latest_payload(self) -> dict:
        return self._payload


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    payload_path = project_root / 'latest_gsi_payload.json'
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

    print('Controlled player parsed')
    pprint(game_state.controlled_player)
    print()

    print('Players parsed')
    print(f'  total players: {len(game_state.players)}')
    for player in game_state.players[:10]:
        print(
            f'  id={player.id} name={player.name} team={player.team} '
            f'pos={player.position} hp={player.health} money={player.money} '
            f'weapon={player.weapon} ammo={player.ammo} alive={player.is_alive}'
        )
    print()

    print('Encoded features')
    pprint(encode_state(game_state))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
