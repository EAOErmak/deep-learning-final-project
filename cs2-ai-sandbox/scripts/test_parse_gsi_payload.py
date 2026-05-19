from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_encoder import encode_state
from game_state import GameState, PlayerState, Vector3
from gsi_server import GSIServer
from gsi_state_reader import GSIStateReader


def main() -> int:
    server = GSIServer()
    payload = {
        'provider': {'name': 'Counter-Strike 2', 'timestamp': 123.0},
        'map': {
            'name': 'de_dust2',
            'mode': 'competitive',
            'phase': 'live',
            'round': 3,
            'team_ct': {'score': 2, 'consecutive_round_losses': 1},
            'team_t': {'score': 0, 'consecutive_round_losses': 2},
        },
        'round': {'phase': 'live', 'bomb': 'planted'},
        'phase_countdowns': {'phase_ends_in': 22.5},
        'player': {
            'steamid': '111',
            'name': 'observer',
            'team': 'CT',
            'position': '10, 20, 30',
            'forward': '1, 0, 0',
            'state': {'health': 100, 'armor': 50, 'money': 4300, 'helmet': True},
            'match_stats': {'money': 4200},
            'weapons': {'weapon_0': {'name': 'M4A1-S', 'state': 'active', 'ammo_clip': 25, 'ammo_reserve': 60}},
        },
    }
    server._store.set_payload(payload)
    state = GSIStateReader(server).read_state()
    assert state is not None
    assert state.controlled_player is not None
    assert state.controlled_player.position is not None
    assert state.controlled_player.position.x == 10.0
    assert state.controlled_player.position.y == 20.0
    assert state.controlled_player.position.z == 30.0
    assert state.controlled_player.money == 4300
    assert state.controlled_player.ammo_reserve == 60
    assert state.map_state.name == 'de_dust2'
    assert state.round_state.phase == 'live'
    assert state.round_state.bomb_state == 'planted'
    assert state.capabilities.has_player_position is True
    assert state.capabilities.has_spatial_state is True
    assert state.players
    empty_enemy_state = GameState(
        provider='test',
        timestamp=0.0,
        controlled_player=PlayerState('1', 'p1', 'CT', Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), 100, 50, 1000, 'M4A1-S', 25, True),
        players=[PlayerState('1', 'p1', 'CT', Vector3(0.0, 0.0, 0.0), Vector3(1.0, 0.0, 0.0), 100, 50, 1000, 'M4A1-S', 25, True)],
        raw={},
    )
    features = encode_state(empty_enemy_state)
    assert features['enemy_visible'] == 0
    assert features['enemy_rel_x'] == 0
    assert features['enemy_distance'] == 0
    assert features['has_spatial_state'] == 0
    print('test_parse_gsi_payload.py OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
