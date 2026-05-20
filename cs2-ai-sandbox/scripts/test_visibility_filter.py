from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_pycache_dir = PROJECT_ROOT / '.cache' / 'pycache'
_pycache_dir.mkdir(parents=True, exist_ok=True)
if getattr(sys, 'pycache_prefix', None) is None:
    sys.pycache_prefix = str(_pycache_dir)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_state import GameState, PlayerState, Vector3
from visibility_filter import filter_visible_enemies


def _player(player_id: str, team: str, pos: tuple[float, float, float], alive: bool = True) -> PlayerState:
    return PlayerState(
        id=player_id,
        name=player_id,
        team=team,
        position=Vector3(*pos),
        forward=Vector3(1.0, 0.0, 0.0),
        health=100 if alive else 0,
        armor=0,
        money=0,
        weapon='none',
        ammo=0,
        is_alive=alive,
    )


def main() -> int:
    controlled = _player('self', 'CT', (0.0, 0.0, 0.0))
    controlled.forward = Vector3(1.0, 0.0, 0.0)
    ally = _player('ally', 'CT', (100.0, 0.0, 0.0))
    enemy_visible = _player('enemy1', 'T', (500.0, 0.0, 0.0))
    enemy_hidden = _player('enemy2', 'T', (0.0, 500.0, 0.0))
    state = GameState(provider='test', timestamp=0.0, controlled_player=controlled, players=[controlled, ally, enemy_visible, enemy_hidden], raw={})
    filtered = filter_visible_enemies(state, fov_degrees=90.0, max_distance=3000.0)
    ids = {player.id for player in filtered.players}
    assert 'enemy1' in ids
    assert 'enemy2' not in ids
    assert 'ally' in ids
    print('test_visibility_filter.py OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
