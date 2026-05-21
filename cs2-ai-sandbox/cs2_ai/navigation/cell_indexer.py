from __future__ import annotations

from cs2_ai.navigation.grid_config import DUST2_GRID_CONFIG, GridConfig
from cs2_ai.navigation.grid_map import GridMap


def build_grid_map(map_name: str = 'de_dust2') -> GridMap:
    normalized = str(map_name).strip().lower()
    if normalized != 'de_dust2':
        raise ValueError(f'Unsupported map for grid indexing: {map_name!r}')
    return GridMap(DUST2_GRID_CONFIG)


def position_to_cell_id(x: float, y: float, z: float, config: GridConfig | None = None) -> int:
    return GridMap(config or DUST2_GRID_CONFIG).position_to_cell_id(x, y, z)


def position_to_indices(x: float, y: float, z: float, config: GridConfig | None = None) -> tuple[int, int, int]:
    return GridMap(config or DUST2_GRID_CONFIG).position_to_indices(x, y, z)
