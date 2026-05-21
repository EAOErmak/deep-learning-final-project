from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True, slots=True)
class GridConfig:
    map_name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    cell_size_xy: float
    cell_size_z: float
    grid_size_x: int = field(init=False)
    grid_size_y: int = field(init=False)
    grid_size_z: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'grid_size_x', int(math.ceil((self.max_x - self.min_x) / self.cell_size_xy)))
        object.__setattr__(self, 'grid_size_y', int(math.ceil((self.max_y - self.min_y) / self.cell_size_xy)))
        object.__setattr__(self, 'grid_size_z', int(math.ceil((self.max_z - self.min_z) / self.cell_size_z)))


DUST2_GRID_CONFIG = GridConfig(
    map_name='de_dust2',
    min_x=-2300.0,
    max_x=1850.0,
    min_y=-1250.0,
    max_y=3200.0,
    min_z=-256.0,
    max_z=512.0,
    cell_size_xy=25.0,
    cell_size_z=16.0,
)
