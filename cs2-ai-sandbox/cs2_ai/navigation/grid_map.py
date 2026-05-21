from __future__ import annotations

from dataclasses import dataclass
import math

from cs2_ai.navigation.grid_config import GridConfig


@dataclass(frozen=True, slots=True)
class GridCell:
    cell_id: int
    ix: int
    iy: int
    iz: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float
    center_x: float
    center_y: float
    center_z: float


class GridMap:
    def __init__(self, config: GridConfig) -> None:
        self.config = config

    def is_inside_bounds(self, x: float, y: float, z: float) -> bool:
        return (
            self.config.min_x <= float(x) <= self.config.max_x
            and self.config.min_y <= float(y) <= self.config.max_y
            and self.config.min_z <= float(z) <= self.config.max_z
        )

    def clamp_indices(self, ix: int, iy: int, iz: int) -> tuple[int, int, int]:
        return (
            min(max(int(ix), 0), self.config.grid_size_x - 1),
            min(max(int(iy), 0), self.config.grid_size_y - 1),
            min(max(int(iz), 0), self.config.grid_size_z - 1),
        )

    def position_to_indices(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        ix = math.floor((float(x) - self.config.min_x) / self.config.cell_size_xy)
        iy = math.floor((float(y) - self.config.min_y) / self.config.cell_size_xy)
        iz = math.floor((float(z) - self.config.min_z) / self.config.cell_size_z)
        return self.clamp_indices(ix, iy, iz)

    def indices_to_cell_id(self, ix: int, iy: int, iz: int) -> int:
        ix, iy, iz = self.clamp_indices(ix, iy, iz)
        return int(ix + self.config.grid_size_x * (iy + self.config.grid_size_y * iz))

    def cell_id_to_indices(self, cell_id: int) -> tuple[int, int, int]:
        value = min(max(int(cell_id), 0), self.config.grid_size_x * self.config.grid_size_y * self.config.grid_size_z - 1)
        plane = self.config.grid_size_x * self.config.grid_size_y
        iz = value // plane
        rem = value % plane
        iy = rem // self.config.grid_size_x
        ix = rem % self.config.grid_size_x
        return self.clamp_indices(ix, iy, iz)

    def cell_center_from_indices(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
        ix, iy, iz = self.clamp_indices(ix, iy, iz)
        center_x = self.config.min_x + (ix + 0.5) * self.config.cell_size_xy
        center_y = self.config.min_y + (iy + 0.5) * self.config.cell_size_xy
        center_z = self.config.min_z + (iz + 0.5) * self.config.cell_size_z
        return float(center_x), float(center_y), float(center_z)

    def cell_center(self, cell_id: int) -> tuple[float, float, float]:
        return self.cell_center_from_indices(*self.cell_id_to_indices(cell_id))

    def position_to_cell_id(self, x: float, y: float, z: float) -> int:
        return self.indices_to_cell_id(*self.position_to_indices(x, y, z))

    def cell_from_position(self, x: float, y: float, z: float) -> GridCell:
        ix, iy, iz = self.position_to_indices(x, y, z)
        cell_id = self.indices_to_cell_id(ix, iy, iz)
        min_x = self.config.min_x + ix * self.config.cell_size_xy
        max_x = min_x + self.config.cell_size_xy
        min_y = self.config.min_y + iy * self.config.cell_size_xy
        max_y = min_y + self.config.cell_size_xy
        min_z = self.config.min_z + iz * self.config.cell_size_z
        max_z = min_z + self.config.cell_size_z
        center_x, center_y, center_z = self.cell_center_from_indices(ix, iy, iz)
        return GridCell(
            cell_id=cell_id,
            ix=ix,
            iy=iy,
            iz=iz,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_z=min_z,
            max_z=max_z,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
