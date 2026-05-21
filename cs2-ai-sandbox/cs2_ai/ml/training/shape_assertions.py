from __future__ import annotations

from typing import Any


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"Object of type {type(value).__name__} has no shape attribute.")
    return tuple(int(dim) for dim in shape)


def assert_rank(value: Any, expected_rank: int, name: str) -> tuple[int, ...]:
    shape = _shape_of(value)
    if len(shape) != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got shape {shape}.")
    return shape


def assert_shape(value: Any, expected_shape: tuple[int | None, ...], name: str) -> tuple[int, ...]:
    shape = _shape_of(value)
    if len(shape) != len(expected_shape):
        raise ValueError(f"{name} must have shape {expected_shape}, got {shape}.")
    for axis, (actual_dim, expected_dim) in enumerate(zip(shape, expected_shape, strict=True)):
        if expected_dim is not None and actual_dim != expected_dim:
            raise ValueError(
                f"{name} axis {axis} must be {expected_dim}, got {actual_dim}. Full shape: {shape}."
            )
    return shape


def assert_temporal_features(value: Any, *, seq_len: int, feature_dim: int, name: str = "features") -> tuple[int, int, int]:
    shape = assert_shape(value, (None, seq_len, feature_dim), name)
    return shape[0], shape[1], shape[2]
