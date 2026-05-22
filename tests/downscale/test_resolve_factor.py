"""resolve_factor: target_resolution -> (factor, achieved)."""

import numpy as np
import pytest

from geohalo.downscale import resolve_factor
from geohalo.grid import GridSpec


def test_none_target_returns_factor_1() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([0.0, 1.0, 2.0]))
    factor, achieved = resolve_factor(grid, None)
    assert factor == 1
    assert achieved == 1.0


def test_negative_target_raises() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="> 0"):
        resolve_factor(grid, -0.5)


def test_zero_target_raises() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="> 0"):
        resolve_factor(grid, 0.0)


def test_non_square_grid_raises() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="non-square"):
        resolve_factor(grid, 0.5)


def test_target_larger_than_dlat_keeps_factor_1() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    factor, _ = resolve_factor(grid, 5.0)
    assert factor == 1


def test_typical_2x_factor() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([0.0, 1.0, 2.0]))
    factor, achieved = resolve_factor(grid, 0.5)
    assert factor == 2
    np.testing.assert_allclose(achieved, 0.5)


def test_typical_4x_factor() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([0.0, 1.0, 2.0]))
    factor, achieved = resolve_factor(grid, 0.25)
    assert factor == 4
    np.testing.assert_allclose(achieved, 0.25)
