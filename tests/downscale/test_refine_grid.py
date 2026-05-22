"""refine_grid: sub-divide a GridSpec by an integer factor."""

import numpy as np
import pytest

from geohalo.downscale import refine_grid
from geohalo.grid import GridSpec


def test_factor_1_returns_same_grid() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    out = refine_grid(grid, 1)
    assert out is grid


def test_factor_negative_raises() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match=">= 1"):
        refine_grid(grid, 0)


def test_factor_2_doubles_resolution() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([0.0, 1.0]))
    refined = refine_grid(grid, 2)
    assert refined.shape == (6, 4)
    new_dlat = abs(refined.lats[1] - refined.lats[0])
    np.testing.assert_allclose(new_dlat, 0.5)


def test_refined_digest_differs() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([0.0, 1.0]))
    refined = refine_grid(grid, 2)
    assert refined.digest != grid.digest
