"""Spherical cell area against the closed-form integral."""

import numpy as np
from hypothesis import given, settings

from geohalo.grid import EARTH_RADIUS_M, GridSpec
from tests._strategies import regular_grid_st


def test_full_grid_area_matches_closed_form() -> None:
    """Sum of cell areas equals R^2 · (sin φ_top - sin φ_bot) · Δλ."""
    lats = np.array([-1.0, 0.0, 1.0])
    lons = np.array([0.0, 1.0, 2.0])
    grid = GridSpec(lats=lats, lons=lons)
    # Edges: lats -> [-1.5, -0.5, 0.5, 1.5], lons -> [-0.5, 0.5, 1.5, 2.5]
    total = grid.cell_area.sum()
    expected = (
        (EARTH_RADIUS_M ** 2)
        * (np.sin(np.deg2rad(1.5)) - np.sin(np.deg2rad(-1.5)))
        * np.deg2rad(3.0)
    )
    np.testing.assert_allclose(total, expected, rtol=1e-12)


def test_cell_areas_positive() -> None:
    grid = GridSpec(lats=np.arange(5.0), lons=np.arange(6.0))
    assert (grid.cell_area > 0).all()


def test_cell_area_decreases_polewards() -> None:
    """For evenly-spaced lats, cell area shrinks as |lat| grows."""
    lats = np.array([0.0, 30.0, 60.0])
    lons = np.array([0.0, 1.0])
    grid = GridSpec(lats=lats, lons=lons)
    col0 = grid.cell_area[:, 0]
    assert col0[0] > col0[1] > col0[2]


@given(grid=regular_grid_st())
@settings(max_examples=50, deadline=None)
def test_cell_areas_positive_and_finite(grid: GridSpec) -> None:
    assert np.isfinite(grid.cell_area).all()
    assert (grid.cell_area > 0).all()
