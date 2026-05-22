"""build_downscale_operator: linear-method sparse operator baked into W."""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from geohalo.downscale import build_downscale_operator, downscale_plane
from geohalo.grid import GridSpec
from tests._strategies import regular_grid_st


def test_factor_1_returns_identity() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    op = build_downscale_operator(grid, 1)
    n = grid.shape[0] * grid.shape[1]
    np.testing.assert_array_equal(op.toarray(), np.eye(n))


def test_negative_factor_raises() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match=">= 1"):
        build_downscale_operator(grid, 0)


@given(grid=regular_grid_st(), factor=st.integers(min_value=2, max_value=3))
@settings(max_examples=30, deadline=None)
def test_operator_matches_downscale_plane_iter_1(
    grid: GridSpec,
    factor: int,
) -> None:
    """M @ flat(field) == downscale_plane(field, factor, iterations=1).ravel()."""
    rng = np.random.default_rng(seed=42)
    field = rng.standard_normal(grid.shape)
    op = build_downscale_operator(grid, factor)
    out_op = (op @ field.ravel()).reshape(grid.shape[0] * factor, grid.shape[1] * factor)
    out_kernel = downscale_plane(field, factor, iterations=1)
    np.testing.assert_allclose(out_op, out_kernel, rtol=1e-10, atol=1e-10)
