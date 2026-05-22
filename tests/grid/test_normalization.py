"""GridSpec lat normalization and shape."""

import numpy as np
import pytest

from geohalo.grid import GridSpec


def test_ascending_lats_kept_as_is() -> None:
    grid = GridSpec(lats=np.array([0.0, 1.0, 2.0]), lons=np.array([0.0, 1.0]))
    assert grid.lats_were_descending is False
    np.testing.assert_array_equal(grid.lats, [0.0, 1.0, 2.0])


def test_descending_lats_flipped_to_ascending() -> None:
    grid = GridSpec(lats=np.array([2.0, 1.0, 0.0]), lons=np.array([0.0, 1.0]))
    assert grid.lats_were_descending is True
    np.testing.assert_array_equal(grid.lats, [0.0, 1.0, 2.0])


def test_shape_records_lat_lon_sizes() -> None:
    grid = GridSpec(lats=np.arange(5.0), lons=np.arange(6.0))
    assert grid.shape == (5, 6)


def test_non_1d_lats_raises() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        GridSpec(lats=np.zeros((2, 2)), lons=np.arange(2.0))


def test_non_1d_lons_raises() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        GridSpec(lats=np.arange(2.0), lons=np.zeros((2, 2)))
