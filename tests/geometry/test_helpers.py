import numpy as np
import pytest

from geohalo.geometry import ensure_ascending_lats, midpoint_edges, require_regular_grid, same_grid


def test_midpoint_edges_uniform() -> None:
    centres = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(midpoint_edges(centres), [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])


def test_midpoint_edges_non_uniform() -> None:
    centres = np.array([0.0, 1.0, 3.0, 6.0])
    np.testing.assert_allclose(midpoint_edges(centres), [-0.5, 0.5, 2.0, 4.5, 7.5])


def test_midpoint_edges_too_short_raises() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        midpoint_edges(np.array([1.0]))


def test_ensure_ascending_already_ascending() -> None:
    lats = np.array([-2.0, -1.0, 0.0])
    out, flipped = ensure_ascending_lats(lats)
    np.testing.assert_array_equal(out, lats)
    assert flipped is False


def test_ensure_ascending_descending_flips() -> None:
    out, flipped = ensure_ascending_lats(np.array([2.0, 1.0, 0.0]))
    np.testing.assert_array_equal(out, [0.0, 1.0, 2.0])
    assert flipped is True


def test_require_regular_grid_accepts_uniform() -> None:
    require_regular_grid(np.array([0.0, 0.25, 0.5, 0.75]), "longitude")  # no raise
    require_regular_grid(np.array([5.0]), "single")  # too short to judge -> no raise


def test_require_regular_grid_rejects_irregular() -> None:
    with pytest.raises(ValueError, match="regularly spaced"):
        require_regular_grid(np.array([0.0, 1.0, 3.0, 6.0]), "latitude")


def test_same_grid() -> None:
    lat, lon = np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])
    assert same_grid(lat, lon, lat.copy(), lon.copy())
    assert not same_grid(lat, lon, lat, np.array([0.0, 1.0]))
    assert not same_grid(lat, lon, np.array([0.0, 1.5]), lon)
