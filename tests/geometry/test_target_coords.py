import numpy as np
import pytest

from geohalo.geometry import target_coords_from_resolution


def test_refine() -> None:
    src_lat = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    src_lon = np.array([0.0, 0.25, 0.5])
    tlat, _tlon = target_coords_from_resolution(src_lat, src_lon, 0.05)
    assert tlat[0] == 0.0
    np.testing.assert_allclose(tlat[1] - tlat[0], 0.05)
    assert tlat[-1] <= 1.0 + 1e-9


def test_coarsen() -> None:
    src = np.arange(0.0, 1.01, 0.1)
    tlat, _ = target_coords_from_resolution(src, src, 0.5)
    np.testing.assert_allclose(tlat[1] - tlat[0], 0.5)


def test_invalid_resolution() -> None:
    src = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="> 0"):
        target_coords_from_resolution(src, src, 0.0)
