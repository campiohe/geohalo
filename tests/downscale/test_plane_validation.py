"""downscale_plane input validation."""

import numpy as np
import pytest

from geohalo.downscale import downscale_plane


def test_factor_lt_1_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        downscale_plane(np.zeros((3, 3)), 0)


def test_iterations_lt_1_raises() -> None:
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        downscale_plane(np.zeros((3, 3)), 2, iterations=0)


def test_non_2d_raises() -> None:
    with pytest.raises(ValueError, match="2-D"):
        downscale_plane(np.zeros((3, 3, 3)), 2)


def test_factor_1_returns_input_unchanged() -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = downscale_plane(data, 1)
    np.testing.assert_array_equal(out, data)
