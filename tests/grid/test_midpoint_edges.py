"""_midpoint_edges helper."""

import numpy as np
import pytest

from geohalo.grid import _midpoint_edges


def test_known_input() -> None:
    edges = _midpoint_edges(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(edges, [-0.5, 0.5, 1.5, 2.5])


def test_irregular_spacing() -> None:
    edges = _midpoint_edges(np.array([0.0, 1.0, 3.0]))
    np.testing.assert_allclose(edges, [-0.5, 0.5, 2.0, 4.0])


def test_size_lt_2_raises() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        _midpoint_edges(np.array([0.0]))
