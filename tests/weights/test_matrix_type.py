"""Weights.matrix is a scipy CSR."""

import scipy.sparse as sp

from geohalo.geometry import PolygonSet
from geohalo.grid import GridSpec
from geohalo.weights import compute_weights


def test_returns_csr(simple_grid: GridSpec, simple_polygons: PolygonSet) -> None:
    weights = compute_weights(simple_polygons, simple_grid)
    assert isinstance(weights.matrix, sp.csr_matrix)
