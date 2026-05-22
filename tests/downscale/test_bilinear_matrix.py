"""_build_bilinear_matrix_1d: partition-of-unity rows."""

import numpy as np
import scipy.sparse as sp

from geohalo.downscale import _build_bilinear_matrix_1d


def test_rows_sum_to_one() -> None:
    m = _build_bilinear_matrix_1d(5, 3)
    row_sums = np.asarray(m.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, 1.0, rtol=1e-12)


def test_factor_1_is_identity() -> None:
    m = _build_bilinear_matrix_1d(4, 1)
    np.testing.assert_array_equal(m.toarray(), np.eye(4))


def test_shape() -> None:
    m = _build_bilinear_matrix_1d(5, 3)
    assert isinstance(m, sp.csr_matrix)
    assert m.shape == (15, 5)
