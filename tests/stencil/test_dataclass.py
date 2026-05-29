import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from geohalo.stencil import EmptyOverlapError, Stencil


def test_stencil_is_frozen() -> None:
    s = Stencil(
        occupancy_matrix=sp.csr_matrix((2, 6), dtype=np.float64),
        keys=pd.Index(["a", "b"]),
        lats=np.array([0.0, 1.0]),
        lons=np.array([0.0, 1.0, 2.0]),
        digest=b"\x00" * 32,
    )
    assert dataclasses.is_dataclass(s)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.lats = np.array([2.0, 3.0])  # type: ignore[misc]


def test_empty_overlap_error_carries_key() -> None:
    assert "polygon_42" in str(EmptyOverlapError("polygon_42"))


def test_row_sums_computed() -> None:
    matrix = sp.csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]]))
    s = Stencil(
        occupancy_matrix=matrix, keys=pd.Index(["a", "b"]),
        lats=np.array([0.0]), lons=np.array([0.0, 1.0, 2.0]), digest=b"\x00" * 32,
    )
    np.testing.assert_allclose(s.row_sums, [3.0, 7.0])


def test_spherical_correction_default_true() -> None:
    s = Stencil(
        occupancy_matrix=sp.csr_matrix((1, 2), dtype=np.float64),
        keys=pd.Index(["a"]), lats=np.array([0.0]), lons=np.array([0.0, 1.0]),
        digest=b"\x00" * 32,
    )
    assert s.spherical_correction is True
