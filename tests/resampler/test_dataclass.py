import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from geohalo.resampler import Resampler


def test_resampler_is_frozen() -> None:
    r = Resampler(
        transform_matrix=sp.csr_matrix((4, 2), dtype=np.float64),
        source_lat=np.array([0.0]),
        source_lon=np.array([0.0, 1.0]),
        target_lat=np.array([0.0, 0.5]),
        target_lon=np.array([0.0, 0.5]),
        digest=b"\x00" * 32,
    )
    assert dataclasses.is_dataclass(r)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.digest = b"\x01" * 32  # type: ignore[misc]
