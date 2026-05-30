import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from geohalo.bias_tree import BiasTree


def test_frozen() -> None:
    t = BiasTree(rollup_matrix=sp.csr_matrix(np.eye(2)), keys=pd.Index(["a", "b"]), digest=b"\x00" * 32)
    assert dataclasses.is_dataclass(t)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.digest = b"\x01" * 32  # type: ignore[misc]


def test_leaf_keys() -> None:
    matrix = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]))
    t = BiasTree(rollup_matrix=matrix, keys=pd.Index(["l1", "l2", "p"]), digest=b"\x00" * 32)
    assert list(t.leaf_keys) == ["l1", "l2"]


def test_how_default() -> None:
    t = BiasTree(rollup_matrix=sp.csr_matrix(np.eye(2)), keys=pd.Index(["a", "b"]), digest=b"\x00" * 32)
    assert t.how == "mean"
