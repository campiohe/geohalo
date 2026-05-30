import numpy as np
import pandas as pd
import pytest
import xarray as xr

from geohalo.api import aggregate_bias, aggregate_bias_with_tree
from geohalo.bias_tree import BiasTree


def _leaves(values: dict) -> xr.DataArray:
    return xr.DataArray(list(values.values()), dims=("geom",),
                        coords={"geom": list(values.keys())})


def _edges(rows: list[tuple]) -> pd.DataFrame:
    has_w = len(rows[0]) == 3
    data = {"parent": [r[1] for r in rows]}
    if has_w:
        data["weight"] = [r[2] for r in rows]
    return pd.DataFrame(data, index=pd.Index([r[0] for r in rows], name="child"))


def test_tree_mean() -> None:
    tree = BiasTree.compute(_edges([("a", "p"), ("b", "p")]), how="mean")
    out = aggregate_bias_with_tree(_leaves({"a": 2.0, "b": 4.0}), tree)
    np.testing.assert_allclose(out.values, [2.0, 4.0, 3.0])


def test_tree_sum() -> None:
    tree = BiasTree.compute(_edges([("a", "p", 1.0), ("b", "p", 3.0)]), weight_col="weight", how="sum")
    out = aggregate_bias_with_tree(_leaves({"a": 2.0, "b": 4.0}), tree)
    np.testing.assert_allclose(out.values, [2.0, 4.0, 14.0])


def test_tree_nan_mean() -> None:
    tree = BiasTree.compute(_edges([("a", "p"), ("b", "p")]), how="mean")
    out = aggregate_bias_with_tree(_leaves({"a": np.nan, "b": 4.0}), tree)
    np.testing.assert_allclose(out.values, [np.nan, 4.0, 4.0])


def test_wrapper() -> None:
    out = aggregate_bias(_leaves({"a": 2.0, "b": 4.0}), _edges([("a", "p"), ("b", "p")]))
    np.testing.assert_allclose(out.values, [2.0, 4.0, 3.0])


def test_missing_leaf_raises() -> None:
    tree = BiasTree.compute(_edges([("a", "p"), ("b", "p")]), how="mean")
    with pytest.raises(ValueError, match="missing"):
        aggregate_bias_with_tree(_leaves({"a": 2.0}), tree)
