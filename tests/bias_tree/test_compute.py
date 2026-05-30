import numpy as np
import pandas as pd
import pytest

from geohalo.bias_tree import BiasTree


def _edges(rows: list[tuple]) -> pd.DataFrame:
    has_w = len(rows[0]) == 3
    data = {"parent": [r[1] for r in rows]}
    if has_w:
        data["weight"] = [r[2] for r in rows]
    df = pd.DataFrame(data, index=[r[0] for r in rows])
    df.index.name = "child"
    return df


def test_mean_two_leaves() -> None:
    tree = BiasTree.compute(_edges([("a", "p"), ("b", "p")]), how="mean")
    assert list(tree.keys) == ["a", "b", "p"]
    np.testing.assert_allclose(tree.rollup_matrix.toarray(), [[1, 0], [0, 1], [0.5, 0.5]])


def test_mean_weighted() -> None:
    tree = BiasTree.compute(_edges([("a", "p", 1.0), ("b", "p", 3.0)]), weight_col="weight", how="mean")
    np.testing.assert_allclose(tree.rollup_matrix.toarray(), [[1, 0], [0, 1], [0.25, 0.75]])


def test_sum_weighted() -> None:
    tree = BiasTree.compute(_edges([("a", "p", 1.0), ("b", "p", 3.0)]), weight_col="weight", how="sum")
    np.testing.assert_allclose(tree.rollup_matrix.toarray(), [[1, 0], [0, 1], [1.0, 3.0]])
    assert tree.how == "sum"


def test_multi_level() -> None:
    tree = BiasTree.compute(
        _edges([("a", "p1"), ("b", "p1"), ("c", "p2"), ("p1", "root"), ("p2", "root")]),
        how="mean",
    )
    assert list(tree.keys) == ["a", "b", "c", "p1", "p2", "root"]
    np.testing.assert_allclose(
        tree.rollup_matrix.toarray(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0, 1.0], [0.25, 0.25, 0.5]],
    )


def test_duplicate_index_raises() -> None:
    edges = pd.DataFrame({"parent": ["p", "p"]}, index=["a", "a"])
    with pytest.raises(ValueError, match=r"unique|duplicate"):
        BiasTree.compute(edges)


def test_cycle_raises() -> None:
    with pytest.raises(ValueError, match=r"cycle|leaf"):
        BiasTree.compute(_edges([("a", "b"), ("b", "a")]))


def test_int_keys() -> None:
    tree = BiasTree.compute(_edges([(1, 10), (2, 10)]), how="mean")
    assert list(tree.keys) == [1, 2, 10]


def test_tuple_keys() -> None:
    edges = pd.DataFrame(
        {"parent": [("BR",), ("BR",)]},
        index=pd.Index([("BR", "SP"), ("BR", "RJ")]),
    )
    tree = BiasTree.compute(edges, how="mean")
    assert list(tree.keys) == [("BR", "RJ"), ("BR", "SP"), ("BR",)]


def test_integer_dtype_weight_column() -> None:
    # np.int64 is not a Python `int`; the validation must still accept it.
    edges = pd.DataFrame({"parent": ["p", "p"], "weight": np.array([1, 3])}, index=["a", "b"])
    edges.index.name = "child"
    tree = BiasTree.compute(edges, weight_col="weight", how="mean")
    np.testing.assert_allclose(tree.rollup_matrix.toarray(), [[1, 0], [0, 1], [0.25, 0.75]])


def test_nonpositive_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        BiasTree.compute(_edges([("a", "p", 1.0), ("b", "p", 0.0)]), weight_col="weight")


def test_digest_changes_with_how() -> None:
    edges = _edges([("a", "p", 1.0), ("b", "p", 3.0)])
    assert (
        BiasTree.compute(edges, weight_col="weight", how="mean").digest
        != BiasTree.compute(edges, weight_col="weight", how="sum").digest
    )
