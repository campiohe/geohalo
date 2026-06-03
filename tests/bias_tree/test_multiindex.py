import pandas as pd
import xarray as xr

import geohalo as ghl
from geohalo.bias_tree import BiasTree


def _mi_edges() -> pd.DataFrame:
    """Same-arity (2-level) hierarchy: (scenario, region) leaves -> (scenario, ALL)."""
    return pd.DataFrame(
        {"parent": [("x", "ALL"), ("x", "ALL"), ("y", "ALL")]},
        index=pd.MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "c")], names=["scenario", "region"],
        ),
    )


def test_keys_are_multiindex_with_names() -> None:
    tree = BiasTree.compute(_mi_edges())
    assert isinstance(tree.keys, pd.MultiIndex)
    assert tree.keys.names == ["scenario", "region"]
    assert isinstance(tree.leaf_keys, pd.MultiIndex)
    assert list(tree.leaf_keys) == [("x", "a"), ("x", "b"), ("y", "c")]
    assert ("x", "ALL") in list(tree.keys)
    assert ("y", "ALL") in list(tree.keys)


def test_aggregate_bias_roundtrip_preserves_multiindex() -> None:
    mi = pd.MultiIndex.from_tuples(
        [("x", "a"), ("x", "b"), ("y", "c")], names=["scenario", "region"],
    )
    reduced = xr.DataArray([10.0, 20.0, 30.0], dims="geom", coords={"geom": mi})

    out = ghl.aggregate_bias(reduced, _mi_edges())

    assert isinstance(out.indexes["geom"], pd.MultiIndex)
    assert out.indexes["geom"].names == ["scenario", "region"]
    # (x, ALL) = mean(10, 20) = 15 ; (y, ALL) = 30
    assert float(out.sel(geom=("x", "ALL"))) == 15.0
    assert float(out.sel(geom=("y", "ALL"))) == 30.0
    # the levels remain selectable: selecting a scenario collapses to its regions
    assert set(out.sel(scenario="x").indexes["region"]) == {"a", "b", "ALL"}


def test_varying_arity_multiindex_degrades_to_object_index() -> None:
    # Shorter-arity parents (a real geographic hierarchy) can't form one MultiIndex;
    # it degrades to a flat object Index of tuples rather than raising or NaN-padding.
    edges = pd.DataFrame(
        {"parent": [("x",), ("x",)]},  # 1-tuple parents under a 2-level index
        index=pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")], names=["scenario", "region"]),
    )
    tree = BiasTree.compute(edges)
    assert not isinstance(tree.keys, pd.MultiIndex)
    assert set(tree.keys) == {("x", "a"), ("x", "b"), ("x",)}


def test_scalar_parent_under_multiindex_degrades() -> None:
    # A non-tuple parent mixed with tuple leaves is also varying-arity: degrade, don't raise.
    edges = pd.DataFrame(
        {"parent": ["x", "x"]},
        index=pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")], names=["scenario", "region"]),
    )
    tree = BiasTree.compute(edges)
    assert not isinstance(tree.keys, pd.MultiIndex)
    assert set(tree.keys) == {("x", "a"), ("x", "b"), "x"}


def test_digest_changes_with_level_names() -> None:
    e1 = _mi_edges()
    e2 = e1.copy()
    e2.index = e2.index.set_names(["scn", "reg"])
    assert BiasTree.compute(e1).digest != BiasTree.compute(e2).digest


def test_flat_scalar_keys_stay_plain_index() -> None:
    edges = pd.DataFrame({"parent": ["p", "p"]}, index=["a", "b"])
    edges.index.name = "child"
    tree = BiasTree.compute(edges)
    assert not isinstance(tree.keys, pd.MultiIndex)
    assert list(tree.keys) == ["a", "b", "p"]


def test_varying_arity_object_index_still_supported() -> None:
    # Plain object Index of varying-arity tuples (NOT a MultiIndex) keeps working.
    edges = pd.DataFrame(
        {"parent": [("BR",), ("BR",)]},
        index=pd.Index([("BR", "SP"), ("BR", "RJ")]),
    )
    tree = BiasTree.compute(edges)
    assert not isinstance(tree.keys, pd.MultiIndex)
    assert list(tree.keys) == [("BR", "RJ"), ("BR", "SP"), ("BR",)]
