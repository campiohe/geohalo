"""Composed-matrix correctness: leaf rows identity, parent rows weighted."""

import numpy as np

from geohalo.bias import BiasHierarchy


def test_leaf_rows_are_identity() -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    mat = h.matrix.toarray()
    leaf_indices = [h.nodes.index(k) for k in h.leaf_keys]
    for j, leaf_idx in enumerate(leaf_indices):
        row = mat[leaf_idx]
        expected = np.zeros(len(h.leaf_keys))
        expected[j] = 1.0
        np.testing.assert_array_equal(row, expected)


def test_parent_row_is_weighted_normalized_average() -> None:
    h = BiasHierarchy.build([
        (("p",), ("c1",), 2.0),
        (("p",), ("c2",), 3.0),
    ])
    mat = h.matrix.toarray()
    p_idx = h.nodes.index(("p",))
    # leaf_keys is sorted -> [("c1",), ("c2",)]
    np.testing.assert_allclose(mat[p_idx], [2.0 / 5.0, 3.0 / 5.0])


def test_multi_level_chain() -> None:
    """Great-grandparent -> grandparent -> parent -> leaves."""
    h = BiasHierarchy.build([
        (("gg",), ("g",), 1.0),
        (("g",), ("p1",), 1.0),
        (("g",), ("p2",), 1.0),
        (("p1",), ("l1",), 1.0),
        (("p1",), ("l2",), 1.0),
        (("p2",), ("l3",), 1.0),
    ])
    assert h.leaf_keys == [("l1",), ("l2",), ("l3",)]
    mat = h.matrix.toarray()
    gg_idx = h.nodes.index(("gg",))
    # gg = g; g = (p1 + p2) / 2; p1 = (l1+l2)/2; p2 = l3
    # so gg = ((l1+l2)/2 + l3) / 2 = [0.25, 0.25, 0.5]
    np.testing.assert_allclose(mat[gg_idx], [0.25, 0.25, 0.5])
