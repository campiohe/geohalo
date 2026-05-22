"""BiasHierarchy.digest invariants."""

from geohalo.bias import BiasHierarchy


def test_digest_invariant_to_edge_order() -> None:
    a = BiasHierarchy.build([
        (("p",), ("c1",), 1.0),
        (("p",), ("c2",), 1.0),
    ])
    b = BiasHierarchy.build([
        (("p",), ("c2",), 1.0),
        (("p",), ("c1",), 1.0),
    ])
    assert a.digest == b.digest


def test_digest_changes_with_weight() -> None:
    a = BiasHierarchy.build([(("p",), ("c",), 1.0)])
    b = BiasHierarchy.build([(("p",), ("c",), 2.0)])
    assert a.digest != b.digest
