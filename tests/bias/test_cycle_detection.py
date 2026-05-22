"""Cycle detection in BiasHierarchy.build."""

import pytest

from geohalo.bias import BiasHierarchy


def test_simple_cycle_raises() -> None:
    with pytest.raises(ValueError, match="cycle"):
        BiasHierarchy.build([
            (("a",), ("b",), 1.0),
            (("b",), ("a",), 1.0),
        ])


def test_three_node_cycle_raises() -> None:
    with pytest.raises(ValueError, match="cycle"):
        BiasHierarchy.build([
            (("a",), ("b",), 1.0),
            (("b",), ("c",), 1.0),
            (("c",), ("a",), 1.0),
        ])
