"""BiasHierarchy.build input validation."""

import pytest

from geohalo.bias import BiasHierarchy


def test_empty_edges_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        BiasHierarchy.build([])


def test_zero_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        BiasHierarchy.build([(("p",), ("c",), 0.0)])


def test_negative_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        BiasHierarchy.build([(("p",), ("c",), -1.0)])


def test_inf_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        BiasHierarchy.build([(("p",), ("c",), float("inf"))])


def test_nan_weight_raises() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        BiasHierarchy.build([(("p",), ("c",), float("nan"))])


def test_non_tuple_parent_raises() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        BiasHierarchy.build([("p", ("c",), 1.0)])  # type: ignore[list-item]


def test_non_tuple_child_raises() -> None:
    with pytest.raises(TypeError, match="must be a tuple"):
        BiasHierarchy.build([(("p",), "c", 1.0)])  # type: ignore[list-item]


def test_arity_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length 1, expected 2"):
        BiasHierarchy.build(
            [(("p",), ("c",), 1.0)],
            key_names=("country", "region"),
        )


def test_duplicate_edge_raises() -> None:
    with pytest.raises(ValueError, match="duplicate edge"):
        BiasHierarchy.build([
            (("p",), ("c",), 1.0),
            (("p",), ("c",), 2.0),
        ])


def test_empty_key_names_raises() -> None:
    with pytest.raises(ValueError, match="at least one name"):
        BiasHierarchy.build(
            [(("p",), ("c",), 1.0)],
            key_names=(),
        )
